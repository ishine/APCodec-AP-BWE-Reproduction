from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from exp_750.utils import AttrDict
from exp_750.dataset import amp_pha_specturm, load_wav
from exp_750.models import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np
import time  
import logging  
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re

h = None
device = None
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # 同时输出到终端
)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(h):

    #额外的参数
    n_quantizers = 3
    codebook_size = 1024
    seq_len = 252
    #"/mnt/nvme_share/srt30/Datasets/VCTK-0.92/wav8/origin/train/p225_005.wav"
    #"/mnt/nvme_share/srt30/Datasets/VCTK-0.92/wav8/origin/test/p360_001.wav"
    file_path = "/mnt/nvme_share/srt30/Datasets/VCTK-0.92/wav8/origin/test/p360_021.wav"
    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])
    encoder.eval()
    decoder.eval()

    print("Loading LoRA repair model...")
    tokenizer = AutoTokenizer.from_pretrained(h.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    new_special_tokens = {"additional_special_tokens": [f"<token_{i}>" for i in range(3072)] + ["<missing>"]}
    tokenizer.add_special_tokens(new_special_tokens)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        h.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, h.adapter_path)
    model.eval()
    
    print("start inference...")
    with torch.no_grad():
        start_time = time.time()
        #压缩，生成token id
        raw_wav, sr = librosa.load(file_path, sr=h.sampling_rate, mono=True)
        raw_wav = torch.FloatTensor(raw_wav).to(device)
        logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
        latent,original_codes,_,_ = encoder(logamp, pha)
        print(f"original_codes.shape: {original_codes.shape}")
        codes_np = np.squeeze(original_codes.cpu().numpy(), axis=0)
        offset = np.array([i*codebook_size for i in range(n_quantizers)])[:,np.newaxis]
        codes_with_offset = (codes_np + offset).T
        original_tokens= codes_with_offset.flatten().astype(int)
        tokens_len = len(original_tokens)
        print(f"总token数：{tokens_len}")

        start_repair_time = time.time()

        repaired_tokens = []
        for i in range(0, 1+tokens_len//seq_len):
            cur_seq = original_tokens[i*seq_len: min((i+1)*seq_len, tokens_len)]
            frames = cur_seq.reshape(-1,n_quantizers)
            input_tokens = []
            for frame in frames:
                kept = [f"<token_{t}>" for t in frame[:2]]
                input_tokens.extend(kept)
                input_tokens.append("<missing>")

            input_str = " ".join(input_tokens)
            user_text = (
                    "<start_of_turn>user\n"
                    "Repair the text by filling in the <missing> tokens:\n"
                    f"{input_str}<end_of_turn>\n"
                    "<start_of_turn>model\n"
                )
            
            inputs = tokenizer(user_text, return_tensors="pt").to(device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )

            input_ids_len = inputs.input_ids.shape[1]
            newly_generated_ids = generated_ids[0][input_ids_len:]
            generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=False)
            print(f"newly generated text_{i}: {generated_text}")
            repaired_seq = [int(m) for m in re.findall(r"<token_(\d+)>", generated_text)]
            repaired_tokens.extend(repaired_seq)

        end_repair_time = time.time()
        print(f"repair generation finished in {end_repair_time - start_repair_time} seconds")
        print(f"Repaired token sequence length: {len(repaired_tokens)}")

        if len(repaired_tokens) != len(original_tokens):
            print("Warning: Repaired token sequence differs from original, truncating...")
            repaired_tokens = np.resize(repaired_tokens, original_tokens.shape)

        repaired_tokens = np.array(repaired_tokens)
        #计算accuracy
        comparison = original_tokens == repaired_tokens
        accuracy = np.sum(comparison) / len(original_tokens)

        original_frames = original_tokens.reshape(-1, 3)
        repaired_frames = repaired_tokens.reshape(-1, 3)
        in_comparison = original_frames[:,:2] == repaired_frames[:,:2]
        in_accuracy = np.sum(in_comparison) / (len(original_tokens) * 2 / 3)
        out_comparison = original_frames[:,2] == repaired_frames[:,2]
        out_accuracy = np.sum(out_comparison) / (len(original_tokens) / 3)
        print(f"accuracy: {accuracy:.3f}, in_accuracy: {in_accuracy:.3f}, out_accuracy: {out_accuracy:.3f}")

        repaired_tokens = repaired_tokens.reshape(-1, n_quantizers).T
        i_offset = np.array([i*codebook_size for i in range(n_quantizers)])[:,np.newaxis]
        repaired_codes = torch.tensor(repaired_tokens - i_offset, dtype=torch.long, device=device).unsqueeze(0)
        original_latent,_,_ = encoder.quantizer.from_codes(original_codes)
        repaired_latent,_,_ = encoder.quantizer.from_codes(repaired_codes)

        #重建为音频
        _, _, _, _, y_original = decoder(original_latent)
        original_audio = y_original.squeeze()
        original_audio = original_audio.cpu().numpy()

        _, _, _, _, y_repaired = decoder(repaired_latent)
        repaired_audio = y_repaired.squeeze()
        repaired_audio = repaired_audio.cpu().numpy()

        sf.write(os.path.join(h.test_wav_output_dir, "original.wav"), original_audio, h.sampling_rate,'PCM_16')
        sf.write(os.path.join(h.test_wav_output_dir, "repaired.wav"), repaired_audio, h.sampling_rate,'PCM_16')

def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/config.json'

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(h)


if __name__ == '__main__':
    main()