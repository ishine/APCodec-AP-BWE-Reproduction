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
    handlers=[logging.StreamHandler()]
)

def load_checkpoint(filepath, device):
    """Loads a PyTorch model checkpoint."""
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    """Scans for the latest checkpoint file in a directory."""
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(h):

    n_quantizers = 3
    codebook_size = 1024
    chunk_len = 126
    llm_batch_size = 4

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

    total_tokens = n_quantizers * codebook_size
    new_special_tokens = {"additional_special_tokens": [f"<token_{i}>" for i in range(total_tokens)] + ["<missing>"]}
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

    os.makedirs(h.total_test_wav_output_dir, exist_ok=True)
    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    total_duration = 0.0
    total_bits = 0
    total_processing_time = 0.0
    total_tokens = 0
    failed_files = []
    success = 0
    correct_num = 0
    in_correct_num = 0
    out_correct_num = 0

    print("Starting inference on the test directory...")
    with torch.no_grad():
        for filename in tqdm(filelist, desc="Processing test dataset"):
            file_start_time = time.time()
            input_file_path = os.path.join(h.test_input_wavs_dir, filename)

            raw_wav, sr = librosa.load(input_file_path, sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration
            raw_wav = torch.FloatTensor(raw_wav).to(device)

            logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
            latent, original_codes, _, _ = encoder(logamp, pha)
            
            codes_np = np.squeeze(original_codes.cpu().numpy(), axis=0)
            offset = np.array([i * codebook_size for i in range(n_quantizers)])[:, np.newaxis]
            codes_with_offset = (codes_np + offset).T
            original_tokens = codes_with_offset.flatten().astype(int)
            tokens_len = len(original_tokens)

            T = original_codes.shape[2]
            bits_per_index = math.ceil(math.log2(codebook_size))
            bits_per_frame = bits_per_index * 2
            file_bits = bits_per_frame * T
            total_bits += file_bits

            all_prompts = []
            num_chunks = math.ceil(tokens_len / chunk_len)
            
            for i in range(num_chunks):
                cur_chunk = original_tokens[i * chunk_len : min((i + 1) * chunk_len, tokens_len)]
                if len(cur_chunk) == 0:
                    continue
                frames = cur_chunk.reshape(-1, n_quantizers)
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
                all_prompts.append(user_text)

            repaired_tokens = []
            
            prompt_iterator = (
                tqdm(range(0, len(all_prompts), llm_batch_size), desc=f"Repairing {filename}", leave=False)
                if len(all_prompts) > 1 else range(0, len(all_prompts), llm_batch_size)
            )

            for i in prompt_iterator:
                batch_prompts = all_prompts[i : i + llm_batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
                input_ids_len = inputs.input_ids.shape[1]
                for j in range(generated_ids.shape[0]):
                    newly_generated_ids = generated_ids[j][input_ids_len:]
                    generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=False)
                    repaired_chunk = [int(m) for m in re.findall(r"<token_(\d+)>", generated_text)]
                    repaired_tokens.extend(repaired_chunk)

            if len(repaired_tokens) != tokens_len:
                logging.warning(f"File {filename} failed: Repaired token count ({len(repaired_tokens)}) does not match original ({tokens_len}). Skipping this file.")
                failed_files.append(filename)
                total_processing_time += (time.time() - file_start_time)
                continue
            
            repaired_tokens = np.array(repaired_tokens)

            #计算accuracy
            comparison = original_tokens == repaired_tokens
            correct_num += np.sum(comparison)
            total_tokens += len(original_tokens)

            original_frames = original_tokens.reshape(-1, 3)
            repaired_frames = repaired_tokens.reshape(-1, 3)
            in_comparison = original_frames[:,:2] == repaired_frames[:,:2]
            in_correct_num += np.sum(in_comparison)
            in_accuracy = np.sum(in_comparison) / (len(original_tokens) * 2 / 3)
            out_comparison = original_frames[:,2] == repaired_frames[:,2]
            out_correct_num += np.sum(out_comparison)

            repaired_tokens = repaired_tokens.reshape(-1, n_quantizers).T
            i_offset = np.array([i * codebook_size for i in range(n_quantizers)])[:, np.newaxis]
            repaired_codes = torch.tensor(repaired_tokens - i_offset, dtype=torch.long, device=device).unsqueeze(0)

            repaired_latent, _, _ = encoder.quantizer.from_codes(repaired_codes)
            _, _, _, _, y_repaired = decoder(repaired_latent)
            repaired_audio = y_repaired.squeeze().cpu().numpy()

            sf.write(os.path.join(h.total_test_wav_output_dir, filename.split('.')[0]+'.wav'), repaired_audio, h.sampling_rate,'PCM_16')
            success += 1
            total_processing_time += (time.time() - file_start_time)

        bitrate_kbps = (total_bits / total_duration) / 1000 
        accuracy = correct_num / total_tokens
        in_accuracy = in_correct_num / (total_tokens / 3 * 2)
        out_accuracy = out_correct_num / (total_tokens / 3)

        logging.info("-------------------- Inference Summary --------------------")
        logging.info(f"Total files processed: {len(filelist)}")
        logging.info(f"Total files successful: {success}")
        logging.info(f"Total files failed (length mismatch or error): {len(failed_files)}")
        if failed_files:
            logging.warning(f"Failed files list: {failed_files}")
        logging.info(f"Total processing time: {total_processing_time:.3f} seconds")
        logging.info(f"Total audio duration processed: {total_duration:.3f} seconds")
        logging.info(f"Total bits transmitted: {total_bits}")
        logging.info(f"Calculated Bitrate: {bitrate_kbps:.4f} kbps")
        logging.info(f"accuracy: {accuracy:.5f}")
        logging.info(f"in_accuracy: {in_accuracy:.5f}")
        logging.info(f"out_accuracy: {out_accuracy:.5f}")
        logging.info("---------------------------------------------------------")


def main():
    print('Initializing Merged Inference Process..')

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