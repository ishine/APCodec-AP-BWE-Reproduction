import glob
import os
import json
import torch
import random
import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from exp_fsq.utils import AttrDict
from exp_fsq.dataset import amp_pha_specturm
from exp_fsq.models import Encoder, Decoder
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logging.info(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Complete.")
    return checkpoint_dict

def drop_token(codes, drop_prob=0.1):
    result = []
    for i in range(0, len(codes), 4):
        if random.random() >= drop_prob:
            result.extend(codes[i:i+4])
    return result

def generate_tokens(h,filelist,encoder,tokenizer,device):

    tokenset = []

    with torch.no_grad():
        for filename in tqdm(filelist, desc="generating token", unit="file"):
            audio, sr = librosa.load(filename, sr=h.sampling_rate, mono=True)
            audio = torch.FloatTensor(audio).to(device)
            audio = audio.unsqueeze(0)
            logamp, pha, _, _ = amp_pha_specturm(audio, h.n_fft, h.hop_size, h.win_size)
            channel = logamp.size(2)
            segment_size = 400
            num_segments = channel // segment_size + 1
            for i in range(num_segments):
                start = i * segment_size
                end = min(start + segment_size, channel)
                if end - start < 7:  # 跳过过短的片段
                    continue
                amp_segment = logamp[:, :, start:end]
                pha_segment = pha[:, :, start:end]
                _, codes, _, _ = encoder(amp_segment, pha_segment)
                codes = codes.squeeze(0).cpu().numpy() #[n_quantizers, T]
                codes = codes.transpose(1, 0).flatten().tolist() #[n_quantizers*T]
                dropped_codes = drop_token(codes, h.drop_prob)
                gt_str = ' '.join(str(token) for token in codes)
                gt_token_len = len(tokenizer(gt_str, add_special_tokens=False)["input_ids"])
                tokenset.append({
                    "dropped_tokens": dropped_codes,
                    "original_len": len(codes),
                    "gt_token_len": gt_token_len,
                    "filename": os.path.basename(filename),
                })

    return tokenset

def prediction(tokenset, tokenizer,instruction,model):

    file_segments = {}
    
    for token in tqdm(tokenset, desc="predict token", unit="segment"):
        filename = token['filename']
        original_len = token['original_len']
        dropped_str = ' '.join(map(str, token['dropped_tokens']))
        gt_token_len = token['gt_token_len']
        logging.info(f"Processing {filename}, original_len:{original_len}, ground truth token length: {gt_token_len}")
        chat = [{"role": "user", "content": f"{instruction}\n{dropped_str}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        print("predicting tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gt_token_len+5,
                do_sample=False,  # 贪婪解码
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[-1]
        predicted_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        try:
            predicted_tokens = list(map(int, predicted_text.strip().split()))
        except (ValueError, TypeError) as e:
            predicted_tokens = token['dropped_tokens']
            
        if len(predicted_tokens) > original_len:
            predicted_tokens = predicted_tokens[:original_len]

        if filename not in file_segments:
            file_segments[filename] = []
        file_segments[filename].append((predicted_tokens))

    return file_segments

def repair_audio(tokens, encoder, decoder, device):
    n_quantizers = 4
    total_length = len(tokens)
    t = total_length // n_quantizers
    codes = np.array(tokens).reshape(t, n_quantizers).transpose(1,0)
    codes = torch.tensor(codes, device=device).unsqueeze(0)
    with torch.no_grad():
        latent, _, _ = encoder.quantizer.from_codes(codes)
        logamp_g, pha_g, _, _, y_g = decoder(latent)
        audio = y_g.squeeze()
        audio = audio.cpu().numpy()
    return audio

def main():
    print('Initializing Inference Process..')

    #Config
    model_path = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
    adapter_path = "/mnt/nvme_share/srt30/checkpoint/exp_new_llm/checkpoint-280000"
    instruction_prompt = "Repair the following token sequence:"
    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_llm/config.json'
    with open(config_file, 'r') as f:
        json_config = json.load(f)
    h = AttrDict(json_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])
    encoder.eval()
    decoder.eval()
    logging.info(f"Loaded encoder and decoder models.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = bnb_config,
        device_map = {"":0},
        trust_remote_code = True
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = model.to("cuda:0")

    filelist = sorted(glob.glob(os.path.join(h.test_input_wavs_dir, '*.wav')))
    tokenset = generate_tokens(h, filelist,encoder,tokenizer,device)
    file_segments = prediction(tokenset, tokenizer, instruction_prompt, model)
    for filename in tqdm(file_segments, desc="repair audio", unit="file"):
        segments = file_segments[filename]
        total_token = []
        for tokens in segments:
            total_token.extend(tokens)
        repaired_audio = repair_audio(total_token, encoder, decoder, device)
        output_path = os.path.join(h.test_wav_output_dir, f"{filename}")
        sf.write(output_path, repaired_audio, h.sampling_rate)
    
    logging.info(f"Inference complete. Repaired audio saved to {output_path}")

if __name__ == "__main__":
    main()