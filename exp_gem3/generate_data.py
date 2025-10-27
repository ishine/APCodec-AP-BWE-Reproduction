from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import json
import torch
import random
import librosa
from exp_750.utils import AttrDict
from exp_750.dataset import amp_pha_specturm
from exp_750.models import Encoder
import numpy as np
from tqdm import tqdm
import logging 

# 配置日志
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

def generate_dataset(h, device):
    encoder = Encoder(h).to(device)
    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    encoder.eval()

    training_dir = h.training_input_wavs_dir
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"Training directory not found at: {training_dir}")
    filelist = sorted(os.listdir(training_dir))

    print(f"Generating token corpus from {len(filelist)} audio files...")
    missing_token = "<missing>"
    ft_data = []
    codebook_size = 1024
    n_quantizers = 3

    with torch.no_grad():
        for filename in tqdm(filelist, desc="Processing audio files"):
            try:
                raw_wav, sr = librosa.load(os.path.join(training_dir, filename), sr=h.sampling_rate, mono=True)
                raw_wav = torch.FloatTensor(raw_wav).to(device)
                logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
                _, codes, _, _ = encoder(logamp, pha)
                codes_np = np.squeeze(codes.cpu().numpy(), axis=0)
                offset = np.array([i*codebook_size for i in range(n_quantizers)])[:,np.newaxis]
                codes_with_offset = (codes_np + offset).T
                tokens_flat = codes_with_offset.flatten().astype(int).tolist()
                num_tokens = len(tokens_flat)
                if num_tokens > 950:
                    logging.info(f"Skipping file {filename} because it has {num_tokens} tokens (> 950).")
                    continue

                input_parts = []
                for i in range(0, num_tokens, 3):
                    frame = tokens_flat[i:i+3]
                    kept = [f"<token_{t}>" for t in frame[:2]]
                    input_parts.extend(kept)
                    input_parts.append(missing_token)

                input_str = " ".join(input_parts)
                output_str = " ".join([f"<token_{t}>" for t in tokens_flat])
                ft_data.append({
                    "input": input_str,
                    "output": output_str
                })
            except Exception as e:
                logging.warning(f"Skipping file {filename} due to an error: {e}")


    with open(h.training_data_json, 'w', encoding='utf-8') as f:
        json.dump(ft_data, f, indent=2, ensure_ascii=False)

    logging.info(f"\nDataset generation complete. Total samples: {len(ft_data)}")
    logging.info(f"Token dataset saved to '{h.training_data_json}'")

if __name__ == '__main__':
    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/config.json'
    with open(config_file) as f:
        data = f.read()
    
    h = AttrDict(json.loads(data))
    torch.manual_seed(h.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_dataset(h, device)