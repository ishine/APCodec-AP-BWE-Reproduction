from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from exp_300.utils import AttrDict
from exp_300.dataset import amp_pha_specturm, load_wav
from exp_300.models import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np
import time  
import logging  
import math  

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
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])

    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    codebook_size = 4096

    total_duration = 0.0  # 总音频时长
    total_bits = 0  # 总比特数

    with torch.no_grad():
        start_time = time.time()
        for i, filename in enumerate(filelist):

            raw_wav, sr = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration
            raw_wav = torch.FloatTensor(raw_wav).to(device)
            logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
            
            latent,codes,_,_ = encoder(logamp, pha)
            logamp_g, pha_g, _, _, y_g = decoder(latent)
            n_codebooks = codes.shape[1]  # 码本数量
            T = codes.shape[2]  # 时间步数
            bits_per_index = math.ceil(math.log2(codebook_size))  # 每个索引的比特数
            bits_per_frame = n_codebooks * bits_per_index
            file_bits = bits_per_frame * T
            total_bits += file_bits
            latent = latent.squeeze()
            audio = y_g.squeeze()
            logamp = logamp_g.squeeze()
            pha = pha_g.squeeze()
            latent = latent.cpu().numpy()
            audio = audio.cpu().numpy()
            logamp = logamp.cpu().numpy()
            pha = pha.cpu().numpy()

            sf.write(os.path.join(h.test_wav_output_dir, filename.split('.')[0]+'.wav'), audio, h.sampling_rate,'PCM_16')
        total_inference_time = time.time() - start_time
    bitrate_kbps = (total_bits / total_duration) / 1000
    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")
    logging.info(f"Total duration: {total_duration:.3f} seconds, Total bits: {total_bits}, Bitrate: {bitrate_kbps:.3f} kbps")


def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_300/config.json'

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

