from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from exp_bwe.utils import AttrDict
from exp_bwe.dataset import amp_pha_spectrum, load_wav
from exp_bwe.models import Encoder, Decoder, BWE
import soundfile as sf
import librosa
import numpy as np
import time  
import logging  
import math
import torchaudio.functional as F_audio

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
    bwe =   BWE(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])
    state_dict_bwe = load_checkpoint(h.checkpoint_file_load_BWE, device)
    bwe.load_state_dict(state_dict_bwe['bwe'])

    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    codebook_size = 1024

    total_duration = 0.0  # 总音频时长
    total_bits = 0  # 总比特数

    with torch.no_grad():
        start_time = time.time()
        for i, filename in enumerate(filelist):

            raw_wav, sr = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration
            raw_wav = torch.FloatTensor(raw_wav).to(device)
            wav_nb = F_audio.resample(raw_wav, orig_freq=h.sampling_rate, new_freq=h.low_sampling_rate)
            logamp, pha, _, _ = amp_pha_spectrum(wav_nb.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
            latent,codes,_,_ = encoder(logamp, pha)
            y_g = decoder(latent)
            #扩充
            logamp_nb_g, pha_nb_g, rea_nb_g, imag_nb_g = amp_pha_spectrum(y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            logamp_wb_g, pha_wb_g = bwe(logamp_nb_g, pha_nb_g)
            rea_wb_g = torch.exp(logamp_wb_g)*torch.cos(pha_wb_g)
            imag_wb_g = torch.exp(logamp_wb_g)*torch.sin(pha_wb_g)
            spec_wb_g = torch.complex(rea_wb_g, imag_wb_g)
            y_wb_g = torch.istft(spec_wb_g, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=torch.hann_window(h.win_size).to(latent.device), center=True) 

            n_codebooks = codes.shape[1]  # 码本数量
            T = codes.shape[2]  # 时间步数
            bits_per_index = math.ceil(math.log2(codebook_size))  # 每个索引的比特数
            bits_per_frame = n_codebooks * bits_per_index
            file_bits = bits_per_frame * T
            total_bits += file_bits
            latent = latent.squeeze()
            audio = y_wb_g.squeeze()
            audio = audio.cpu().numpy()

            sf.write(os.path.join(h.test_wav_output_dir, filename.split('.')[0]+'.wav'), audio, h.sampling_rate,'PCM_16')
        total_inference_time = time.time() - start_time
    bitrate_kbps = (total_bits / total_duration) / 1000
    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")
    logging.info(f"Total duration: {total_duration:.3f} seconds, Total bits: {total_bits}, Bitrate: {bitrate_kbps:.3f} kbps")


def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_bwe/config.json'

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