from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from exp_id.utils import AttrDict
from exp_id.dataset import amp_pha_specturm, load_wav
from exp_id.models import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np
import time  
import logging  
import math  
import csv

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

    codebook_size = 1024

    total_duration = 0.0  # 总音频时长
    total_bits = 0  # 总比特数

    all_speakers = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p236', 'p237', 
        'p238', 'p239', 'p240', 'p241', 'p243', 'p244', 'p245', 'p246', 'p247', 'p248', 'p249', 'p250', 'p251', 'p252', 
        'p253', 'p254', 'p255', 'p256', 'p257', 'p258', 'p259', 'p260', 'p261', 'p262', 'p263', 'p264', 'p265', 'p266', 
        'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p276', 'p277', 'p278', 'p279', 'p280',
        'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294', 'p295', 'p297', 'p298',
        'p299', 'p300', 'p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p310', 'p311', 'p312', 'p313',
        'p314', 'p315', 'p316', 'p317', 'p318', 'p323', 'p326', 'p329', 'p330', 'p333', 'p334', 'p335', 'p336', 'p339', 
        'p340', 'p341', 'p343', 'p345', 'p347', 'p351', 'p360', 'p361', 'p362', 'p363', 'p364', 'p374', 'p376']
    
    spk2id = {spk: i for i, spk in enumerate(all_speakers)}
    id2spk = {i: spk for i, spk in enumerate(all_speakers)}
    results = []

    with torch.no_grad():
        start_time = time.time()
        for i, filename in enumerate(filelist):

            raw_wav, sr = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration
            raw_wav = torch.FloatTensor(raw_wav).to(device)
            logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
            latent,codes,_,_,logits,_ = encoder(logamp, pha, None)
            logamp_g, pha_g, _, _, y_g = decoder(latent)

            predict_id = torch.argmax(logits, dim=-1).item()
            predict_spk = id2spk[predict_id]
            true_spk = filename.split("_")[0]
            results.append([filename, true_spk, predict_spk])

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

    csv_path = os.path.join(h.test_speaker_output_dir, "speaker_predictions.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
        
    bitrate_kbps = (total_bits / total_duration) / 1000
    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")
    logging.info(f"Total duration: {total_duration:.3f} seconds, Total bits: {total_bits}, Bitrate: {bitrate_kbps:.3f} kbps")


def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_id/config.json'

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

