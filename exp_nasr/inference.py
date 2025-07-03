from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from exp_nasr.utils import AttrDict
from exp_nasr.dataset import amp_pha_specturm, load_wav
from exp_nasr.models import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np
import torchaudio.functional as F_audio

h = None
device = None


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

def greedy_decode(log_probs, vocab):
    """
    对 CTC log_probs 进行贪婪解码，生成 token 索引和文本。
    log_probs: [T, B, V]
    vocab: 索引到字符的映射表
    返回: (token 索引列表, 文本)
    """
    pred_ids = torch.argmax(log_probs, dim=-1)  # [T, B]
    pred_ids = pred_ids.permute(1, 0)  # [B, T]
    indices = []
    text = []
    prev = None
    for idx in pred_ids[0].cpu().numpy():  # 假设批量大小为 1
        if idx != 0 and idx != prev:  # 0 是 <blank>
            indices.append(int(idx))
            text.append(vocab[idx])
        prev = idx
    return indices, ''.join(text)


def inference(h):
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])

    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    os.makedirs(h.test_wav_output_dir, exist_ok=True)
    os.makedirs(h.test_label_output_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    vocab = {
        0: '<blank>', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
        8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
        16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
        24: 'x', 25: 'y', 26: 'z', 27:" "
        }
    
    predicted_labels = {}

    with torch.no_grad():
        for i, filename in enumerate(filelist):
            raw_wav, _ = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            raw_wav = torch.FloatTensor(raw_wav).to(device)
            raw_wav = raw_wav.unsqueeze(0)
            logamp, pha, _, _, = amp_pha_specturm(raw_wav, h.n_fft, h.hop_size, h.win_size)
            latent,_,_,_,_,log_probs = encoder(logamp, pha)
            indices, pred_text = greedy_decode(log_probs, vocab)
            logamp_g, pha_g, _, _, y_g = decoder(latent)

            file_id = filename.split('.')[0]  # 例如 p364_060
            predicted_labels[file_id] = {
                "indices": str(indices),  # 格式如 "[23,8,25,...]"
                "target_length": len(indices),
                "text": pred_text
            }
            latent = latent.squeeze()
            audio = y_g.squeeze()
            logamp = logamp_g.squeeze()
            pha = pha_g.squeeze()
            latent = latent.cpu().numpy()
            audio = audio.cpu().numpy()
            logamp = logamp.cpu().numpy()
            pha = pha.cpu().numpy()

            sf.write(os.path.join(h.test_wav_output_dir, filename.split('.')[0]+'.wav'), audio, h.sampling_rate,'PCM_16')
    
    with open(os.path.join(h.test_label_output_dir, 'predicted_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(predicted_labels, f, ensure_ascii=False, indent=4)

def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_nasr/config.json'

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

