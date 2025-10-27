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
import torch.nn as nn

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

class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ffn(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class AudioTransformer(nn.Module):
    def __init__(self, vocab_size=3073, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.2, max_len=1155, pad_id=4096):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad_mask = (x == self.pad_id)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits #[B, T, vocab_size]


def inference(h):
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    model = AudioTransformer(vocab_size=3073, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=1155, pad_id=3072).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])
    state_dict_model = load_checkpoint(h.checkpoint_file_load_prob, device)
    model.load_state_dict(state_dict_model['model'])

    filelist = sorted(os.listdir(h.test_input_wavs_dir))

    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()
    model.eval()

    max_len = 1155
    codebook_size = 1024
    n_quantizers = 3
    total_duration = 0.0  # 总音频时长
    total_bits = 0  # 总比特数
    total_tokens = 0
    total_correct = 0
    accuracy = 0.0

    with torch.no_grad():
        start_time = time.time()
        for j, filename in enumerate(filelist):
            raw_wav, sr = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            duration = len(raw_wav) / sr
            total_duration += duration
            raw_wav = torch.FloatTensor(raw_wav).to(device)
            logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)

            original_latent ,original_codes,_,_ = encoder(logamp, pha)
            codes_np = np.squeeze(original_codes.cpu().numpy(), axis=0)
            offset = np.array([i*codebook_size for i in range(n_quantizers)])[:,np.newaxis]
            codes_with_offset = (codes_np + offset).T
            original_tokens= codes_with_offset.flatten().astype(int)

            repaired_tokens = []
            context = []
            for i in range(0, len(original_tokens)//n_quantizers):
                current_frame = original_tokens[i*n_quantizers:(i+1)*n_quantizers].tolist()
                repaired_tokens.extend(current_frame[:2])
                context.extend(current_frame[:2])

                if len(context) > max_len:
                    context = context[-max_len:]
                input_ids = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(input_ids)
                next_token = torch.argmax(logits[0, -1, :]).item()

                repaired_tokens.append(next_token)
                context.append(next_token)

            repaired_tokens = np.array(repaired_tokens)
            #计算准确率，只计算预测部分
            total_tokens += len(original_tokens) // 3
            total_correct += np.sum((original_tokens == repaired_tokens)) - len(original_tokens) // 3 * 2

            repaired_tokens = repaired_tokens.reshape(-1, n_quantizers).T
            i_offset = np.array([i*codebook_size for i in range(n_quantizers)])[:,np.newaxis]
            repaired_codes = torch.tensor(repaired_tokens - i_offset, dtype=torch.long, device=device).unsqueeze(0)
            repaired_latent,_,_ = encoder.quantizer.from_codes(repaired_codes)

            _, _, _, _, repaired_y_g = decoder(repaired_latent)
            repaired_audio = repaired_y_g.squeeze()
            repaired_audio = repaired_audio.cpu().numpy()

            #计算比特数
            T = original_codes.shape[2] # 时间步数
            bits_per_index = math.ceil(math.log2(codebook_size))  # 每个索引的比特数
            file_bits = 2 * bits_per_index * T
            total_bits += file_bits

            sf.write(os.path.join(h.test_wav_output_dir, filename.split('.')[0]+'.wav'), repaired_audio, h.sampling_rate,'PCM_16')

        accuracy = total_correct / total_tokens
        total_inference_time = time.time() - start_time
    bitrate_kbps = (total_bits / total_duration) / 1000
    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")
    logging.info(f"Total duration: {total_duration:.3f} seconds, Total bits: {total_bits}, Bitrate: {bitrate_kbps:.3f} kbps")
    logging.info(f"Total tokens: {total_tokens}, Total correct: {total_correct}, Accuracy: {accuracy:.3f}")



def main():
    print('Initializing Inference Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_do/config.json'

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