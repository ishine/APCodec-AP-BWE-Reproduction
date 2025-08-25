from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import json
import torch
from exp_tran.utils import AttrDict
from exp_tran.dataset import amp_pha_specturm
from exp_tran.models import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def greedy_decode(encoder, memory, vocab, max_len=50, bos_idx=1, eos_idx=2):
    device = memory.device
    generated_tokens = torch.full((1, max_len), eos_idx, dtype=torch.long, device=device)
    current_tokens = torch.tensor([[bos_idx]], device=device) 
    for t in range(max_len):
        logits = encoder.asr_decoder(current_tokens.transpose(0, 1), memory)[-1]  # [1, vocab_size]
        next_token = torch.argmax(logits, dim=-1)  # [1]
        generated_tokens[0, t] = next_token
        if next_token == eos_idx:
            break
        current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
    indices = generated_tokens[0, :t+1].cpu().numpy().tolist()
    if eos_idx in indices:
        indices = indices[:indices.index(eos_idx)]
    text = ''.join([vocab[idx] for idx in indices if idx in vocab and idx != bos_idx])
    return indices, text

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
        0: '<blank>', 1: '<bos>', 2: '<eos>', 3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g',
        10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n', 17: 'o',
        18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u', 24: 'v', 25: 'w',
        26: 'x', 27: 'y', 28: 'z', 29: ' '
    }

    predicted_labels = {}

    with torch.no_grad():
        for filename in filelist:
            raw_wav, _ = librosa.load(os.path.join(h.test_input_wavs_dir, filename), sr=h.sampling_rate, mono=True)
            raw_wav = torch.FloatTensor(raw_wav).to(device).unsqueeze(0)
            logamp, pha, _, _, = amp_pha_specturm(raw_wav, h.n_fft, h.hop_size, h.win_size)
            lbq, latent, codes, _, _, _ = encoder(logamp, pha, text_labels=None)
            memory = lbq.permute(2, 0, 1)
            indices, pred_text = greedy_decode(encoder, memory, vocab)
            logamp_g, pha_g, _, _, y_g = decoder(latent)
            file_id = filename.split('.')[0]
            predicted_labels[file_id] = {
                "indices": str(indices),
                "target_length": len(indices),
                "text": pred_text
            }
            audio = y_g.squeeze().cpu().numpy()
            sf.write(os.path.join(h.test_wav_output_dir, f'{file_id}.wav'), audio, h.sampling_rate, 'PCM_16')

    with open(os.path.join(h.test_label_output_dir, 'predicted_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(predicted_labels, f, ensure_ascii=False, indent=4)

def main():
    print('Initializing Inference Process..')
    config_file = '/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_tran/config.json'
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(h)

if __name__ == '__main__':
    main()