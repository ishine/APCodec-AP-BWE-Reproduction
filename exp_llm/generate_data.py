from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import json
import torch
import random
import librosa
from exp_fsq.utils import AttrDict
from exp_fsq.dataset import amp_pha_specturm
from exp_fsq.models import Encoder
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

def drop_token(codes, drop_prob=0.1):
    result = []
    for i in range(0, len(codes), 4):
        if random.random() >= drop_prob:
            result.extend(codes[i:i+4])
    return result

def generate_data(h):

    encoder = Encoder(h).to(device)
    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    encoder.eval()

    filelist = sorted(glob.glob(os.path.join(h.input_wavs_dir, '*.wav')))
    dataset = []

    with torch.no_grad():
        for filename in filelist:
            audio, sr = librosa.load(filename, sr=h.sampling_rate, mono=True)
            audio = torch.FloatTensor(audio).to(device)
            audio = audio.unsqueeze(0)
            if audio.size(1) >= h.segment_size:
                max_audio_start = audio.size(1) - h.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + h.segment_size] #[1,T]
            logamp, pha, _, _ = amp_pha_specturm(audio, h.n_fft, h.hop_size, h.win_size)
            _, codes, _, _ = encoder(logamp, pha)
            codes = codes.squeeze(0).cpu().numpy() #[n_quantizers, T]
            codes = codes.transpose(1, 0).flatten().tolist() #[n_quantizers*T]
            dropped_codes = drop_token(codes, h.drop_prob)
            dataset.append({
                "dropped_tokens": dropped_codes,
                "original_tokens": codes,
            })

    with open(h.output_file, 'w') as f:
        f.write('[\n')
        for item in dataset:
            f.write(json.dumps(item, separators=(',', ':')) + ',\n')
        f.write(']\n')
    logging.info(f"Dataset saved to {h.output_file}, total samples: {len(dataset)}")
    return dataset

def main():
    print('Initializing Generate Data Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-Reproduction/exp_llm/config.json'
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

    generate_data(h)


if __name__ == '__main__':
    main()