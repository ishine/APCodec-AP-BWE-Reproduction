import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import librosa
import torchaudio.functional as F_audio

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size, dtype=y.dtype, device=y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True, return_complex=True)
    spec = torch.view_as_real(spec)  # 将复数张量转为 [..., 2] 格式（实部和虚部）
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec  # [batch_size, n_fft/2+1, frames]

def amp_pha_specturm(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size, dtype=y.dtype, device=y.device)

    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)  # [batch_size, n_fft//2+1, frames, 2]
    rea = stft_spec[:, :, :, 0]  # [batch_size, n_fft//2+1, frames]
    imag = stft_spec[:, :, :, 1]  # [batch_size, n_fft//2+1, frames]
    log_amplitude = torch.log(torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2))) + 1e-5)
    phase = torch.atan2(imag, rea)
    return log_amplitude, phase, rea, imag

def get_dataset_filelist(input_training_wav_list, input_validation_wav_list, mixed_training_wav_list, mixed_validation_wav_list):
    with open(input_training_wav_list, 'r') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]
    with open(input_validation_wav_list, 'r') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]
    with open(mixed_training_wav_list, 'r') as fi:
        training_mixed_files = [x for x in fi.read().split('\n') if len(x) > 0]
    with open(mixed_validation_wav_list, 'r') as fi:
        validation_mixed_files = [x for x in fi.read().split('\n') if len(x) > 0]
        
    return training_files, validation_files, training_mixed_files, validation_mixed_files

class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_files, mixed_files, segment_size, n_fft, num_mels_for_loss,
                 hop_size, win_size, sampling_rate, ratio, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, rank=0):
        self.clean_files = clean_files
        self.mixed_files = mixed_files

        random.seed(1234 + rank)  # Use rank-specific seed for shuffling
        if shuffle:
            # 同步打乱 clean 和 mixed 文件，确保对应关系
            paired_files = list(zip(self.clean_files, self.mixed_files))
            random.shuffle(paired_files)
            self.clean_files, self.mixed_files = zip(*paired_files)
            self.clean_files = list(self.clean_files)
            self.mixed_files = list(self.mixed_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.ratio = ratio
        self.split = split
        self.n_fft = n_fft
        self.num_mels_for_loss = num_mels_for_loss
        self.hop_size = hop_size
        self.win_size = win_size
        self.cached_clean_wav = None
        self.cached_mixed_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.rank = rank

    def __getitem__(self, index):
        filename_clean = self.clean_files[index]
        filename_mixed = self.mixed_files[index]
        if self._cache_ref_count == 0:
            audio_clean = load_wav(filename_clean, self.sampling_rate)
            audio_mixed = load_wav(filename_mixed, self.sampling_rate)
            self.cached_clean_wav = audio_clean
            self.cached_mixed_wav = audio_mixed
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio_clean = self.cached_clean_wav
            audio_mixed = self.cached_mixed_wav
            self._cache_ref_count -= 1

        audio_clean = torch.FloatTensor(audio_clean)  # [T]
        audio_mixed = torch.FloatTensor(audio_mixed)  # [T]

        audio = audio_clean.unsqueeze(0)  # [1, T]
        audio_mixed = audio_mixed.unsqueeze(0)  # [1, T]

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start+self.segment_size]
                audio_mixed = audio_mixed[:, audio_start: audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                audio_mixed = torch.nn.functional.pad(audio_mixed, (0, self.segment_size - audio_mixed.size(1)), 'constant')
        else:
            if audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size > 0:
                audio = audio[:, 0:-(audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size)]
                audio_mixed = audio_mixed[:, 0:-(audio_mixed.size(1) - (audio_mixed.size(1)//self.hop_size) * self.hop_size)]
            if (audio.size(1)//self.hop_size + 1) % self.ratio > 0:
                audio = audio[:,0: (((audio.size(1)//self.hop_size + 1) // self.ratio) * self.ratio - 1) * self.hop_size]
                audio_mixed = audio_mixed[:,0: (((audio_mixed.size(1)//self.hop_size + 1) // self.ratio) * self.ratio - 1) * self.hop_size]

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels_for_loss,self.sampling_rate, self.hop_size, self.win_size, 0, None,) #[1,n_fft/2+1,frames]

        log_amplitude, phase, rea, imag = amp_pha_specturm(audio_mixed, self.n_fft, self.hop_size, self.win_size) #[1,n_fft/2+1,frames]

        return (log_amplitude.squeeze(), phase.squeeze(), rea.squeeze(), imag.squeeze(), audio.squeeze(0), mel_loss.squeeze())

    def __len__(self):
        return len(self.clean_files)