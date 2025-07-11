import os
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from rich.progress import track
from pystoi.stoi import stoi


def stft(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    stft_pha = torch.angle(stft_spec)

    return stft_mag, stft_pha


def cal_snr(pred, target):
    snr = (20 * torch.log10(torch.norm(target, dim=-1) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()
    return snr


def cal_lsd(pred, target):
    sp = torch.log10(stft(pred)[0].square().clamp(1e-8))
    st = torch.log10(stft(target)[0].square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


def anti_wrapping_function(x):
    return x - torch.round(x / (2 * np.pi)) * 2 * np.pi


def cal_apd(pred, target):
    pha_pred = stft(pred)[1]
    pha_target = stft(target)[1]
    dim_freq = 1025
    dim_time = pha_pred.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(device)
    gd_r = torch.matmul(pha_target.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(pha_pred.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(device)
    iaf_r = torch.matmul(pha_target, iaf_matrix)
    iaf_g = torch.matmul(pha_pred, iaf_matrix)

    apd_ip = anti_wrapping_function(pha_pred - pha_target).square().mean(dim=1).sqrt().mean()
    apd_gd = anti_wrapping_function(gd_r - gd_g).square().mean(dim=1).sqrt().mean()
    apd_iaf = anti_wrapping_function(iaf_r - iaf_g).square().mean(dim=1).sqrt().mean()

    return apd_ip, apd_gd, apd_iaf

def cal_mcd(pred, target, n_mfcc=13, sample_rate=48000, n_fft=1024, hop_size=40):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_size,
            'n_mels': 80
        }
    ).to(pred.device)
    
    mfcc_pred = mfcc_transform(pred)
    mfcc_target = mfcc_transform(target)

    mcd = (mfcc_pred - mfcc_target).square().mean(dim=[1, 2]).sqrt().mean() * (10.0 / np.log(10.0))
    
    return mcd

def cal_stoi_score(pred, target, sr):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return stoi(target_np, pred_np, sr, extended=False)

def cal_sdr(pred, target, eps=1e-8):
    signal_power = (target ** 2).sum()
    noise_power = ((target - pred) ** 2).sum()
    sdr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return sdr

def cal_si_snr(pred, target, eps=1e-8):
    target = target - target.mean()
    pred = pred - pred.mean()
    scale = (pred * target).sum(dim=1) / (target.norm(p=2, dim=1) ** 2 + eps)
    s_target = scale * target
    e_noise = pred - s_target
    si_sdr = 10 * torch.log10((s_target.norm(p=2) ** 2 + eps) / (e_noise.norm(p=2) ** 2 + eps))
    return si_sdr


def main(h):

    wav_indexes = os.listdir(h.reference_wav_dir)
    
    metrics = {
        'lsd':[], 'apd_ip': [], 'apd_gd': [], 'apd_iaf': [],
        'snr':[], 'mcd':[], 'stoi':[], 'sdr':[],'si_snr':[]
    }

    for wav_index in track(wav_indexes):

        ref_path = os.path.join(h.reference_wav_dir, wav_index)
        syn_path = os.path.join(h.synthesis_wav_dir, wav_index)

        ref_wav, ref_sr = torchaudio.load(ref_path)
        syn_wav, syn_sr = torchaudio.load(syn_path)

        length = min(ref_wav.size(1), syn_wav.size(1))
        ref_wav = ref_wav[:, : length].to(device)
        syn_wav = syn_wav[:, : length].to(device)

        lsd_score = cal_lsd(syn_wav, ref_wav)
        apd_score = cal_apd(syn_wav, ref_wav)
        snr_score = cal_snr(syn_wav, ref_wav)
        mcd_score = cal_mcd(syn_wav, ref_wav)
        stoi_score = cal_stoi_score(syn_wav, ref_wav, sr=ref_sr)
        sdr_score = cal_sdr(syn_wav, ref_wav)
        si_snr_score = cal_si_snr(syn_wav, ref_wav)

        metrics['lsd'].append(lsd_score)
        metrics['apd_ip'].append(apd_score[0])
        metrics['apd_gd'].append(apd_score[1])
        metrics['apd_iaf'].append(apd_score[2])
        metrics['snr'].append(snr_score)
        metrics['mcd'].append(mcd_score)
        metrics['stoi'].append(torch.tensor(stoi_score))
        metrics['sdr'].append(sdr_score)
        metrics['si_snr'].append(si_snr_score)

    print('LSD: {:.3f}'.format(torch.stack(metrics['lsd']).mean()))
    print('SNR: {:.3f}'.format(torch.stack(metrics['snr']).mean()))
    print('APD_IP: {:.3f}'.format(torch.stack(metrics['apd_ip']).mean()))
    print('APD_GD: {:.3f}'.format(torch.stack(metrics['apd_gd']).mean()))
    print('APD_IAF: {:.3f}'.format(torch.stack(metrics['apd_iaf']).mean()))
    print('MCD: {:.3f}'.format(torch.stack(metrics['mcd']).mean()))
    print('STOI: {:.3f}'.format(torch.stack(metrics['stoi']).mean()))
    print('SDR: {:.3f}'.format(torch.stack(metrics['sdr']).mean()))
    print('SI-SNR: {:.3f}'.format(torch.stack(metrics['si_snr']).mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--reference_wav_dir', default='/mnt/nvme_share/srt30/Datasets/VCTK-0.92/wav48/origin/test')
    parser.add_argument('--synthesis_wav_dir', default='/mnt/nvme_share/srt30/checkpoint/exp_fsq/output_wav_1350k')

    h = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(h)