import argparse
import itertools
import json
import os
import time
import warnings

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from exp_bwe.dataset import Dataset, amp_pha_spectrum, get_dataset_filelist, mel_spectrogram
from exp_bwe.models import Decoder, Encoder, BWE, MultiPeriodDiscriminator, MultiResolutionDiscriminator, MultiScaleDiscriminator, STFT_consistency_loss, \
    amplitude_loss, discriminator_loss, feature_loss, generator_loss, phase_loss
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from exp_bwe.utils import AttrDict, build_env, load_checkpoint, plot_spectrogram, save_checkpoint, scan_checkpoint


torch.backends.cudnn.benchmark = True

def train(h):
    torch.autograd.set_detect_anomaly(True)

    torch.cuda.manual_seed(h.seed)  
    device = torch.device('cuda:{:d}'.format(0))

    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    bwe = BWE(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    print("Encoder: ")
    print(encoder)
    print("Decoder: ")
    print(decoder)
    print("BWE: ")
    print(bwe)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_encoder = scan_checkpoint(h.checkpoint_path, "encoder_")
        cp_decoder = scan_checkpoint(h.checkpoint_path, "decoder_")
        cp_bwe = scan_checkpoint(h.checkpoint_path, "bwe_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_encoder is None or cp_decoder is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_encoder = load_checkpoint(cp_encoder, device)
        state_dict_decoder = load_checkpoint(cp_decoder, device)
        state_dict_bwe = load_checkpoint(cp_bwe, device)
        state_dict_do = load_checkpoint(cp_do, device)
        encoder.load_state_dict(state_dict_encoder["encoder"])
        decoder.load_state_dict(state_dict_decoder["decoder"])
        bwe.load_state_dict(state_dict_bwe["bwe"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    optim_g = torch.optim.AdamW(itertools.chain(encoder.parameters(), decoder.parameters(), bwe.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                        h.hop_size, h.win_size, h.sampling_rate, h.low_sampling_rate, h.ratio,
                        n_cache_reuse=0,shuffle=False, device=device)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                            sampler=None, batch_size=h.batch_size, 
                            pin_memory=True, drop_last=True,)

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels_for_loss,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.low_sampling_rate,
        h.ratio,
        False,
        False,
        n_cache_reuse=0,
        device=device,
    )

    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )


    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    encoder.train()
    decoder.train()
    bwe.train()
    mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            audio_wb, audio_nb = batch
            audio_wb = torch.autograd.Variable(audio_wb.to(device, non_blocking=True)).unsqueeze(1)
            audio_nb = torch.autograd.Variable(audio_nb.to(device, non_blocking=True)).unsqueeze(1)

            logamp_wb, pha_wb, rea_wb, imag_wb = amp_pha_spectrum(audio_wb.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            logamp_nb, pha_nb, rea_nb, imag_nb = amp_pha_spectrum(audio_nb.squeeze(1), h.n_fft, h.hop_size, h.win_size)

            latent, codes, commitment_loss, codebook_loss = encoder(logamp_nb, pha_nb)

            audio_nb_g = decoder(latent)

            logamp_nb_g, pha_nb_g, rea_nb_g, imag_nb_g = amp_pha_spectrum(audio_nb_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            
            logamp_wb_g, pha_wb_g = bwe(logamp_nb_g, pha_nb_g)
            rea_wb_g = torch.exp(logamp_wb_g)*torch.cos(pha_wb_g)
            imag_wb_g = torch.exp(logamp_wb_g)*torch.sin(pha_wb_g)
            spec_wb_g = torch.complex(rea_wb_g, imag_wb_g)
            y_wb_g = torch.istft(spec_wb_g, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=torch.hann_window(h.win_size).to(latent.device), center=True) 
            y_wb_g = y_wb_g.unsqueeze(1)

            y_wb_g_mel = mel_spectrogram(y_wb_g.squeeze(1), h.n_fft, h.num_mels_for_loss ,h.sampling_rate, h.hop_size, h.win_size, 0, None,)

            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(audio_wb, y_wb_g.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audio_wb, y_wb_g.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            L_D = loss_disc_s * 0.1 + loss_disc_f

            L_D.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(mpd.parameters(), mrd.parameters()),max_norm=1.0)
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # Losses defined on log amplitude spectra
            L_A = amplitude_loss(logamp_wb, logamp_wb_g)

            L_IP, L_GD, L_PTD = phase_loss(pha_wb, pha_wb_g, h.n_fft, pha_wb.size()[-1])
            # Losses defined on phase spectra
            L_P = L_IP + L_GD + L_PTD

            _, _, rea_g_final, imag_g_final = amp_pha_spectrum(y_wb_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            L_C = STFT_consistency_loss(rea_wb_g, rea_g_final, imag_wb_g, imag_g_final)

            L_R = F.l1_loss(rea_wb, rea_wb_g)
            L_I = F.l1_loss(imag_wb, imag_wb_g)
            # Losses defined on reconstructed STFT spectra
            L_S = L_C + 2.25 * (L_R + L_I)

            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(audio_wb, y_wb_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(audio_wb, y_wb_g)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_g)

            y_wb_mel = mel_spectrogram(audio_wb.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)
            y_wb_g_mel = mel_spectrogram(y_wb_g.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)

            L_GAN_G = loss_gen_s * 0.1 + loss_gen_f
            L_FM = loss_fm_s * 0.1 + loss_fm_f
            L_Mel = F.l1_loss(y_wb_mel, y_wb_g_mel)
            L_Mel_L2 = amplitude_loss(y_wb_mel, y_wb_g_mel)
            # Losses defined on final waveforms
            L_W = L_GAN_G + L_FM + 45 * L_Mel + 45 * L_Mel_L2

            L_G = 45 * L_A + 100 * L_P + 20 * L_S + L_W + codebook_loss * 10 + commitment_loss * 2.5

            L_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), decoder.parameters(), bwe.parameters()), max_norm=1.0)
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    A_error = amplitude_loss(logamp_wb, logamp_wb_g).item()
                    IP_error, GD_error, PTD_error = phase_loss(pha_wb, pha_wb_g, h.n_fft, pha_wb.size()[-1])
                    IP_error = IP_error.item()
                    GD_error = GD_error.item()
                    PTD_error = PTD_error.item()
                    C_error = STFT_consistency_loss(rea_wb_g, rea_g_final, imag_wb_g, imag_g_final).item()
                    R_error = F.l1_loss(rea_wb, rea_wb_g).item()
                    I_error = F.l1_loss(imag_wb, imag_wb_g).item()
                    Mel_error = F.l1_loss(y_wb_mel, y_wb_g_mel).item()
                    Mel_L2_error = amplitude_loss(y_wb_mel, y_wb_g_mel).item()
                    commit_loss = commitment_loss.item()

                print(
                    "Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Phase Time Difference Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, Mel Spectrogram L2 Loss : {:4.3f}, Commit Loss : {:4.3f}, s/b : {:4.3f}".format(
                        steps,
                        L_G,
                        A_error,
                        IP_error,
                        GD_error,
                        PTD_error,
                        C_error,
                        R_error,
                        I_error,
                        Mel_error,
                        Mel_L2_error,
                        commit_loss,
                        time.time() - start_b,
                    )
                )

            # checkpointing (只在主进程)
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/encoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"encoder": encoder.state_dict()})

                checkpoint_path = "{}/decoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"decoder": decoder.state_dict()})

                checkpoint_path = "{}/bwe_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"bwe": bwe.state_dict()})

                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )

            # Tensorboard summary logging 
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)
                
            # Validation
            if steps % h.validation_interval == 0:
                encoder.eval()
                decoder.eval()
                bwe.eval()
                torch.cuda.empty_cache()
                val_A_err_tot = 0
                val_IP_err_tot = 0
                val_GD_err_tot = 0
                val_PTD_err_tot = 0
                val_C_err_tot = 0
                val_R_err_tot = 0
                val_I_err_tot = 0
                val_Mel_err_tot = 0
                val_Mel_L2_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        audio_wb, audio_nb = batch
                        audio_wb = torch.autograd.Variable(audio_wb.to(device, non_blocking=True)).unsqueeze(1)
                        audio_nb = torch.autograd.Variable(audio_nb.to(device, non_blocking=True)).unsqueeze(1)

                        logamp_wb, pha_wb, rea_wb, imag_wb = amp_pha_spectrum(audio_wb.squeeze(1), h.n_fft, h.hop_size, h.win_size)
                        logamp_nb, pha_nb, rea_nb, imag_nb = amp_pha_spectrum(audio_nb.squeeze(1), h.n_fft, h.hop_size, h.win_size)

                        latent,codes, commitment_loss, codebook_loss = encoder(logamp_nb, pha_nb)

                        audio_nb_g = decoder(latent)

                        logamp_nb_g, pha_nb_g, rea_nb_g, imag_nb_g = amp_pha_spectrum(audio_nb_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
                        
                        logamp_wb_g, pha_wb_g = bwe(logamp_nb_g, pha_nb_g)
                        rea_wb_g = torch.exp(logamp_wb_g)*torch.cos(pha_wb_g)
                        imag_wb_g = torch.exp(logamp_wb_g)*torch.sin(pha_wb_g)
                        spec_wb_g = torch.complex(rea_wb_g, imag_wb_g)
                        y_wb_g = torch.istft(spec_wb_g, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=torch.hann_window(h.win_size).to(latent.device), center=True) 
                        y_wb_g = y_wb_g.unsqueeze(1)

                        y_wb_mel = mel_spectrogram(audio_wb.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)
                        y_wb_g_mel = mel_spectrogram(y_wb_g.squeeze(1), h.n_fft, h.num_mels_for_loss ,h.sampling_rate, h.hop_size, h.win_size, 0, None,)

                        _, _, rea_g_final, imag_g_final = amp_pha_spectrum(y_wb_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)

                        val_A_err_tot += amplitude_loss(logamp_wb, logamp_wb_g).item()
                        val_IP_err, val_GD_err, val_PTD_err = phase_loss(pha_wb, pha_wb_g, h.n_fft, pha_wb.size()[-1])
                        val_IP_err_tot += val_IP_err.item()
                        val_GD_err_tot += val_GD_err.item()
                        val_PTD_err_tot += val_PTD_err.item()
                        val_C_err_tot += STFT_consistency_loss(rea_wb_g, rea_g_final, imag_wb_g, imag_g_final).item()
                        val_R_err_tot += F.l1_loss(rea_wb, rea_wb_g).item()
                        val_I_err_tot += F.l1_loss(imag_wb, imag_wb_g).item()
                        val_Mel_err_tot += F.l1_loss(y_wb_mel, y_wb_g_mel).item()
                        val_Mel_L2_err_tot += amplitude_loss(y_wb_mel, y_wb_g_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j), audio_wb[0], steps, h.sampling_rate
                                )
                                sw.add_figure(
                                    "gt/y_logamp_{}".format(j),
                                    plot_spectrogram(logamp_wb[0].cpu().numpy()),
                                    steps,
                                )
                                sw.add_figure(
                                    "gt/y_pha_{}".format(j),
                                    plot_spectrogram(pha_wb[0].cpu().numpy()),
                                    steps,
                                )

                            sw.add_audio(
                                "generated/y_g_{}".format(j),
                                y_wb_g[0],
                                steps,
                                h.sampling_rate,
                            )
                            sw.add_figure(
                                "generated/y_g_logamp_{}".format(j),
                                plot_spectrogram(logamp_wb_g[0].cpu().numpy()),
                                steps,
                            )
                            sw.add_figure(
                                "generated/y_g_pha_{}".format(j),
                                plot_spectrogram(pha_wb_g[0].cpu().numpy()),
                                steps,
                            )

                    val_A_err = val_A_err_tot / (j + 1)
                    val_IP_err = val_IP_err_tot / (j + 1)
                    val_GD_err = val_GD_err_tot / (j + 1)
                    val_PTD_err = val_PTD_err_tot / (j + 1)
                    val_C_err = val_C_err_tot / (j + 1)
                    val_R_err = val_R_err_tot / (j + 1)
                    val_I_err = val_I_err_tot / (j + 1)
                    val_Mel_err = val_Mel_err_tot / (j + 1)
                    val_Mel_L2_err = val_Mel_L2_err_tot / (j + 1)
                    sw.add_scalar("Validation/Amplitude_Loss", val_A_err, steps)
                    sw.add_scalar("Validation/Instantaneous_Phase_Loss", val_IP_err, steps)
                    sw.add_scalar("Validation/Group_Delay_Loss", val_GD_err, steps)
                    sw.add_scalar("Validation/Phase_Time_Difference_Loss", val_PTD_err, steps)
                    sw.add_scalar("Validation/STFT_Consistency_Loss", val_C_err, steps)
                    sw.add_scalar("Validation/Real_Part_Loss", val_R_err, steps)
                    sw.add_scalar("Validation/Imaginary_Part_Loss", val_I_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_L2_loss", val_Mel_L2_err, steps)

                encoder.train()
                decoder.train()
                bwe.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        print("Time taken for epoch {} is {} sec\n".format(epoch + 1, int(time.time() - start)))


def main():
    print("Initializing Training Process..")

    config_file = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_bwe/config.json"

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, "config.json", h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass
    train(h)

if __name__ == "__main__":
    main()