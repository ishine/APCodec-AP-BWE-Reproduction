import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from exp_rasr.dataset import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from exp_rasr.models import Encoder, Decoder, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, amplitude_loss, phase_loss, STFT_consistency_loss, MultiResolutionDiscriminator
from exp_rasr.utils import AttrDict, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

mp.set_start_method('spawn', force=True)

torch.backends.cudnn.benchmark = True

def remove_module_prefix(state_dict):
    """去掉DataParallel保存的参数的module前缀"""
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # 去除 "module." 前缀
        new_state_dict[k] = v
    return new_state_dict

def collate_fn(batch):
    audio, text_label, target_length, input_length, basename= zip(*batch)
    max_audio_length = max(len(a) for a in audio)
    audio_padded = torch.stack([
        torch.nn.functional.pad(a.clone().detach() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float), 
                                (0, max_audio_length - len(a)), 'constant')
        for a in audio
    ])
    max_target_length = max(len(indices) for indices in text_label)
    text_labels = torch.zeros(len(batch), max_target_length, dtype=torch.long)
    for i, indices in enumerate(text_label):
        text_labels[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)

    target_lengths = torch.tensor(target_length, dtype=torch.long)
    input_lengths = torch.tensor(input_length, dtype=torch.long)
    
    return audio_padded, text_labels, target_lengths, input_lengths, basename

def train(rank, world_size, h):
   
    init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)

    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    if rank == 0:
        print("Encoder: ")
        print(encoder)
        print("Decoder: ")
        print(decoder)
        os.makedirs(h.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_encoder = scan_checkpoint(h.checkpoint_path, 'encoder_')
        cp_decoder = scan_checkpoint(h.checkpoint_path, 'decoder_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')

    steps = 0
    if cp_encoder is None or cp_decoder is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_encoder = load_checkpoint(cp_encoder, 'cpu')
        state_dict_decoder = load_checkpoint(cp_decoder, 'cpu')
        state_dict_do = load_checkpoint(cp_do, 'cpu')
        encoder.load_state_dict(remove_module_prefix(state_dict_encoder['encoder']))
        decoder.load_state_dict(remove_module_prefix(state_dict_decoder['decoder']))
        mpd.load_state_dict(remove_module_prefix(state_dict_do['mpd']))
        mrd.load_state_dict(remove_module_prefix(state_dict_do['mrd']))
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    encoder = DistributedDataParallel(encoder, device_ids=[rank])
    decoder = DistributedDataParallel(decoder, device_ids=[rank])
    mpd = DistributedDataParallel(mpd, device_ids=[rank])
    mrd = DistributedDataParallel(mrd, device_ids=[rank])

    optim_g = torch.optim.AdamW(itertools.chain(encoder.parameters(), decoder.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                       h.hop_size, h.win_size, h.sampling_rate, h.ratio, h.label_path, n_cache_reuse=0,
                       shuffle=True, device=device)
    
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, 
                              sampler=train_sampler,
                              batch_size=h.batch_size // world_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    if rank == 0:
        validset = Dataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                           h.hop_size, h.win_size, h.sampling_rate, h.ratio, h.label_path, False, False, n_cache_reuse=0,
                           device=device)
                           
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=collate_fn)

        sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    encoder.train()
    decoder.train()
    mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        
        train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            audio, text_labels, target_lengths, input_lengths, basename = batch
            audio = audio.to(device, non_blocking=True).unsqueeze(1)
            text_labels = text_labels.to(device, non_blocking=True)
            logamp, pha, rea, imag = amp_pha_specturm(audio.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            _, latent, _, commitment_loss, codebook_loss, asr_loss = encoder(logamp, pha, text_labels)
            logamp_g, pha_g, rea_g, imag_g, y_g = decoder(latent)
            y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels_for_loss ,h.sampling_rate, h.hop_size, h.win_size, 0, None,)
            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(audio, y_g.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audio, y_g.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            L_D = loss_disc_s * 0.1 + loss_disc_f
            L_D.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            L_A = amplitude_loss(logamp, logamp_g)
            L_IP, L_GD, L_PTD = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
            L_P = L_IP + L_GD + L_PTD
            _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
            L_R = F.l1_loss(rea, rea_g)
            L_I = F.l1_loss(imag, imag_g)
            L_S = L_C + 2.25 * (L_R + L_I)
            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(audio, y_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(audio, y_g)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_g)
            y_mel = mel_spectrogram(audio.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)
            y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)
            L_GAN_G = loss_gen_s * 0.1 + loss_gen_f
            L_FM = loss_fm_s * 0.1 + loss_fm_f
            L_Mel = F.l1_loss(y_mel, y_g_mel)
            L_Mel_L2 = amplitude_loss(y_mel, y_g_mel)
            L_W = L_GAN_G + L_FM + 45 * L_Mel + 45 * L_Mel_L2
            L_G = (
                45 * L_A
                + 100 * L_P
                + 20 * L_S
                + L_W
                + 10 * codebook_loss
                + 2.5 * commitment_loss
                + 10 * asr_loss
            )
            L_G.backward()
            optim_g.step()
            
            if rank == 0:
                # STDOUT logging
                if steps % h.stdout_interval == 0:
                    with torch.no_grad():
                        A_error = amplitude_loss(logamp, logamp_g).item()
                        IP_error, GD_error, PTD_error = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
                        IP_error = IP_error.item()
                        GD_error = GD_error.item()
                        PTD_error = PTD_error.item()
                        C_error = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final).item()
                        R_error = F.l1_loss(rea, rea_g).item()
                        I_error = F.l1_loss(imag, imag_g).item()
                        Mel_error = F.l1_loss(y_mel, y_g_mel).item()
                        Mel_L2_error = amplitude_loss(y_mel, y_g_mel).item()
                        commit_loss = commitment_loss.item()
                        asr_loss = asr_loss.item()

                        print(
                            "Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Phase Time Difference Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, Mel Spectrogram L2 Loss : {:4.3f}, Commit Loss : {:4.3f}, ASR Loss : {:4.3f}, s/b : {:4.3f}".format(
                                steps, L_G, A_error, IP_error, GD_error, PTD_error, C_error, R_error, I_error,
                                Mel_error, Mel_L2_error, commit_loss, asr_loss, time.time() - start
                            )
                        )

                # checkpointing
                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/encoder_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {"encoder": encoder.module.state_dict()})
                    checkpoint_path = "{}/decoder_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {"decoder": decoder.module.state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": mpd.module.state_dict(),
                            "mrd": mrd.module.state_dict(),
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
                    sw.add_scalar("Training/ASR_Loss", asr_loss, steps)

                # Validation
                if steps % h.validation_interval == 0:
                    encoder.eval()
                    decoder.eval()
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
                    val_asr_err_tot = 0

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            audio,text_labels, target_lengths, input_lengths, basename = batch
                            audio = audio.to(device, non_blocking=True).unsqueeze(1)
                            text_labels = text_labels.to(device, non_blocking=True)
                            logamp, pha, rea, imag = amp_pha_specturm(audio.squeeze(1), h.n_fft, h.hop_size, h.win_size)
                            _, latent,_, _, _, asr_loss = encoder.module(logamp.to(device), pha.to(device), None)
                            logamp_g, pha_g, rea_g, imag_g, y_g = decoder.module(latent)
                            y_mel = mel_spectrogram(audio.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None, center=True)
                            y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate, h.hop_size, h.win_size, 0, None,)
                            _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)

                            val_A_err_tot += amplitude_loss(logamp, logamp_g).item()
                            val_IP_err, val_GD_err, val_PTD_err = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
                            val_IP_err_tot += val_IP_err.item()
                            val_GD_err_tot += val_GD_err.item()
                            val_PTD_err_tot += val_PTD_err.item()
                            val_C_err_tot += STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final).item()
                            val_R_err_tot += F.l1_loss(rea, rea_g).item()
                            val_I_err_tot += F.l1_loss(imag, imag_g).item()
                            val_Mel_err_tot += F.l1_loss(y_mel, y_g_mel).item()
                            val_Mel_L2_err_tot += amplitude_loss(y_mel, y_g_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio("gt/y_{}".format(j), audio[0], steps, h.sampling_rate)
                                    sw.add_figure("gt/y_logamp_{}".format(j),plot_spectrogram(logamp[0].cpu().numpy()),steps)
                                    sw.add_figure("gt/y_pha_{}".format(j),plot_spectrogram(pha[0].cpu().numpy()),steps)

                                sw.add_audio("generated/y_g_{}".format(j),y_g[0],steps,h.sampling_rate)
                                sw.add_figure("generated/y_g_logamp_{}".format(j),plot_spectrogram(logamp_g[0].cpu().numpy()),steps)
                                sw.add_figure("generated/y_g_pha_{}".format(j),plot_spectrogram(pha_g[0].cpu().numpy()),steps)

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

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
    
    destroy_process_group()


def main():
    print('Initializing Training Process..')

    config_file = '/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_rasr/config.json'

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    if not os.path.exists(h.checkpoint_path):
         os.makedirs(h.checkpoint_path, exist_ok=True)
    build_env(config_file, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22345' 

        print(f"Training with {n_gpus} GPUs.")
        mp.spawn(train, nprocs=n_gpus, args=(n_gpus, h,))
    else:
        print("Training on a single GPU.")
        train(0, 1, h)


if __name__ == '__main__':
    main()