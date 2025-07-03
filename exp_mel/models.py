import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from exp_mel.utils import init_weights, get_padding
import numpy as np
from exp_mel.quantize import ResidualFSQ

LRELU_SLOPE = 0.1

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x, cond_embedding_id = None) :
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()

        self.input_channels=h.num_mels
        self.h=h
        self.dim=256
        self.num_layers=4
        self.adanorm_num_embeddings=None
        self.intermediate_dim=512
        self.embed_mel = nn.Conv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.norm_mel = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value =  1 / self.num_layers
        self.convnext_mel = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.final_layer_norm_mel = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        #out_dim = h.n_fft + 2
        self.out_mel = torch.nn.Linear(self.dim, h.mel_Encoder_channel)

        self.mel_Encoder_downsample_output_conv = weight_norm(Conv1d(h.mel_Encoder_channel, h.mel_Encoder_channel//2, h.mel_Encoder_output_downconv_kernel_size, h.ratio, 
                                                  padding=get_padding(h.mel_Encoder_output_downconv_kernel_size, 1)))
        
        self.latent_output_conv = weight_norm(Conv1d(h.mel_Encoder_channel//2, h.latent_dim, h.latent_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.latent_output_conv_kernel_size, 1)))

        self.mel_Encoder_downsample_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)

        self.quantizer = ResidualFSQ(
            input_dim=h.latent_dim,
            codebook_dim=h.latent_dim,
            n_quantizers=4,
            codebook_size=1024,
            quantizer_dropout=False
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mel):

        mel_encode = self.embed_mel(mel)
        mel_encode = self.norm_mel(mel_encode.transpose(1, 2))
        mel_encode = mel_encode.transpose(1, 2)
        for conv_block in self.convnext_mel:
            mel_encode = conv_block(mel_encode, cond_embedding_id=None)
        mel_encode = self.final_layer_norm_mel(mel_encode.transpose(1, 2))
        mel_encode = self.out_mel(mel_encode).transpose(1, 2)
        mel_encode = self.mel_Encoder_downsample_output_conv(mel_encode)

        latent = self.latent_output_conv(mel_encode)
        latent,codes,_,commitment_loss,codebook_loss = self.quantizer(latent)

        return latent,codes,commitment_loss,codebook_loss

class Decoder(torch.nn.Module):
    def __init__(self, h):
        super(Decoder, self).__init__()

        self.h=h
        self.dim=256
        self.num_layers=4
        self.adanorm_num_embeddings=None
        self.intermediate_dim=512

        self.latent_input_conv = weight_norm(Conv1d(h.latent_dim, h.mel_Decoder_channel//2, h.latent_input_conv_kernel_size, 1, 
                                                 padding=get_padding(h.latent_input_conv_kernel_size, 1)))

        self.mel_Decoder_upsample1_input_conv = weight_norm(ConvTranspose1d(h.mel_Decoder_channel//2, h.mel_Decoder_channel,
                                                           h.mel_Decoder_input_upconv_kernel_size, h.ratio, padding=4))

        self.mel_Decoder_upsample2_input_conv = weight_norm(ConvTranspose1d(h.mel_Decoder_channel, h.mel_Decoder_channel,
                                                           h.hop_size*2, h.hop_size, padding=h.hop_size//2))

        self.embed_mel = nn.Conv1d(h.mel_Decoder_channel, self.dim, kernel_size=7, padding=3)
        self.norm_mel = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value =  1 / self.num_layers
        self.convnext_mel = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        
        self.final_layer_norm_mel = nn.LayerNorm(self.dim, eps=1e-6)

        self.apply(self._init_weights)

        self.out_mel = torch.nn.Linear(self.dim, h.latent_dim)

        self.mel_Decoder_output_conv = weight_norm(Conv1d(h.latent_dim, 1, h.mel_Decoder_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.mel_Decoder_output_conv_kernel_size, 1)))
        self.mel_Decoder_output_conv.apply(init_weights)
        self.output_activation = nn.Tanh()  

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, latent):
        #input: [B, 32, T']
        latent = self.latent_input_conv(latent) #[B,128,T']
        mel = self.mel_Decoder_upsample1_input_conv(latent) #[B,256,T]
        mel = self.mel_Decoder_upsample2_input_conv(mel) #[B,256,t]
        mel = self.embed_mel(mel) #[B,256,t]
        mel = self.norm_mel(mel.transpose(1, 2)) #[B,t,256]
        mel = mel.transpose(1, 2) #[B,256,t]
        for conv_block in self.convnext_mel:
            mel = conv_block(mel, cond_embedding_id=None)
        mel = self.final_layer_norm_mel(mel.transpose(1, 2)) #[B,256,t]
        mel = self.out_mel(mel).transpose(1, 2) #[B,32,t]
        audio = self.mel_Decoder_output_conv(mel) #[B,1,t]
        audio = self.output_activation(audio)

        return audio

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions= ((512, 20, 160), (1024, 40, 320), (2048, 80, 640)),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=r, num_embeddings=num_embeddings) for r in resolutions]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) :
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None) :
        fmap = []
        x=x.squeeze(1)
        
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram

def phase_loss(phase_r, phase_g, n_fft, frames):

    MSELoss = torch.nn.MSELoss()
    min_T = min(phase_r.shape[-1], phase_g.shape[-1])
    phase_r = phase_r[..., :min_T]
    phase_g = phase_g[..., :min_T]

    GD_matrix = torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=1)-torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=2)-torch.eye(n_fft//2+1)
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0,2,1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0,2,1), GD_matrix)

    PTD_matrix = torch.triu(torch.ones(frames,frames),diagonal=1)-torch.triu(torch.ones(frames,frames),diagonal=2)-torch.eye(frames)
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r-GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r-PTD_g))


    return IP_loss, GD_loss, PTD_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def amplitude_loss(log_amplitude_r, log_amplitude_g):

    MSELoss = torch.nn.MSELoss()
    min_T = min(log_amplitude_r.shape[-1], log_amplitude_g.shape[-1])
    log_amplitude_r = log_amplitude_r[..., :min_T]
    log_amplitude_g = log_amplitude_g[..., :min_T]
    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

    return loss, gen_losses