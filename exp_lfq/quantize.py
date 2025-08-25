import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2, ceil
from einops import rearrange, reduce
from torch.nn.utils import weight_norm
from torch import einsum

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class LFQ(nn.Module):
    def __init__(
        self,
        input_dim:int,
        codebook_size:int,
        codebook_scale=1.,
        entropy_loss_weight=1.,
        commitment_loss_weight=1.,
        diversity_gamma=1.
        ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_scale = codebook_scale
        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.diversity_gamma = diversity_gamma

        codebook_dim = int(log2(codebook_size))
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim,kernel_size=1)

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        
        # codes
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self,indices):
        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)
        codes = self.bits_to_codes(bits)
        codes = rearrange(codes, '... c d -> ... (c d)')
        codes = self.out_proj(codes)
        return codes
    
    def forward(self, x, inv_temperature=100):
        x = self.in_proj(x)
        x = rearrange(x, 'b d t -> b t 1 d')
        original_input = x
        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)
        indices = reduce((quantized > 0).int() * self.mask.int(), 'b t c d -> b t c', 'sum')

        if self.training:
            x = x + (quantized - x).detach()
        else:
            x = quantized

        if self.training:
            codebook = self.codebook
            distance = -2 * einsum('... i d, j d -> ... i j', original_input, codebook)
            prob = (-distance * inv_temperature).softmax(dim=-1)
            per_sample_entropy = (-(prob * prob.clamp(min=1e-5).log())).sum(dim=-1).mean()
            avg_prob = reduce(prob, '... 1 d -> 1 d', 'mean')
            codebook_entropy = (-(avg_prob * avg_prob.clamp(min=1e-5).log())).sum(dim=-1).mean()
            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            commit_loss = F.mse_loss(original_input, quantized.detach(), reduction='mean') if self.commitment_loss_weight > 0 else self.zero
        else:
            entropy_aux_loss = commit_loss = self.zero

        x = rearrange(x, 'b t 1 d -> b d t')
        x = self.out_proj(x)
        indices = rearrange(indices, 'b t 1 -> b t')
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        return x, indices, aux_loss

class ResidualLFQ(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_codebooks: int,
        codebook_size: int,
        quantizer_dropout: float = 0.0,
        entropy_loss_weight: float = 1,
        commitment_loss_weight: float = 1,
        diversity_gamma: float = 1.
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList([
            LFQ(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_scale=2 ** -i,
                entropy_loss_weight=entropy_loss_weight,
                commitment_loss_weight=commitment_loss_weight,
                diversity_gamma=diversity_gamma
            ) for i in range(n_codebooks)
        ])

    def forward(self, z, n_quantizers: int = None):

        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)


        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            z_q_i, indices_i, loss_i = quantizer(residual)
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i.detach()
            commitment_loss += (loss_i * mask).mean() 
            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=1)

        return z_q, codes, commitment_loss

if __name__ == "__main__":
    rlfq = ResidualLFQ(input_dim=512, n_codebooks=4, codebook_size=1024, quantizer_dropout=False)
    x = torch.randn(16, 512, 80)
    y = rlfq(x)
