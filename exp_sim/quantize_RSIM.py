from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from einx import get_at
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class SimVQ(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        rotation_trick: bool = True,
        input_to_quantize_commit_loss_weight: float = 0.25,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight

        codebook = torch.randn(codebook_size, codebook_dim) * (codebook_dim ** -0.5)
        self.register_buffer('frozen_codebook', codebook)
        self.codebook_transform = WNConv1d(codebook_dim, input_dim, kernel_size=1)

    @property
    def codebook(self):
        return self.codebook_transform(self.frozen_codebook.unsqueeze(-1)).squeeze(-1)

    def indices_to_codes(self, indices):
        frozen_codes = get_at('[c] d, b t -> b t d', self.frozen_codebook, indices)
        quantized = self.codebook_transform(frozen_codes.unsqueeze(-1)).squeeze(-1)
        return quantized

    def forward(self, z):

        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(z, implicit_codebook)
            indices = dist.argmin(dim=-1)
        z_q = get_at('[c] d, b t -> b t d', implicit_codebook, indices)
        commit_loss = (F.mse_loss(z.detach(), z_q) +F.mse_loss(z, z_q.detach()) * self.input_to_quantize_commit_loss_weight)
        if self.rotation_trick:
            z_q = rotate_to(z, z_q)
        else:
            z_q = (z_q - x).detach() + z

        return z_q, indices, commit_loss * self.commitment_weight

class ResidualSimVQ(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        rotation_trick: bool = True,
        input_to_quantize_commit_loss_weight: float = 0.25,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList(
            [
                SimVQ(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim[i],
                    rotation_trick=rotation_trick,
                    input_to_quantize_commit_loss_weight=input_to_quantize_commit_loss_weight,
                    commitment_weight=commitment_weight
                )
                for i in range(n_codebooks)
            ]
        )

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_i = rearrange(residual, 'b d t -> b t d')
            z_q_i, indices_i, commit_loss_i = quantizer(z_i)
            z_q_i = rearrange(z_q_i, 'b t d -> b d t')
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss += (commit_loss_i * mask).mean()
            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=1)

        return  z_q, codes, commitment_loss,

if __name__ == "__main__":
    rvq = ResidualSimVQ(quantizer_dropout=0.2)
    x = torch.randn(16, 512, 80)
    y = rvq(x)