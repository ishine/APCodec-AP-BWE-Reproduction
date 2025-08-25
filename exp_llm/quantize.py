import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from vector_quantize_pytorch.finite_scalar_quantization import FSQ

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class FSQWrapper(nn.Module):
    def __init__(
        self,
        levels,
        input_dim,
        codebook_dim,
        num_codebooks,
        keep_num_codebooks_dim,
        channel_first,
        projection_has_bias,
        return_indices,
        force_quantization_f32,
        preserve_symmetry,
        noise_dropout
    ):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

        self.fsq = FSQ(
            levels=levels,
            dim=codebook_dim,
            num_codebooks=num_codebooks,
            keep_num_codebooks_dim=keep_num_codebooks_dim,
            channel_first=channel_first,
            projection_has_bias=projection_has_bias,
            return_indices=return_indices,
            force_quantization_f32=force_quantization_f32,
            preserve_symmetry=preserve_symmetry,
            noise_dropout=noise_dropout
        )

    def forward(self, x):
        z_e = self.in_proj(x)
        z_q, indices = self.fsq(z_e)
        z_q = self.out_proj(z_q)
        return z_q, indices, z_e

    def indices_to_codes(self, indices):
        codes = self.fsq.indices_to_codes(indices)
        return self.out_proj(codes)

class ResidualFSQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_quantizers: int = 4,
        codebook_size: int = 1024,
        codebook_dim: int = 32,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_quantizers = n_quantizers
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        levels = [4, 4, 4, 4, 4]
        assert torch.prod(torch.tensor(levels)).item() == codebook_size, \
            f"Levels {levels} produce codebook size {torch.prod(torch.tensor(levels)).item()}, expected {codebook_size}"

        self.quantizers = nn.ModuleList([
            FSQWrapper(
                levels=levels,
                input_dim=input_dim,
                codebook_dim=codebook_dim,
                num_codebooks=1,
                keep_num_codebooks_dim=False,
                channel_first=True,
                projection_has_bias=True,
                return_indices=True,
                force_quantization_f32=True,
                preserve_symmetry=False,
                noise_dropout=quantizer_dropout
            ) for _ in range(n_quantizers)
        ])

    def forward(self, z, n_quantizers: int = None):
        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = 0.0
        codebook_loss = 0.0
        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_quantizers
        if self.training and self.quantizer_dropout > 0:
            n_quantizers_tensor = torch.ones((z.shape[0],), device=z.device) * self.n_quantizers
            dropout = torch.randint(1, self.n_quantizers + 1, (z.shape[0],), device=z.device)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers_tensor[:n_dropout] = dropout[:n_dropout]
        else:
            n_quantizers_tensor = torch.full((z.shape[0],), n_quantizers, device=z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, indices_i, z_e_i = quantizer(residual)

            mask = (torch.full((z.shape[0],), i, device=z.device) < n_quantizers_tensor).float()[:, None, None]
            z_q = z_q + z_q_i * mask
            residual = residual - z_q_i

            commitment_loss_i = F.mse_loss(z_e_i, z_q_i.detach(), reduction="none").mean([1, 2])
            codebook_loss_i = F.mse_loss(z_q_i, z_e_i.detach(), reduction="none").mean([1, 2])
            commitment_loss += (commitment_loss_i * mask.squeeze()).mean()
            codebook_loss += (codebook_loss_i * mask.squeeze()).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)  # [B, N, T]
        latents = torch.cat(latents, dim=1)  # [B, N*D, T]

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = torch.zeros(codes.shape[0], self.input_dim, codes.shape[2], device=codes.device)
        for i in range(self.n_quantizers):
            z_p_i = self.quantizers[i].indices_to_codes(codes[:, i, :])
            z_q = z_q + z_p_i
        return z_q

    def from_latents(self, latents: torch.Tensor):
        z_q = torch.zeros(latents.shape[0], self.input_dim, latents.shape[2], device=latents.device)
        z_p = []
        codes = []
        dims = torch.cumsum(torch.tensor([0] + [self.codebook_dim] * self.n_quantizers, device=latents.device), dim=0)
        for i in range(self.n_quantizers):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i, _ = self.quantizers[i](latents[:, j:k, :])
            z_q = z_q + z_p_i
            z_p.append(self.quantizers[i].in_proj(z_p_i))
            codes.append(codes_i)
        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

if __name__ == "__main__":
    model = ResidualFSQ(input_dim=32, n_quantizers=4, codebook_size=1024, codebook_dim=32, quantizer_dropout=False)
    x = torch.randn(2, 32, 25)
    latent, codes, _, _, _ = model(x)
    res_latent = model.from_codes(codes)
    print(f"latent:{latent}")
    print(f"res_latent:{res_latent}")
    print("Reconstructed z_q shape:", res_latent.shape)
    print("Difference between z_q and reconstructed_z_q:", torch.norm(latent - res_latent))