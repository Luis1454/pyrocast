"""
src/models/stochastic_fno.py
----------------------------
Architecture Stochastic Fourier Neural Operator (SFNO 2D) pour la prediction
spatio-temporelle de propagation de feux de foret sous incertitude.
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    Couche de convolution spectrale 2D dans le domaine de Fourier.
    Applique une transformation lineaire parametree sur les k_max premiers modes de Fourier.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor, latent_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        w1 = self.weights1
        w2 = self.weights2
        if latent_mod is not None:
            mod = latent_mod.view(batchsize, self.in_channels, 1, 1).to(torch.cfloat)
            w1_mod = w1.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
            w2_mod = w2.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
        else:
            w1_mod = None
            w2_mod = None

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, x_ft.shape[-2] // 2)
        m2 = min(self.modes2, x_ft.shape[-1])

        if w1_mod is not None:
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,bioxy->boxy", x_ft[:, :, :m1, :m2], w1_mod[:, :, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,bioxy->boxy", x_ft[:, :, -m1:, :m2], w2_mod[:, :, :, :m1, :m2])
        else:
            out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2[:, :, -m1:, :m2])

        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x_out


class StochasticFourierNeuralOperator2D(nn.Module):
    """
    Modele predictif Stochastic Fourier Neural Operator (SFNO 2D).
    Entree : [Batch, in_channels=8, H, W]
    Sortie : [Batch, out_steps, H, W] (Probabilite spatiale de propagation)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_steps: int = 7,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        latent_dim: int = 16,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_steps = out_steps
        self.width = width
        self.latent_dim = latent_dim

        self.p = nn.Conv2d(in_channels, self.width, kernel_size=1)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width)
        )

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(num_layers)
        ])

        self.q = nn.Sequential(
            nn.Conv2d(self.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_steps, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, z_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if z_noise is None:
            z_noise = torch.zeros((B, self.latent_dim), device=x.device)

        latent_mod = self.latent_encoder(z_noise)
        feat = self.p(x)

        for spec_layer, spat_layer in zip(self.spectral_layers, self.spatial_layers):
            x1 = spec_layer(feat, latent_mod=latent_mod)
            x2 = spat_layer(feat)
            feat = F.gelu(x1 + x2)

        out = self.q(feat)
        prob_out = torch.sigmoid(out)
        return prob_out

    @torch.no_grad()
    def sample_ensemble_rollout(
        self,
        input_tensor: torch.Tensor,
        num_ensembles: int = 30,
        noise_std: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        device = input_tensor.device
        batch_input = input_tensor.repeat(num_ensembles, 1, 1, 1)
        z_samples = torch.randn((num_ensembles, self.latent_dim), device=device) * noise_std

        ensemble_preds = self(batch_input, z_samples)
        mean_prob = torch.mean(ensemble_preds, dim=0)
        std_prob = torch.std(ensemble_preds, dim=0)
        p90_prob = torch.quantile(ensemble_preds, 0.90, dim=0)

        return mean_prob, std_prob, p90_prob


class SpectralConv3d(nn.Module):
    """
    Couche de convolution spectrale 3D dans le domaine de Fourier.
    Applique une transformation linéaire paramétrée sur les modes (modes1, modes2, modes3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", input_tensor, weights)

    def forward(self, x: torch.Tensor, latent_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        batchsize = x.shape[0]
        # x shape : [B, C, D, H, W]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        w1, w2, w3, w4 = self.weights1, self.weights2, self.weights3, self.weights4
        if latent_mod is not None:
            mod = latent_mod.view(batchsize, self.in_channels, 1, 1, 1).to(torch.cfloat)
            w1_mod = w1.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
            w2_mod = w2.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
            w3_mod = w3.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
            w4_mod = w4.unsqueeze(0) * (1.0 + 0.15 * mod.unsqueeze(2))
        else:
            w1_mod = w2_mod = w3_mod = w4_mod = None

        D, H, W = x.size(-3), x.size(-2), x.size(-1)
        out_ft = torch.zeros(
            batchsize, self.out_channels, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, max(1, D // 2))
        m2 = min(self.modes2, max(1, H // 2))
        m3 = min(self.modes3, x_ft.shape[-1])

        if w1_mod is not None:
            out_ft[:, :, :m1, :m2, :m3] = torch.einsum("bixyz,bioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], w1_mod[:, :, :, :m1, :m2, :m3])
            out_ft[:, :, -m1:, :m2, :m3] = torch.einsum("bixyz,bioxyz->boxyz", x_ft[:, :, -m1:, :m2, :m3], w2_mod[:, :, :, :m1, :m2, :m3])
            out_ft[:, :, :m1, -m2:, :m3] = torch.einsum("bixyz,bioxyz->boxyz", x_ft[:, :, :m1, -m2:, :m3], w3_mod[:, :, :, :m1, :m2, :m3])
            out_ft[:, :, -m1:, -m2:, :m3] = torch.einsum("bixyz,bioxyz->boxyz", x_ft[:, :, -m1:, -m2:, :m3], w4_mod[:, :, :, :m1, :m2, :m3])
        else:
            out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], w1[:, :, :m1, :m2, :m3])
            out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], w2[:, :, :m1, :m2, :m3])
            out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], w3[:, :, :m1, :m2, :m3])
            out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], w4[:, :, :m1, :m2, :m3])

        x_out = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))
        return x_out


class StochasticFourierNeuralOperator3D(nn.Module):
    """
    Modèle prédictif Stochastic Fourier Neural Operator 3D (SFNO 3D Volumique).
    Entrée : [Batch, in_channels=7, D, H, W] (Température, Pression, U, V, W, Carburant, MNT)
    Sortie : [Batch, out_channels=7, D, H, W] (État thermo-dynamique 3D transporté)
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 7,
        width: int = 32,
        modes1: int = 6,
        modes2: int = 6,
        modes3: int = 6,
        latent_dim: int = 16,
        num_layers: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.latent_dim = latent_dim

        self.p = nn.Conv3d(in_channels, self.width, kernel_size=1)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width)
        )

        self.spectral_layers = nn.ModuleList([
            SpectralConv3d(self.width, self.width, modes1, modes2, modes3) for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(num_layers)
        ])

        self.q = nn.Sequential(
            nn.Conv3d(self.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, z_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, D, H, W = x.shape
        if z_noise is None:
            z_noise = torch.zeros((B, self.latent_dim), device=x.device)

        latent_mod = self.latent_encoder(z_noise)
        feat = self.p(x)

        for spec_layer, spat_layer in zip(self.spectral_layers, self.spatial_layers):
            x1 = spec_layer(feat, latent_mod=latent_mod)
            x2 = spat_layer(feat)
            feat = F.gelu(x1 + x2)

        out = self.q(feat)
        return out

    @torch.no_grad()
    def sample_ensemble_rollout_3d(
        self,
        input_tensor: torch.Tensor,
        num_ensembles: int = 10,
        noise_std: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        device = input_tensor.device
        batch_input = input_tensor.repeat(num_ensembles, 1, 1, 1, 1)
        z_samples = torch.randn((num_ensembles, self.latent_dim), device=device) * noise_std

        ensemble_preds = self(batch_input, z_samples)
        mean_field = torch.mean(ensemble_preds, dim=0)
        std_field = torch.std(ensemble_preds, dim=0)
        p90_field = torch.quantile(ensemble_preds, 0.90, dim=0)

        return mean_field, std_field, p90_field
