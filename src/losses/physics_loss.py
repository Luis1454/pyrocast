"""
src/losses/physics_loss.py
--------------------------
Fonctions de perte pour équations aux dérivées partielles raides et combustion :
- L2 Loss
- Sobolev H1 (semi-norme spatiale) : force la conservation des gradients et la netteté du front
- Spectral FFT Loss : force la conservation des hautes fréquences et des micro-tourbillons
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn


class SobolevH1Loss(nn.Module):
    """
    Pénalité sur les gradients spatiaux :
        L_{H1} = || grad(u_pred) - grad(u_gt) ||^2
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        if self.dim == 2:
            gx = diff[..., :, 1:] - diff[..., :, :-1]
            gy = diff[..., 1:, :] - diff[..., :-1, :]
            return torch.mean(gx**2) + torch.mean(gy**2)
        elif self.dim == 3:
            gx = diff[..., :, :, 1:] - diff[..., :, :, :-1]
            gy = diff[..., :, 1:, :] - diff[..., :, :-1, :]
            gz = diff[..., 1:, :, :] - diff[..., :-1, :, :]
            return torch.mean(gx**2) + torch.mean(gy**2) + torch.mean(gz**2)
        raise ValueError(f"Dimension {self.dim} non supportée")


class SpectralFFTLoss(nn.Module):
    """
    Pénalité dans l'espace des nombres d'ondes (Fourier) avec pondération passe-haut.
    """

    def __init__(self, high_freq_gamma: float = 1.2):
        super().__init__()
        self.gamma = high_freq_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        spatial_dims = tuple(range(2, pred.dim()))
        fft_p = torch.fft.rfftn(pred, dim=spatial_dims, norm="ortho")
        fft_t = torch.fft.rfftn(target, dim=spatial_dims, norm="ortho")

        diff_sq = torch.abs(fft_p - fft_t) ** 2

        orig_shapes = pred.shape[2:]
        grids = [
            torch.fft.fftfreq(orig_shapes[i], d=1.0).to(pred.device) if i < len(orig_shapes) - 1
            else torch.fft.rfftfreq(orig_shapes[i], d=1.0).to(pred.device)
            for i in range(len(orig_shapes))
        ]
        mesh = torch.meshgrid(*grids, indexing="ij")
        k_sq = sum(m**2 for m in mesh)
        weights = (1.0 + k_sq) ** self.gamma

        return torch.mean(diff_sq * weights)


class CombinedPhysicsLoss(nn.Module):
    """
    Fonction de perte composite physique complète.
    """

    def __init__(
        self,
        lambda_l2: float = 1.0,
        lambda_h1: float = 0.5,
        lambda_spec: float = 0.1,
        dim: int = 2,
    ):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_h1 = lambda_h1
        self.lambda_spec = lambda_spec

        self.l2 = nn.MSELoss()
        self.h1 = SobolevH1Loss(dim=dim)
        self.spec = SpectralFFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l2_val = self.l2(pred, target)
        h1_val = self.h1(pred, target)
        spec_val = self.spec(pred, target)

        total = self.lambda_l2 * l2_val + self.lambda_h1 * h1_val + self.lambda_spec * spec_val

        return total, {
            "loss_total": total.item(),
            "loss_l2": l2_val.item(),
            "loss_h1": h1_val.item(),
            "loss_spectral": spec_val.item(),
        }
