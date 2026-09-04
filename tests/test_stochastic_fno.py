"""
tests/test_stochastic_fno.py
----------------------------
Tests unitaires pour le modele Stochastic Fourier Neural Operator (SFNO 2D).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from src.models.stochastic_fno import (
    StochasticFourierNeuralOperator2D,
    StochasticFourierNeuralOperator3D,
    SpectralConv2d,
    SpectralConv3d,
)


def test_spectral_conv_and_fno():
    print("=" * 80)
    print("[*] TEST DU STOCHASTIC FOURIER NEURAL OPERATOR (SFNO 2D & SFNO 3D VOLUMIQUE)")
    print("=" * 80)

    device = torch.device("cpu")

    # =========================================================================
    # 1. Test SFNO 2D (Front surfacique)
    # =========================================================================
    print("[1] Test SFNO 2D (Surfacique)...")
    model_2d = StochasticFourierNeuralOperator2D(
        in_channels=8,
        out_steps=7,
        width=32,
        modes1=8,
        modes2=8,
        latent_dim=16,
        num_layers=3
    ).to(device)

    x2d = torch.randn(2, 8, 64, 64, device=device)
    z2d = torch.randn(2, 16, device=device)

    out_2d = model_2d(x2d, z2d)
    print(f"  -> Shape de sortie SFNO 2D : {list(out_2d.shape)}")
    assert out_2d.shape == (2, 7, 64, 64), f"Shape FNO 2D invalide: {out_2d.shape}"
    assert (out_2d >= 0.0).all() and (out_2d <= 1.0).all(), "Probabilités hors [0, 1]."

    # Super-Resolution 2D Zero-Shot
    x_hires_2d = torch.randn(1, 8, 128, 128, device=device)
    out_hires_2d = model_2d(x_hires_2d)
    print(f"  -> Super-Résolution SFNO 2D (Zero-Shot) : {list(out_hires_2d.shape)}")
    assert out_hires_2d.shape == (1, 7, 128, 128)

    # Rollout Ensemble 2D
    mean_2d, std_2d, p90_2d = model_2d.sample_ensemble_rollout(x2d[:1], num_ensembles=10)
    assert mean_2d.shape == (7, 64, 64)

    # =========================================================================
    # 2. Test SFNO 3D (Panache & Thermodynamique Volumique)
    # =========================================================================
    print("[2] Test SFNO 3D (Volumique / Convection)...")
    model_3d = StochasticFourierNeuralOperator3D(
        in_channels=7,
        out_channels=7,
        width=24,
        modes1=4,
        modes2=6,
        modes3=6,
        latent_dim=16,
        num_layers=3
    ).to(device)

    # Entrée 3D : [B=2, C=7, D=12, H=32, W=32] (T, P, u, v, w, Carburant, MNT)
    x3d = torch.randn(2, 7, 12, 32, 32, device=device)
    z3d = torch.randn(2, 16, device=device)

    out_3d = model_3d(x3d, z3d)
    print(f"  -> Shape de sortie SFNO 3D : {list(out_3d.shape)}")
    assert out_3d.shape == (2, 7, 12, 32, 32), f"Shape SFNO 3D invalide: {out_3d.shape}"

    # Super-Résolution 3D Zero-Shot (ex: passage à une grille plus fine [16, 64, 64])
    x_hires_3d = torch.randn(1, 7, 16, 64, 64, device=device)
    out_hires_3d = model_3d(x_hires_3d)
    print(f"  -> Super-Résolution SFNO 3D (Zero-Shot) : {list(out_hires_3d.shape)}")
    assert out_hires_3d.shape == (1, 7, 16, 64, 64), "Échec de l'invariance de maillage SFNO 3D."

    # Rollout Ensemble 3D
    mean_3d, std_3d, p90_3d = model_3d.sample_ensemble_rollout_3d(x3d[:1], num_ensembles=6)
    print(f"  -> Ensemble SFNO 3D N=6 : Mean={list(mean_3d.shape)}, Std={list(std_3d.shape)}, P90={list(p90_3d.shape)}")
    assert mean_3d.shape == (7, 12, 32, 32)
    assert std_3d.shape == (7, 12, 32, 32)

    print("[OK] Modèles Stochastic Fourier Neural Operator (SFNO 2D & 3D) validés à 100%.")


if __name__ == "__main__":
    test_spectral_conv_and_fno()
