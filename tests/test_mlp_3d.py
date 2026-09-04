"""
tests/test_mlp_3d.py
--------------------
Tests unitaires pour le modèle ThermodynamicMLP3D (Continuous Spatio-Temporal Neural Field).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from src.models.mlp_3d import ThermodynamicMLP3D, FourierFeatureEncoder


def test_mlp_3d_thermodynamics():
    print("=" * 80)
    print("[*] TEST DU MODÈLE NEURAL IMPLICIT FIELD / MLP 3D THERMODYNAMIQUE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique d'exécution : {device}")

    # 1. Test de l'encodeur de Fourier
    encoder = FourierFeatureEncoder(in_dim=4, num_frequencies=6, include_input=True).to(device)
    coords = torch.randn(10, 4, device=device) # (x, y, z, t)
    enc_out = encoder(coords)
    expected_dim = 4 + 2 * 4 * 6 # 4 + 48 = 52
    print(f"Dimension encodée Fourier : {enc_out.shape} (attendu: [10, {expected_dim}])")
    assert enc_out.shape == (10, expected_dim)

    # 2. Test du modèle MLP 3D
    mlp = ThermodynamicMLP3D(
        coord_dim=4,
        forcing_dim=3,
        out_dim=7,
        hidden_dim=64,
        num_layers=4,
        num_frequencies=6
    ).to(device)

    # Évaluation normalisée standardisée
    N = 32
    sample_coords = torch.randn(N, 4, device=device) # [N, (x, y, z, t)]
    sample_wind = torch.tensor([1.2, -0.6, 0.1], device=device) # Vent global

    norm_outputs = mlp(sample_coords, sample_wind)
    print(f"Sorties normalisées standardisées MLP 3D sur {N} points : {list(norm_outputs.shape)}")
    assert norm_outputs.shape == (N, 7)

    # Évaluation physique SI
    phys_outputs = mlp.forward_physical(sample_coords, sample_wind)
    print(f"Sorties physiques SI MLP 3D : T min={phys_outputs[:, 0].min():.1f}K, max={phys_outputs[:, 0].max():.1f}K")
    assert (phys_outputs[:, 0] >= 298.15).all(), "La température doit être >= T_ambiante (298.15 K)."
    assert (phys_outputs[:, 5] >= 0.0).all(), "La densité de suie doit être >= 0."
    assert (phys_outputs[:, 6] >= 0.0).all(), "Le débit calorifique HRR doit être >= 0."

    # 3. Test de l'évaluation volumique dense continue
    grid_eval = mlp.evaluate_grid_3d(
        grid_shape=(8, 16, 16), # (D=8, H=16, W=16)
        domain_bounds=(-200.0, 200.0, -200.0, 200.0, 0.0, 300.0),
        time_val=15.0, # t = 15 min
        wind_vec=sample_wind,
        device=device
    )

    print(f"Grille dense 3D évaluée :")
    print(f"  -> Température 3D [D, H, W] : {list(grid_eval['temperature_k'].shape)}")
    print(f"  -> Vitesse verticale W 3D   : {list(grid_eval['w'].shape)}")
    print(f"  -> Densité de suie 3D       : {list(grid_eval['soot_density'].shape)}")

    assert grid_eval["temperature_k"].shape == (8, 16, 16)
    assert grid_eval["w"].shape == (8, 16, 16)
    assert grid_eval["soot_density"].shape == (8, 16, 16)

    # 4. Test d'entraînement & flux de gradient sur données normalisées
    target_norm = torch.randn(N, 7, device=device)
    loss = torch.nn.functional.mse_loss(norm_outputs, target_norm)
    loss.backward()
    print(f"Perte MSE normalisée & rétropropagation validée (Loss = {loss.item():.4f})")
    assert loss.item() < 10.0, f"La perte normalisée doit être d'ordre 1 (actuel: {loss.item()})"

    print("[OK] Modèle ThermodynamicMLP3D validé à 100%.")


if __name__ == "__main__":
    test_mlp_3d_thermodynamics()
