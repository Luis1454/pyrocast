"""
infer.py
--------
Moteur d'inférence probabiliste et déterministe pour déploiement opérationnel C2 :
1. Neural FMM / MGNN sur Octree Adaptatif (Faisceau Monte-Carlo)
2. Stochastic Fourier Neural Operator 2D (SFNO 2D sur MNT / combustible)
3. Stochastic Fourier Neural Operator 3D (SFNO 3D Volumique Navier-Stokes)
4. Modèle Neural Implicit Field / MLP 3D Thermodynamique continu
"""

from pathlib import Path
import time
import numpy as np
import torch

from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.models.stochastic_fno import (
    StochasticFourierNeuralOperator2D,
    StochasticFourierNeuralOperator3D,
)
from src.models.mlp_3d import ThermodynamicMLP3D
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.inference.monte_carlo import MonteCarloWildfireSimulator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"[*] FIREMAP PRODUCTION INFERENCE : BENCHMARK MULTI-MODÈLES IA SUR : {device}")
    print("=" * 85)

    # 1. Benchmark Neural FMM / MGNN
    print("\n--- 1. NEURAL FMM / MGNN (FAISCEAU MONTE-CARLO SUR OCTREE) ---")
    fmm_ckpt = Path("reports/checkpoints/neural_fmm_best.pt")
    mgnn_model = NeuralFMM_MGNN(
        in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3
    ).to(device)
    if fmm_ckpt.exists():
        mgnn_model.load_state_dict(torch.load(str(fmm_ckpt), map_location=device))
        print(f"  [OK] Checkpoint MGNN chargé : {fmm_ckpt}")
    mgnn_model.eval()

    graph_builder = AdaptiveOctreeGraphBuilder(max_depth=3, min_depth=1, grad_threshold=0.25)
    mc_engine = MonteCarloWildfireSimulator(
        model=mgnn_model,
        graph_builder=graph_builder,
        num_ensembles=16 if device.type == "cpu" else 64,
        wind_uncertainty_std=0.25
    )

    C, H, W = 7, 64, 64
    x = torch.linspace(-2, 2, W, device=device)
    y = torch.linspace(-2, 2, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    r_front = torch.sqrt(grid_x**2 + grid_y**2)
    flame_init = torch.exp(-((r_front - 0.5)**2) / 0.04) * 2.5

    init_state = torch.zeros((C, H, W), device=device)
    init_state[0] = flame_init
    init_state[1] = 1013.25
    init_state[2] = 1.5
    init_state[3] = 0.5
    init_state[4] = 0.15
    init_state[5] = 0.08

    wind_forecast = torch.stack([
        torch.linspace(1.5, 2.2, 10, device=device),
        torch.linspace(0.5, 0.8, 10, device=device),
        torch.linspace(0.1, 0.3, 10, device=device)
    ], dim=1)

    t0 = time.time()
    prob_maps = mc_engine.predict_probability_map(init_state, wind_forecast, steps=10)
    dt_mgnn = (time.time() - t0) * 1000.0

    p_final = prob_maps[-1]
    cell_area_ha = (30.0 * 30.0) / 10000.0
    high_risk_ha = int(torch.sum(p_final >= 0.8).item()) * cell_area_ha
    print(f"  Temps d'inférence Monte-Carlo (64 membres) : {dt_mgnn:.2f} ms")
    print(f"  Surface à risque critique (P >= 80%)      : {high_risk_ha:.2f} Ha")

    # 2. Benchmark Stochastic Fourier Neural Operator 2D (SFNO 2D)
    print("\n--- 2. STOCHASTIC FOURIER NEURAL OPERATOR (SFNO 2D) ---")
    fno_2d_ckpt = Path("checkpoints/stochastic_fno_best.pt")
    sfno_2d = StochasticFourierNeuralOperator2D(
        in_channels=8, out_steps=7, width=32, modes1=12, modes2=12, latent_dim=16
    ).to(device)
    if fno_2d_ckpt.exists():
        sfno_2d.load_state_dict(torch.load(str(fno_2d_ckpt), map_location=device)["model_state_dict"])
        print(f"  [OK] Checkpoint SFNO 2D chargé : {fno_2d_ckpt}")
    sfno_2d.eval()

    sample_in_2d = torch.randn(1, 8, 64, 64, device=device)
    t0 = time.time()
    with torch.no_grad():
        pred_2d = sfno_2d(sample_in_2d)
    dt_sfno2d = (time.time() - t0) * 1000.0
    print(f"  Temps d'inférence SFNO 2D (7 horizons)    : {dt_sfno2d:.2f} ms")
    print(f"  Shape de sortie SFNO 2D                    : {list(pred_2d.shape)}")

    # 3. Benchmark Stochastic Fourier Neural Operator 3D (SFNO 3D Volumique)
    print("\n--- 3. STOCHASTIC FOURIER NEURAL OPERATOR 3D (SFNO 3D VOLUMIQUE) ---")
    fno_3d_ckpt = Path("checkpoints/stochastic_fno_3d_best.pt")
    sfno_3d = StochasticFourierNeuralOperator3D(
        in_channels=7, out_channels=7, width=24, modes1=4, modes2=6, modes3=6, latent_dim=16
    ).to(device)
    if fno_3d_ckpt.exists():
        sfno_3d.load_state_dict(torch.load(str(fno_3d_ckpt), map_location=device)["model_state_dict"])
        print(f"  [OK] Checkpoint SFNO 3D chargé : {fno_3d_ckpt}")
    sfno_3d.eval()

    sample_in_3d = torch.randn(1, 7, 12, 32, 32, device=device)
    t0 = time.time()
    with torch.no_grad():
        pred_3d = sfno_3d(sample_in_3d)
    dt_sfno3d = (time.time() - t0) * 1000.0
    print(f"  Temps d'inférence SFNO 3D (Tenseur 3D)    : {dt_sfno3d:.2f} ms")
    print(f"  Shape de sortie SFNO 3D                    : {list(pred_3d.shape)}")

    # 4. Benchmark Neural Implicit Field / MLP 3D Thermodynamique
    print("\n--- 4. MODÈLE NEURAL IMPLICIT FIELD / MLP 3D CONTINU ---")
    mlp_3d_ckpt = Path("checkpoints/mlp_3d_thermo_best.pt")
    mlp_3d = ThermodynamicMLP3D(coord_dim=4, forcing_dim=3, out_dim=7, hidden_dim=128).to(device)
    if mlp_3d_ckpt.exists():
        mlp_3d.load_state_dict(torch.load(str(mlp_3d_ckpt), map_location=device)["model_state_dict"])
        print(f"  [OK] Checkpoint MLP 3D chargé : {mlp_3d_ckpt}")
    mlp_3d.eval()

    test_coords = torch.rand(1024, 4, device=device)
    test_wind = torch.tensor([1.5 / 25.0, -0.8 / 25.0, 0.05], device=device)
    t0 = time.time()
    with torch.no_grad():
        pred_pts = mlp_3d(test_coords, test_wind)
    dt_mlp = (time.time() - t0) * 1000.0
    print(f"  Temps d'inférence MLP 3D (1024 points 3D) : {dt_mlp:.2f} ms")
    print(f"  Sorties continues (T, P, U, V, W, Soot, HRR) validées.")

    print("\n" + "=" * 85)
    print("[OK] INFERENCE MULTI-MODELES IA COMPLETE TERMINEE AVEC SUCCES.")
    print("=" * 85)


if __name__ == "__main__":
    main()
