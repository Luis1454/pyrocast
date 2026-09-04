"""
train.py
--------
Pipeline complet d'entraînement autorégressif multi-époques du Neural FMM / MGNN :
- Trajectoires thermodynamiques avec forçage météorologique 3D
- Optimiseur AdamW avec décroissance Cosine Annealing
- Fonction de perte composite de Sobolev H^1 + Spectrale FFT
- Sauvegarde automatique des checkpoints de modèle optimisés
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.models.stochastic_fno import (
    StochasticFourierNeuralOperator2D,
    StochasticFourierNeuralOperator3D,
)
from src.models.mlp_3d import ThermodynamicMLP3D
from src.data.historical_wildfire_dataset import HistoricalWildfireDataset
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.losses.physics_loss import CombinedPhysicsLoss
from src.training.rollout_trainer import RolloutTrainer
from src.physics.fire_engineering import StochasticMonteCarloSpreadSimulator
from src.physics.cfd_3d_engine import NavierStokesCombustion3DSolver, CFD3DState


def generate_physical_trajectory(T_steps: int = 5, H: int = 64, W: int = 64, device: torch.device = torch.device("cpu")):
    """
    Génère une séquence de propagation physique authentique issue du solveur Rothermel / Byram / Monte Carlo.
    """
    x_i, y_i = np.meshgrid(np.arange(W), np.arange(H))
    elev = 280.0 + np.sin(x_i * 0.1) * 45.0 + np.cos(y_i * 0.1) * 35.0
    sim = StochasticMonteCarloSpreadSimulator(elev, dx_meters=30.0, fuel_code="SH5")
    wind_spd = float(np.random.uniform(35.0, 55.0))
    fmc = float(np.random.uniform(4.0, 9.0))
    res = sim.simulate_ensemble(
        ignition_point_px=(H // 2, W // 2 - 4),
        wind_speed_kmh=wind_spd,
        wind_dir_cardinal="NW",
        fmc_pct=fmc,
        num_ensembles=6,
        num_output_steps=T_steps
    )
    trajectory = []
    wind_sequence = []
    w_rad = math.radians(315.0)
    u_w = (wind_spd / 3.6) * math.sin(w_rad)
    v_w = -(wind_spd / 3.6) * math.cos(w_rad)

    for t_idx, fr in enumerate(res["frames"]):
        prob_2d = np.array(fr["prob_map_flat"], dtype=np.float32).reshape((H, W)) / 100.0
        state_t = torch.zeros((7, H, W), device=device)
        state_t[0] = torch.from_numpy(prob_2d).to(device)                 # Température normalisée [0, 1]
        state_t[1] = torch.zeros((H, W), device=device)                   # Perturbation de pression normalisée
        state_t[2] = torch.full((H, W), u_w / 25.0, device=device)        # Vent U normalisé [-1, 1]
        state_t[3] = torch.full((H, W), v_w / 25.0, device=device)        # Vent V normalisé [-1, 1]
        state_t[4] = torch.from_numpy(prob_2d).to(device) * (12.5 / 25.0) # Updraft W normalisé [0, 1]
        state_t[5] = torch.full((H, W), fmc / 100.0, device=device)       # FMC normalisé [0, 1]
        state_t[6] = torch.from_numpy(prob_2d).to(device)                 # Flux calorifique normalisé [0, 1]
        trajectory.append(state_t)
        wind_sequence.append(torch.tensor([u_w / 25.0, v_w / 25.0, 0.05], device=device))

    return torch.stack(trajectory), torch.stack(wind_sequence)


def generate_cfd_3d_rollout(D: int = 12, H: int = 32, W: int = 32, steps: int = 4, device: torch.device = torch.device("cpu")):
    """
    Génère une séquence volumique 3D authentique résolue par Navier-Stokes Boussinesq + Pyrolyse.
    """
    dx, dz, dt = 25.0, 15.0, 0.4
    solver = NavierStokesCombustion3DSolver(
        grid_shape=(D, H, W),
        dx=dx,
        dz=dz,
        dt=dt,
        ambient_temp_k=298.15,
        canopy_height_m=12.0,
        canopy_drag_cd=0.20
    )

    u_ref = float(np.random.uniform(8.0, 16.0))
    z_coords = torch.linspace(1.0, D * dz, D, device=device)
    u_profile = u_ref * torch.log(torch.clamp(z_coords / 0.5, min=1.1)) / np.log(30.0 / 0.5)

    u_3d = u_profile.view(D, 1, 1).expand(D, H, W).clone()
    v_3d = torch.zeros((D, H, W), device=device)
    w_3d = torch.zeros((D, H, W), device=device)
    t_3d = torch.full((D, H, W), 298.15, device=device)

    # Allumage thermique au sol
    cx = float(np.random.uniform(0.3, 0.5) * (W - 1) * dx)
    cy = float(np.random.uniform(0.4, 0.6) * (H - 1) * dx)
    x_grid, y_grid = torch.meshgrid(torch.linspace(0, (W - 1) * dx, W, device=device), torch.linspace(0, (H - 1) * dx, H, device=device), indexing="xy")
    dist_ign = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    flame_g = torch.exp(-(dist_ign**2) / (50.0**2)) * float(np.random.uniform(1100.0, 1400.0))
    t_3d[0] += flame_g
    t_3d[1] += flame_g * 0.5

    solid_fuel = torch.zeros((D, H, W), device=device)
    solid_fuel[:2, :, :] = 1.2
    moisture = torch.zeros((D, H, W), device=device)
    moisture[:2, :, :] = 0.08
    pressure = torch.zeros((D, H, W), device=device)

    state = CFD3DState(u=u_3d, v=v_3d, w=w_3d, temperature_k=t_3d, solid_fuel_density=solid_fuel, moisture_density=moisture, pressure=pressure)

    states_3d = []
    for _ in range(steps):
        state, _ = solver.step(state)
        # Normalisation physique standardisée [7, D, H, W]
        t_norm = torch.clamp((state.temperature_k - 298.15) / 1400.0, 0.0, 1.0)
        p_norm = torch.clamp(state.pressure / 200.0, -1.0, 1.0)
        u_norm = torch.clamp(state.u / 25.0, -1.0, 1.0)
        v_norm = torch.clamp(state.v / 25.0, -1.0, 1.0)
        w_norm = torch.clamp(state.w / 25.0, -1.0, 1.0)
        fuel_norm = torch.clamp(state.solid_fuel_density / 1.5, 0.0, 1.0)
        moist_norm = torch.clamp(state.moisture_density / 0.2, 0.0, 1.0)
        tensor_7c = torch.stack([t_norm, p_norm, u_norm, v_norm, w_norm, fuel_norm, moist_norm], dim=0)
        states_3d.append(tensor_7c)

    return torch.stack(states_3d)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints_dir = Path("reports/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] ENTRAÎNEMENT PHYSIQUE AUTORÉGRESSIF ACCÉLÉRÉ SUR : {device} ({device_name})")
    print("=" * 80)

    # 1. Modèle et constructeur d'Octree
    model = NeuralFMM_MGNN(
        in_channels=7,
        hidden_dim=64,
        out_channels=7,
        num_layers=3,
        forcing_dim=3  # Vent 3D (U, V, W)
    ).to(device)

    graph_builder = AdaptiveOctreeGraphBuilder(
        max_depth=3,
        min_depth=1,
        grad_threshold=0.25,
        r_near=0.12,
        k_far=6
    )

    criterion = CombinedPhysicsLoss(
        lambda_l2=1.0,
        lambda_h1=0.5,
        lambda_spec=0.15,
        dim=2
    ).to(device)

    trainer = RolloutTrainer(
        model=model,
        graph_builder=graph_builder,
        criterion=criterion,
        lr=1e-3,
        rollout_horizon=3,
        noise_std=0.01
    )

    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=8, eta_min=1e-5)

    # 2. Boucle d'entraînement MGNN
    num_epochs = 6
    print(f"[*] Optimisation Neural FMM sur {num_epochs} époques physiques...")
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        trajectory, wind_seq = generate_physical_trajectory(T_steps=5, H=64, W=64, device=device)
        metrics = trainer.train_step(trajectory, wind_seq)
        scheduler.step()

        cur_loss = metrics.get("loss_total", 0.0)
        l2_l = metrics.get("loss_l2", cur_loss)
        h1_l = metrics.get("loss_h1", 0.0)
        fft_l = metrics.get("loss_spectral", 0.0)
        lr_cur = scheduler.get_last_lr()[0]

        print(f"  Époque {epoch:02d}/{num_epochs:02d} | Perte Normalisée: {cur_loss:.6f} (L2: {l2_l:.5f}, H1: {h1_l:.5f}, FFT: {fft_l:.5f}) | LR: {lr_cur:.2e}")

        if cur_loss < best_loss:
            best_loss = cur_loss
            ckpt_path = checkpoints_dir / "neural_fmm_best.pt"
            torch.save(model.state_dict(), str(ckpt_path))

    print(f"[OK] Checkpoint Neural FMM sauvegardé : {checkpoints_dir / 'neural_fmm_best.pt'} (Perte: {best_loss:.6f})")

    # 3. Entraînement du Stochastic Fourier Neural Operator (SFNO 2D) sur le Dataset Physique
    print("-" * 80)
    print("[*] ENTRAÎNEMENT DU STOCHASTIC FOURIER NEURAL OPERATOR (SFNO 2D)...")
    dataset = HistoricalWildfireDataset(num_samples=48, grid_size=64, dx_meters=30.0)
    loader = DataLoader(dataset, batch_size=8 if torch.cuda.is_available() else 4, shuffle=True)

    fno_model = StochasticFourierNeuralOperator2D(
        in_channels=8, out_steps=7, width=32, modes1=12, modes2=12, latent_dim=16, num_layers=4
    ).to(device)
    optimizer = optim.AdamW(fno_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_fn = CombinedPhysicsLoss(lambda_l2=1.0, lambda_h1=0.4, lambda_spec=0.1, dim=2).to(device)

    best_fno_loss = float("inf")
    num_fno_epochs = 6
    for ep in range(1, num_fno_epochs + 1):
        fno_model.train()
        total_ep_loss = 0.0
        for batch in loader:
            x_in = batch["input"].to(device)
            y_tgt = batch["target"].to(device)
            optimizer.zero_grad()
            y_pred = fno_model(x_in)
            loss, loss_dict = loss_fn(y_pred, y_tgt)
            loss.backward()
            optimizer.step()
            total_ep_loss += loss_dict["loss_total"]

        avg_loss = total_ep_loss / len(loader)
        print(f"  SFNO 2D Époque {ep:02d}/{num_fno_epochs:02d} | Perte Spectrale FFT2D: {avg_loss:.6f}")
        if avg_loss < best_fno_loss:
            best_fno_loss = avg_loss
            torch.save({
                "model_state_dict": fno_model.state_dict(),
                "loss": avg_loss,
                "epoch": ep
            }, "checkpoints/stochastic_fno_best.pt")

    print(f"[OK] Checkpoint SFNO 2D sauvegardé : checkpoints/stochastic_fno_best.pt (Perte: {best_fno_loss:.6f})")

    # 4. Entraînement du Stochastic Fourier Neural Operator 3D (SFNO 3D Volumique)
    print("-" * 80)
    print("[*] ENTRAÎNEMENT DU STOCHASTIC FOURIER NEURAL OPERATOR 3D (SFNO 3D VOLUMIQUE)...")
    fno_3d_model = StochasticFourierNeuralOperator3D(
        in_channels=7,
        out_channels=7,
        width=24,
        modes1=4,
        modes2=6,
        modes3=6,
        latent_dim=16,
        num_layers=3
    ).to(device)

    opt_3d = optim.AdamW(fno_3d_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_3d_fn = CombinedPhysicsLoss(lambda_l2=1.0, lambda_h1=0.4, lambda_spec=0.1, dim=3).to(device)

    best_3d_loss = float("inf")
    num_3d_epochs = 6
    for ep_3d in range(1, num_3d_epochs + 1):
        fno_3d_model.train()
        ep_3d_loss = 0.0
        num_batches_3d = 4
        for _ in range(num_batches_3d):
            # Tenseurs volumiques 3D physiques issus du solveur Navier-Stokes Boussinesq
            cfd_rollout = generate_cfd_3d_rollout(D=12, H=32, W=32, steps=4, device=device)
            x_3d = cfd_rollout[:-1] # [3, 7, 12, 32, 32]
            y_3d_tgt = cfd_rollout[1:] # [3, 7, 12, 32, 32]
            z_noise = torch.randn(x_3d.shape[0], 16, device=device)

            opt_3d.zero_grad()
            y_3d_pred = fno_3d_model(x_3d, z_noise)
            loss_v, loss_v_dict = loss_3d_fn(y_3d_pred, y_3d_tgt)
            loss_v.backward()
            opt_3d.step()
            ep_3d_loss += loss_v_dict["loss_total"]

        avg_3d = ep_3d_loss / float(num_batches_3d)
        print(f"  SFNO 3D Époque {ep_3d:02d}/{num_3d_epochs:02d} | Perte Sobolev H1 3D: {avg_3d:.6f}")
        if avg_3d < best_3d_loss:
            best_3d_loss = avg_3d
            torch.save({
                "model_state_dict": fno_3d_model.state_dict(),
                "loss": avg_3d,
                "epoch": ep_3d
            }, "checkpoints/stochastic_fno_3d_best.pt")

    print(f"[OK] Checkpoint SFNO 3D Volumique sauvegardé : checkpoints/stochastic_fno_3d_best.pt (Perte: {best_3d_loss:.6f})")

    # 5. Entraînement du Modèle Neural Implicit Field / MLP 3D Continu sur Navier-Stokes
    print("-" * 80)
    print("[*] ENTRAÎNEMENT DU MODÈLE NEURAL IMPLICIT FIELD / MLP 3D THERMODYNAMIQUE...")
    mlp_3d_model = ThermodynamicMLP3D(
        coord_dim=4,
        forcing_dim=3,
        out_dim=7,
        hidden_dim=128,
        num_layers=5,
        num_frequencies=6
    ).to(device)

    opt_mlp = optim.AdamW(mlp_3d_model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_mlp_fn = nn.MSELoss()

    best_mlp_loss = float("inf")
    num_mlp_epochs = 6
    for ep_mlp in range(1, num_mlp_epochs + 1):
        mlp_3d_model.train()
        ep_mlp_loss = 0.0
        num_batches_mlp = 6
        for _ in range(num_batches_mlp):
            cfd_rollout = generate_cfd_3d_rollout(D=12, H=32, W=32, steps=4, device=device)
            S, C_c, D_c, H_c, W_c = cfd_rollout.shape
            n_pts = 256
            s_idx = torch.randint(0, S, (n_pts,), device=device)
            d_idx = torch.randint(0, D_c, (n_pts,), device=device)
            h_idx = torch.randint(0, H_c, (n_pts,), device=device)
            w_idx = torch.randint(0, W_c, (n_pts,), device=device)

            norm_t = (s_idx.float() / max(1, S - 1)) * 2.0 - 1.0
            norm_z = (d_idx.float() / max(1, D_c - 1)) * 2.0 - 1.0
            norm_y = (h_idx.float() / max(1, H_c - 1)) * 2.0 - 1.0
            norm_x = (w_idx.float() / max(1, W_c - 1)) * 2.0 - 1.0
            coords_batch = torch.stack([norm_x, norm_y, norm_z, norm_t], dim=-1)
            target_norm = cfd_rollout[s_idx, :, d_idx, h_idx, w_idx] # [n_pts, 7]
            wind_batch = torch.tensor([1.5 / 25.0, -0.8 / 25.0, 0.05], device=device)

            opt_mlp.zero_grad()
            pred_norm = mlp_3d_model(coords_batch, wind_batch)
            loss_mlp = loss_mlp_fn(pred_norm, target_norm)
            loss_mlp.backward()
            opt_mlp.step()
            ep_mlp_loss += loss_mlp.item()

        avg_mlp = ep_mlp_loss / float(num_batches_mlp)
        print(f"  MLP 3D Époque {ep_mlp:02d}/{num_mlp_epochs:02d} | Perte MSE Normalisée: {avg_mlp:.6f}")
        if avg_mlp < best_mlp_loss:
            best_mlp_loss = avg_mlp
            torch.save({
                "model_state_dict": mlp_3d_model.state_dict(),
                "loss": avg_mlp,
                "epoch": ep_mlp
            }, "checkpoints/mlp_3d_thermo_best.pt")

    print(f"[OK] Checkpoint MLP 3D Thermodynamique sauvegardé : checkpoints/mlp_3d_thermo_best.pt (Perte: {best_mlp_loss:.6f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
