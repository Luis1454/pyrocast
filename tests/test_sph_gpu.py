"""
tests/test_sph_gpu.py
---------------------
Tests unitaires pour le solveur fluide 3D SPH vectorisé sous PyTorch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
import torch
from src.physics.sph_fire_solver_gpu import PyTorchSPH3DFireSolver


def test_pytorch_sph_solver():
    print("=" * 80)
    print("[*] TEST DU SOLVEUR PYTORCH SPH 3D VECTORISE (CPU/CUDA)")
    print("=" * 80)

    device = torch.device("cpu")
    solver = PyTorchSPH3DFireSolver(max_particles=500, smoothing_length_h=20.0, device=device)

    flames = [
        (-40.0, 25.0, -15.0, 3000.0),
        (0.0, 35.0, 0.0, 4200.0),
        (35.0, 30.0, 20.0, 3500.0)
    ]

    solver.spawn_particles_from_flame_front(flames, count_per_step=100)
    assert solver.num_active == 100

    for _ in range(8):
        solver.step(dt=0.12, wind_vec_3d=(15.0, 0.0, 6.0))

    state = solver.get_renderable_sph_state()
    print(f"Particules SPH actives : {state['count']}")
    print(f"Altitude max atteinte  : {max(p[1] for p in state['positions']):.1f} m")
    print(f"Temperature moyenne    : {np.mean(state['temperatures']):.1f} K")

    assert state['count'] == 100
    assert max(p[1] for p in state['positions']) > 35.0
    assert np.mean(state['temperatures']) > 310.0

    print("[OK] Solveur PyTorch SPH 3D validé à 100%.")


if __name__ == "__main__":
    test_pytorch_sph_solver()
