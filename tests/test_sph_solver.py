"""
tests/test_sph_solver.py
------------------------
Tests unitaires pour le solveur fluide 3D SPH (Smoothed Particle Hydrodynamics).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
from src.physics.sph_fire_solver import SPH3DFireSolver


def test_sph_3d_simulation():
    print("=" * 80)
    print("[*] TEST DU SOLVEUR FLUIDE SPH 3D (SMOOTHED PARTICLE HYDRODYNAMICS)")
    print("=" * 80)

    sph = SPH3DFireSolver(max_particles=300, smoothing_length_h=20.0)

    # Allumage de 3 sources de flammes au sol
    flames = [
        (-50.0, 30.0, -20.0, 3200.0),
        (0.0, 45.0, 0.0, 4500.0),
        (50.0, 35.0, 20.0, 2900.0)
    ]

    sph.spawn_particles_from_flame_front(flames, count_per_step=60)
    assert sph.num_active == 60, f"Nombre de particules SPH incorrect : {sph.num_active}"

    # Pas d'integration SPH
    for step_i in range(10):
        sph.step(dt=0.10, wind_vec_3d=(15.0, 0.0, 5.0))

    state = sph.get_renderable_sph_state()
    print(f"Particules SPH actives      : {state['count']}")
    print(f"Altitude max atteinte (W_up): {max(p[1] for p in state['positions']):.1f} m")
    print(f"Temperature moyenne panache : {np.mean(state['temperatures']):.1f} K")

    # Verifications physiques
    assert state['count'] == 60
    assert max(p[1] for p in state['positions']) > 45.0, "L'ascendance thermique doit faire monter les particules."
    assert np.mean(state['temperatures']) > 300.0, "La temperature des particules fluides doit etre chaude."

    print("[OK] Solveur SPH 3D valide a 100%.")


if __name__ == "__main__":
    test_sph_3d_simulation()
