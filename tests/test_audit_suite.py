"""
tests/test_audit_suite.py
-------------------------
Suite d'Audit et de Régression Globale pour l'ensemble du projet FireMap :
- Validation de l'entraînement (train.py) & convergence des pertes Sobolev H1 + FFT
- Validation de l'inférence Monte-Carlo (infer.py)
- Validation du solveur 3D Navier-Stokes & export VTK (run_3d_cfd_simulation.py)
- Validation de la mission de terrain & exports GeoJSON (run_field_mission.py)
- Validation E2E Playwright de l'interface Web 3D / Three.js (tests/test_playwright_e2e.py)
"""

import sys
import time
import subprocess
from pathlib import Path


def run_stage(name: str, cmd: list) -> bool:
    print("\n" + "=" * 80)
    print(f" [*] EXECUTION DU STAGE : {name}")
    print(f" [*] Commande : {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"\n[OK] STAGE '{name}' VALIDE EN {elapsed:.2f}s")
        return True
    else:
        print(f"\n[X] ECHEC DU STAGE '{name}' (Code {res.returncode})")
        return False


def main():
    print("=" * 85)
    print("               AUDIT ET TEST GLOBAL DE TOUT LE LOGICIEL FIREMAP")
    print("=" * 85)

    stages = [
        ("1. Entraînement Multi-Époques & Checkpoints (train.py)", [sys.executable, "-u", "train.py"]),
        ("2. Inférence Probabiliste Monte-Carlo (infer.py)", [sys.executable, "-u", "infer.py"]),
        ("3. Solveur 3D Navier-Stokes & Export VTK (run_3d_cfd_simulation.py)", [sys.executable, "-u", "run_3d_cfd_simulation.py"]),
        ("4. Mission Terrain & Exports SIG GeoJSON (run_field_mission.py)", [sys.executable, "-u", "run_field_mission.py"]),
        ("5. Ingénierie Physique Rothermel/Byram & Données Réelles (tests/test_fire_engineering.py)", [sys.executable, "-u", "tests/test_fire_engineering.py"]),
        ("6. Stochastic Fourier Neural Operator 2D & 3D (tests/test_stochastic_fno.py)", [sys.executable, "-u", "tests/test_stochastic_fno.py"]),
        ("7. Modèle Neural Implicit Field / MLP 3D Thermodynamique (tests/test_mlp_3d.py)", [sys.executable, "-u", "tests/test_mlp_3d.py"]),
        ("8. Écoulements Thermodynamiques 3D Matplotlib (src/visualization/thermodynamic_flows_mpl.py)", [sys.executable, "-u", "-m", "src.visualization.thermodynamic_flows_mpl"]),
        ("9. Solveur Fluide SPH 3D PyTorch GPU (tests/test_sph_gpu.py)", [sys.executable, "-u", "tests/test_sph_gpu.py"]),
        ("10. Dataset Historique Multispectral (tests/test_historical_dataset.py)", [sys.executable, "-u", "tests/test_historical_dataset.py"]),
        ("11. Moteur d'Assimilation Satellite EnKF (tests/test_data_assimilation_enkf.py)", [sys.executable, "-u", "tests/test_data_assimilation_enkf.py"]),
        ("12. Validation Scientifique IA vs Réalité Satellite (tests/test_ground_truth_validator.py)", [sys.executable, "-u", "tests/test_ground_truth_validator.py"]),
        ("13. Test E2E Playwright Console C2 & Rendu 3D (tests/test_playwright_e2e.py)", [sys.executable, "-u", "tests/test_playwright_e2e.py"]),
    ]

    all_passed = True
    results = []

    for name, cmd in stages:
        passed = run_stage(name, cmd)
        results.append((name, passed))
        if not passed:
            all_passed = False
            break

    print("\n" + "=" * 85)
    print("                           TABLEAU RÉCAPITULATIF DE L'AUDIT")
    print("=" * 85)
    for name, passed in results:
        status_str = "[OK] SUCCES" if passed else "[X] ECHEC"
        print(f"  {status_str:15s} | {name}")

    if all_passed:
        print("\n[OK] TOUS LES MODULES DU LOGICIEL SONT CORRIGES ET VALIDES A 100%.")
    else:
        print("\n[!] CERTAINS TESTS ONT ECHOUE. CORRECTIONS NECESSAIRES.")
        sys.exit(1)


if __name__ == "__main__":
    main()
