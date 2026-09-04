"""
tests/test_data_assimilation_enkf.py
------------------------------------
Tests unitaires pour le moteur d'assimilation de données satellitaires (EnKF).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
from src.operational.data_assimilation_enkf import EnsembleKalmanFilterWildfireAssimilation


def test_enkf_satellite_assimilation():
    print("=" * 80)
    print("[*] TEST DU FILTRE DE KALMAN D'ENSEMBLE (EnKF) - ASSIMILATION SATELLITE")
    print("=" * 80)

    enkf = EnsembleKalmanFilterWildfireAssimilation(
        grid_shape=(64, 64),
        dx_meters=30.0,
        num_ensembles=20,
        observation_noise_std=0.10
    )

    # Ensemble de prédictions FNO a priori avec incertitude
    forecast_ens = np.random.uniform(0.15, 0.45, (20, 64, 64)).astype(np.float32)

    # 4 Détections satellites thermiques VIIRS 375m
    satellite_hotspots = [
        (30.0, 28.0, 0.95),
        (32.0, 31.0, 0.92),
        (40.0, 38.0, 0.88),
        (42.0, 41.0, 0.85)
    ]

    res = enkf.assimilate_satellite_observations(forecast_ens, satellite_hotspots)

    print(f"Points satellites assimilés   : {res['hotspots_assimilated']}")
    print(f"Réduction d'incertitude EnKF  : {res['uncertainty_reduction_pct']} %")
    print(f"Probabilité moyenne recalculée: {np.mean(res['mean_prob_map']):.4f}")

    assert res['hotspots_assimilated'] == 4
    assert res['assimilated_ensemble'].shape == (20, 64, 64)
    assert (res['assimilated_ensemble'] >= 0.0).all() and (res['assimilated_ensemble'] <= 1.0).all()
    assert len(res['assimilated_points_wgs84']) == 4

    print("[OK] Moteur d'assimilation EnKF validé à 100%.")


if __name__ == "__main__":
    test_enkf_satellite_assimilation()
