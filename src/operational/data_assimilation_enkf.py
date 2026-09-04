"""
src/operational/data_assimilation_enkf.py
-----------------------------------------
Moteur d'Assimilation Continue de Donnees Satellitaires (Data Assimilation) :
- Filtre de Kalman d'Ensemble (Ensemble Kalman Filter - EnKF)
- Ingestion des flux de detection thermique NASA FIRMS (VIIRS 375m / MODIS) et Sentinel-2
- Recalage Bayesien temps reel de la distribution d'incertitude du front de feu FNO
"""

import math
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import json

from src.terrain.dem_loader import DigitalElevationModel


class EnsembleKalmanFilterWildfireAssimilation:
    """
    Filtre de Kalman d'Ensemble (EnKF) stochastique pour l'assimilation
    des observations satellitaires (VIIRS, Sentinel-2) et des releves drones/SDIS.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (64, 64),
        dx_meters: float = 30.0,
        num_ensembles: int = 30,
        observation_noise_std: float = 0.15
    ):
        self.H, self.W = grid_shape
        self.grid_size = self.H * self.W
        self.dx = dx_meters
        self.Ne = num_ensembles
        self.obs_noise = observation_noise_std

    def assimilate_satellite_observations(
        self,
        forecast_ensemble: np.ndarray,      # [Ne, H, W]
        satellite_hotspots: List[Tuple[float, float, float]],  # (row, col, confidence_0_to_1)
        dem: Optional[DigitalElevationModel] = None
    ) -> Dict[str, Any]:
        """
        Recale la distribution stochastique des membres FNO sur les observations réelles.
        """
        Ne = forecast_ensemble.shape[0]
        # Aplatissement de l'ensemble d'états [Ne, N_state]
        X_f = forecast_ensemble.reshape(Ne, self.grid_size).astype(np.float64)
        x_mean_f = np.mean(X_f, axis=0)
        X_prime = (X_f - x_mean_f) / math.sqrt(Ne - 1)  # [Ne, N_state]

        if not satellite_hotspots:
            # Aucune nouvelle observation, état inchangé
            return {
                "assimilated_ensemble": forecast_ensemble,
                "mean_prob_map": np.mean(forecast_ensemble, axis=0),
                "hotspots_assimilated": 0,
                "uncertainty_reduction_pct": 0.0
            }

        M = len(satellite_hotspots)
        # Vecteur d'observation y [M]
        y_obs = np.array([pt[2] for pt in satellite_hotspots], dtype=np.float64)
        obs_indices = [int(round(pt[0])) * self.W + int(round(pt[1])) for pt in satellite_hotspots]
        obs_indices = [max(0, min(self.grid_size - 1, idx)) for idx in obs_indices]

        # Operateur d'observation H: projection de l'etat sur les pixels observés
        # Y_f [Ne, M]
        Y_f = X_f[:, obs_indices]
        y_mean_f = np.mean(Y_f, axis=0)
        Y_prime = (Y_f - y_mean_f) / math.sqrt(Ne - 1)  # [Ne, M]

        # Matrice de covariance d'erreur de mesure R [M, M]
        R = np.eye(M, dtype=np.float64) * (self.obs_noise ** 2)

        # Matrice d'innovation S = Y' * Y'^T + R [M, M]
        S = np.dot(Y_prime.T, Y_prime) + R
        try:
            S_inv = np.linalg.pinv(S)
        except Exception:
            S_inv = np.linalg.inv(S + np.eye(M) * 1e-4)

        # Gain de Kalman K = X' * Y'^T * S^-1 [N_state, M]
        K = np.dot(np.dot(X_prime.T, Y_prime), S_inv)  # [N_state, M]

        # Perturbation stochastique des observations y_j = y + epsilon_j
        eps = np.random.normal(0, self.obs_noise, (Ne, M))
        y_perturbed = y_obs + eps

        # Mise a jour d'analyse X_a = X_f + (Y_perturbed - Y_f) * K^T
        innovation = y_perturbed - Y_f  # [Ne, M]
        X_a = X_f + np.dot(innovation, K.T)

        # Clamping des probabilités dans [0.0, 1.0]
        X_a = np.clip(X_a, 0.0, 1.0)
        assimilated_ensemble = X_a.reshape(Ne, self.H, self.W).astype(np.float32)

        # Calcul de la réduction de variance (gain d'information)
        var_f = np.mean(np.var(X_f, axis=0))
        var_a = np.mean(np.var(X_a, axis=0))
        unc_reduction = float(max(0.0, (var_f - var_a) / (var_f + 1e-6) * 100.0))

        mean_prob_map = np.mean(assimilated_ensemble, axis=0)

        return {
            "assimilated_ensemble": assimilated_ensemble,
            "mean_prob_map": mean_prob_map,
            "hotspots_assimilated": M,
            "uncertainty_reduction_pct": round(unc_reduction, 2),
            "assimilated_points_wgs84": self._format_hotspots_geojson(satellite_hotspots, dem)
        }

    def _format_hotspots_geojson(
        self,
        hotspots: List[Tuple[float, float, float]],
        dem: Optional[DigitalElevationModel]
    ) -> List[Dict[str, Any]]:
        features = []
        for r, c, conf in hotspots:
            if dem:
                lat, lon = dem.pixel_to_latlon(c, r)
            else:
                lat, lon = 43.5250 + (0.5 - r/64)*0.1, 5.4420 + (c/64 - 0.5)*0.1
            features.append({
                "type": "Feature",
                "properties": {
                    "sensor": "NASA_FIRMS_VIIRS_375M",
                    "confidence": round(conf * 100, 1),
                    "brightness_temp_k": 365.0
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 5), round(lat, 5)]
                }
            })
        return features
