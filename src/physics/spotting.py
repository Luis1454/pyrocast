"""
src/physics/spotting.py
-----------------------
Modèle physique et stochastique de saut de braises (Spotting / Feux secondaires) :
- Éjection des tisons par le panache ascendant thermique W
- Portance balistique sous le vent U, V
- Probabilité d'allumage secondaire en fonction du FMC cible
"""

from typing import List, Tuple
import numpy as np
import torch


class StochasticEmberSpottingModel:
    """
    Modélise l'apparition discontinue de feux secondaires en avant du front principal.
    """

    def __init__(
        self,
        dx_meters: float = 30.0,
        max_spotting_distance_m: float = 800.0,
        ember_ignition_prob: float = 0.35,
    ):
        self.dx = dx_meters
        self.max_dist = max_spotting_distance_m
        self.base_prob = ember_ignition_prob

    def simulate_spotting_step(
        self,
        temperature_grid: torch.Tensor,
        wind_u: torch.Tensor,
        wind_v: torch.Tensor,
        wind_w: torch.Tensor,
        fuel_moisture: torch.Tensor,
        ignition_temp_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Détecte les zones de flammes intenses et génère des points d'ignition secondaires.
        """
        device = temperature_grid.device
        H, W = temperature_grid.shape[-2], temperature_grid.shape[-1]
        temp_2d = temperature_grid[0] if temperature_grid.dim() == 3 else temperature_grid

        # 1. Masque des zones émettrices de braises (fort panache thermique W et T élevée)
        emitter_mask = (temp_2d > ignition_temp_norm) & (wind_w > 0.3)
        emitter_indices = torch.nonzero(emitter_mask, as_tuple=False)

        new_ignitions_mask = torch.zeros((H, W), device=device, dtype=torch.float32)

        if emitter_indices.shape[0] == 0:
            return new_ignitions_mask

        # Sous-échantillonnage stochastique pour le réalisme
        num_emitters = min(emitter_indices.shape[0], 25)
        perm = torch.randperm(emitter_indices.shape[0])[:num_emitters]
        selected_emitters = emitter_indices[perm]

        for idx in range(num_emitters):
            y_src = int(selected_emitters[idx, 0].item())
            x_src = int(selected_emitters[idx, 1].item())

            # Vitesse et portance locale
            u_loc = float(wind_u[y_src, x_src].item()) if wind_u.dim() == 2 else float(wind_u[0].item())
            v_loc = float(wind_v[y_src, x_src].item()) if wind_v.dim() == 2 else float(wind_v[1].item())
            w_loc = float(wind_w[y_src, x_src].item()) if wind_w.dim() == 2 else float(wind_w[2].item())

            # Distance de vol balistique du tison X = U * sqrt(2 * z_loft / g)
            z_loft = 40.0 * (w_loc ** 1.2)  # Altitude atteinte dans le panache
            flight_time = np.sqrt(2.0 * z_loft / 9.81)
            dist_flight_m = np.random.lognormal(mean=np.log(max(10.0, np.sqrt(u_loc**2 + v_loc**2) * flight_time * 15.0)), sigma=0.4)
            dist_flight_m = min(dist_flight_m, self.max_dist)

            # Direction du vol avec diffusion turbulente angulaire
            angle_wind = np.arctan2(v_loc, u_loc) + np.random.normal(0, 0.2)
            dx_pix = int(dist_flight_m * np.cos(angle_wind) / self.dx)
            dy_pix = int(dist_flight_m * np.sin(angle_wind) / self.dx)

            x_target = x_src + dx_pix
            y_target = y_src + dy_pix

            # Vérification des limites du domaine et allumage
            if 0 <= x_target < W and 0 <= y_target < H:
                fmc_target = float(fuel_moisture[y_target, x_target].item()) if fuel_moisture.dim() == 2 else 0.1
                ignite_prob = self.base_prob * (1.0 - np.clip(fmc_target / 0.25, 0.0, 1.0))
                if np.random.rand() < ignite_prob:
                    new_ignitions_mask[y_target, x_target] = 1.5  # Injection d'énergie d'ignition

        return new_ignitions_mask
