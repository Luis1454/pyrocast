"""
src/operational/tactical_sim.py
-------------------------------
Gestionnaire des interventions tactiques feux de forêts :
- Coupe-feux mécanisés (largeur, tracé GPS)
- Largages de produit retardant (Canadair CL-415 / Dash Q400)
- Barrières naturelles (lacs, cours d'eau, autoroutes)
"""

from typing import List, Tuple
import numpy as np
import torch


class TacticalInterventionManager:
    """
    Applique les actions tactiques des pompiers sur la grille de combustible et d'énergie.
    """

    def __init__(self, grid_shape: Tuple[int, int], dx_meters: float = 30.0):
        self.H, self.W = grid_shape
        self.dx = dx_meters
        # Masque d'incombustibilité (0 = normal, 1 = incombustible / retardant)
        self.incombustible_mask = np.zeros(grid_shape, dtype=np.float32)
        self.fuel_reduction_factor = np.ones(grid_shape, dtype=np.float32)

    def add_firebreak_line(self, p1_px: Tuple[int, int], p2_px: Tuple[int, int], width_meters: float = 60.0):
        """
        Trace une tranchée coupe-feu entre p1 et p2 (ex: création par bulldozer).
        """
        x1, y1 = p1_px
        x2, y2 = p2_px
        radius_px = int(max(1, width_meters / (2.0 * self.dx)))

        num_points = int(np.hypot(x2 - x1, y2 - y1) * 2)
        for t in np.linspace(0, 1, max(num_points, 2)):
            cx = int(x1 + t * (x2 - x1))
            cy = int(y1 + t * (y2 - y1))
            y_min, y_max = max(0, cy - radius_px), min(self.H, cy + radius_px + 1)
            x_min, x_max = max(0, cx - radius_px), min(self.W, cx + radius_px + 1)
            self.incombustible_mask[y_min:y_max, x_min:x_max] = 1.0
            self.fuel_reduction_factor[y_min:y_max, x_min:x_max] = 0.0

    def add_aerial_retardant_drop(self, center_px: Tuple[int, int], length_m: float = 300.0, width_m: float = 50.0):
        """
        Simule un largage de retardant rouge polyphosphate par bombardier d'eau (Canadair).
        """
        cx, cy = center_px
        rx = int(length_m / (2.0 * self.dx))
        ry = int(width_m / (2.0 * self.dx))

        y_min, y_max = max(0, cy - ry), min(self.H, cy + ry + 1)
        x_min, x_max = max(0, cx - rx), min(self.W, cx + rx + 1)

        # Le retardant triple l'humidité d'extinction et réduit de 90% la combustibilité
        self.fuel_reduction_factor[y_min:y_max, x_min:x_max] *= 0.10

    def apply_to_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applique les barrières tactiques sur le tenseur d'état [C, H, W].
        Canal 5 : Carburant (FMC / Fuel mass)
        Canal 0 : Température
        """
        device = state_tensor.device
        fuel_mod = torch.from_numpy(self.fuel_reduction_factor).to(device)
        incomb_mod = torch.from_numpy(self.incombustible_mask).to(device)

        modified_state = state_tensor.clone()
        if modified_state.shape[0] > 5:
            modified_state[5] = modified_state[5] * fuel_mod

        # Extinction forcée sur la tranchée coupe-feu
        modified_state[0] = modified_state[0] * (1.0 - incomb_mod)
        return modified_state
