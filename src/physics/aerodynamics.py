"""
src/physics/aerodynamics.py
---------------------------
Couplage aéro-thermodynamique et pyro-convection :
- Flottabilité d'Archimède (Approximation de Boussinesq) : w_buoyancy = sqrt(2 * g * (T - T0) / T0 * H_flame)
- Aspiration d'air frais à la base du front (In-draft entrainment)
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledPyroAerodynamics(nn.Module):
    """
    Simule la modification du champ de vent 3D par l'intensité thermique de l'incendie.
    """

    def __init__(self, g: float = 9.81, ambient_temp_k: float = 298.15, flame_height_ref: float = 15.0):
        super().__init__()
        self.g = g
        self.t0 = ambient_temp_k
        self.h_flame = flame_height_ref

    def compute_pyro_wind_modification(
        self,
        temperature_k: torch.Tensor,
        base_wind_u: torch.Tensor,
        base_wind_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule les vitesses 3D (u_tot, v_tot, w_updraft) altérées par la thermique du feu.
        """
        temp_excess = torch.clamp(temperature_k - self.t0, min=0.0)

        # 1. Vitesse verticale du panache ascendant W (Boussinesq)
        # w = sqrt(2 * g * (Delta_T / T0) * H_flame)
        w_updraft = torch.sqrt(2.0 * self.g * (temp_excess / self.t0) * self.h_flame + 1e-6)

        # 2. Vitesse d'aspiration radiale à la base (In-draft)
        # L'air frais est attiré vers le centre thermique proportionnellement à -grad(T)
        temp_2d = temp_excess.unsqueeze(0).unsqueeze(0) if temp_excess.dim() == 2 else temp_excess
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=temperature_k.device, dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=temperature_k.device, dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        grad_tx = F.conv2d(temp_2d, sobel_x, padding=1).squeeze()
        grad_ty = F.conv2d(temp_2d, sobel_y, padding=1).squeeze()

        # In-draft horizontal
        indraft_factor = 0.05
        u_indraft = -grad_tx * indraft_factor
        v_indraft = -grad_ty * indraft_factor

        u_total = base_wind_u + u_indraft
        v_total = base_wind_v + v_indraft

        return u_total, v_total, w_updraft
