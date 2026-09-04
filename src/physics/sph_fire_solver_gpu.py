"""
src/physics/sph_fire_solver_gpu.py
----------------------------------
Solveur Fluide 3D sans maillage SPH (Smoothed Particle Hydrodynamics)
vectorise et accelere sous PyTorch (CPU / CUDA GPU).
Permet la simulation temps reel de milliers de particules thermiques,
de l'ascendance pyrocumulonimbus et du transport des gaz chauds et suies.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn


class PyTorchSPH3DFireSolver:
    """
    Solveur SPH Lagrangien 3D haute performance entièrement vectorisé sous PyTorch.
    Supporte l'exécution temps réel CPU/GPU pour 2 000 à 10 000+ particules.
    """

    def __init__(
        self,
        max_particles: int = 2500,
        smoothing_length_h: float = 25.0,
        ambient_temp_k: float = 298.15,
        rest_density: float = 1.20,
        speed_of_sound: float = 45.0,
        gravity: float = 9.81,
        thermal_expansion_beta: float = 0.0034,
        device: Optional[torch.device] = None
    ):
        self.max_particles = max_particles
        self.h = smoothing_length_h
        self.h2 = self.h ** 2
        self.ambient_temp = ambient_temp_k
        self.rho0 = rest_density
        self.cs = speed_of_sound
        self.g = gravity
        self.beta = thermal_expansion_beta

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Constantes de normalisation du noyau spline cubique 3D
        self.kernel_norm = 1.0 / (math.pi * (self.h ** 3))
        self.grad_kernel_norm = 1.0 / (math.pi * (self.h ** 4))

        # Buffers PyTorch
        self.num_active = 0
        self.pos = torch.zeros((max_particles, 3), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((max_particles, 3), dtype=torch.float32, device=self.device)
        self.temp = torch.full((max_particles,), ambient_temp_k, dtype=torch.float32, device=self.device)
        self.density = torch.full((max_particles,), rest_density, dtype=torch.float32, device=self.device)
        self.pressure = torch.zeros((max_particles,), dtype=torch.float32, device=self.device)
        self.mass = torch.full((max_particles,), 12.0, dtype=torch.float32, device=self.device)
        self.soot = torch.zeros((max_particles,), dtype=torch.float32, device=self.device)
        self.age = torch.zeros((max_particles,), dtype=torch.float32, device=self.device)

    def spawn_particles_from_flame_front(
        self,
        flame_sources: List[Tuple[float, float, float, float]],
        count_per_step: int = 80
    ):
        """
        flame_sources: Liste de (wx, wy_ground, wz, heat_kw)
        """
        if not flame_sources:
            return

        sources_arr = np.array(flame_sources, dtype=np.float32)
        n_src = len(sources_arr)

        for _ in range(count_per_step):
            if self.num_active >= self.max_particles:
                idx = int(torch.argmax(self.age[:self.num_active]).item())
            else:
                idx = self.num_active
                self.num_active += 1

            src = sources_arr[np.random.randint(n_src)]
            wx, wy, wz, heat = float(src[0]), float(src[1]), float(src[2]), float(src[3])

            self.pos[idx, 0] = float(wx + np.random.uniform(-8.0, 8.0))
            self.pos[idx, 1] = float(wy + np.random.uniform(2.0, 6.0))
            self.pos[idx, 2] = float(wz + np.random.uniform(-8.0, 8.0))

            updraft_init = float(min(25.0, (heat / 800.0) * np.random.uniform(4.0, 10.0)))
            self.vel[idx, 0] = float(np.random.uniform(-0.5, 0.5))
            self.vel[idx, 1] = updraft_init
            self.vel[idx, 2] = float(np.random.uniform(-0.5, 0.5))

            t_val = float(min(1750.0, self.ambient_temp + heat * 0.40 * np.random.uniform(0.7, 1.3)))
            self.temp[idx] = t_val
            self.density[idx] = float(self.rho0 * (self.ambient_temp / t_val))
            self.soot[idx] = float(min(1.0, heat / 2500.0))
            self.age[idx] = 0.0

    @torch.no_grad()
    def step(self, dt: float = 0.12, wind_vec_3d: Tuple[float, float, float] = (12.0, 0.0, 6.0)):
        N = self.num_active
        if N == 0:
            return

        pos = self.pos[:N]
        vel = self.vel[:N]
        temp = self.temp[:N]
        mass = self.mass[:N]

        # 1. Matrice des distances par paires r_ij
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
        dist2 = torch.sum(diff ** 2, dim=-1)         # [N, N]
        dist = torch.sqrt(dist2 + 1e-6)

        q = dist / self.h
        # Noyau Spline Cubique W(q)
        w_ij = torch.where(
            q <= 1.0,
            1.0 - 1.5 * (q ** 2) + 0.75 * (q ** 3),
            torch.where(q <= 2.0, 0.25 * ((2.0 - q) ** 3), torch.zeros_like(q))
        ) * self.kernel_norm

        self.density[:N] = torch.sum(mass.unsqueeze(1) * w_ij, dim=1)
        self.density[:N] = torch.clamp(self.density[:N], 0.25, 2.5)

        # 2. Pression d'etat Tait
        self.pressure[:N] = (self.cs ** 2) * (self.density[:N] - self.rho0)

        # 3. Gradient de pression et Viscosite artificielle
        dw_dr = torch.where(
            q <= 1.0,
            -3.0 * q + 2.25 * (q ** 2),
            torch.where(q <= 2.0, -0.75 * ((2.0 - q) ** 2), torch.zeros_like(q))
        ) * (self.grad_kernel_norm / (dist + 1e-6))

        grad_w = diff * dw_dr.unsqueeze(-1)  # [N, N, 3]

        p_rho2 = self.pressure[:N] / (self.density[:N] ** 2)
        p_term = p_rho2.unsqueeze(1) + p_rho2.unsqueeze(0)  # [N, N]

        f_press = -torch.sum(mass.unsqueeze(1).unsqueeze(2) * p_term.unsqueeze(-1) * grad_w, dim=1)

        # Flottabilite thermique Boussinesq
        f_buoy_y = self.beta * (temp - self.ambient_temp) * self.g

        # Forcage du vent atmospherique
        u_w, v_w, w_w = wind_vec_3d
        f_wind_x = 0.30 * (u_w - vel[:, 0])
        f_wind_z = 0.30 * (w_w - vel[:, 2])

        # Accelerations
        self.vel[:N, 0] += (f_press[:, 0] + f_wind_x) * dt
        self.vel[:N, 1] += (f_press[:, 1] + f_buoy_y) * dt
        self.vel[:N, 2] += (f_press[:, 2] + f_wind_z) * dt

        # Refroidissement radiatif & diffusion thermique
        thermal_diffusion = 0.002 * torch.sum(mass.unsqueeze(1) * w_ij * (temp.unsqueeze(1) - temp.unsqueeze(0)), dim=1)
        self.temp[:N] += thermal_diffusion * dt
        self.temp[:N] -= 0.25 * (self.temp[:N] - self.ambient_temp) * dt
        self.soot[:N] *= (1.0 - 0.08 * dt)

        # Deplacement
        self.pos[:N] += self.vel[:N] * dt
        self.age[:N] += dt

        # Extinction et masquage strict des particules sortant de la parcelle MNT ([-domain/2, +domain/2])
        half_dom = 3200.0
        out_of_bounds = (
            (torch.abs(self.pos[:N, 0]) >= half_dom - 15.0) |
            (torch.abs(self.pos[:N, 2]) >= half_dom - 15.0) |
            (self.pos[:N, 1] > 2500.0) |
            (self.age[:N] > 180.0)
        )
        self.pos[:N, 1][out_of_bounds] = -1000.0
        self.temp[:N][out_of_bounds] = self.ambient_temp
        self.soot[:N][out_of_bounds] = 0.0

    def get_renderable_sph_state(self) -> Dict[str, Any]:
        N = self.num_active
        if N == 0:
            return {"count": 0, "positions": [], "temperatures": [], "soot": [], "densities": []}

        # Filtrage des particules actives et dans les limites
        valid_mask = (self.pos[:N, 1] >= -50.0) & (self.soot[:N] > 0.01)
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)
        
        if valid_indices.numel() == 0:
            return {"count": 0, "positions": [], "temperatures": [], "soot": [], "densities": []}

        pos_np = self.pos[valid_indices].cpu().numpy().round(2).tolist()
        temp_np = self.temp[valid_indices].cpu().numpy().round(1).tolist()
        soot_np = self.soot[valid_indices].cpu().numpy().round(3).tolist()
        dens_np = self.density[valid_indices].cpu().numpy().round(3).tolist()

        return {
            "count": len(pos_np),
            "positions": pos_np,
            "temperatures": temp_np,
            "soot": soot_np,
            "densities": dens_np
        }
