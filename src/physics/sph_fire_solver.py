"""
src/physics/sph_fire_solver.py
------------------------------
Solveur Fluide 3D sans maillage SPH (Smoothed Particle Hydrodynamics)
pour la simulation haute fidelite du panache thermique, de l'ascendance
pyrocumulonimbus et du transport lagrangien des gaz chauds et suies.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch


class SPH3DFireSolver:
    """
    Solveur SPH Lagrangien 3D thermo-convectif.
    Chaque particule porte position (x, y, z), vitesse (u, v, w),
    temperature T, masse m, densite rho, pression P et suie.
    """

    def __init__(
        self,
        max_particles: int = 1500,
        smoothing_length_h: float = 25.0,
        ambient_temp_k: float = 298.15,
        rest_density: float = 1.20,      # kg/m3 (Air chaud)
        speed_of_sound: float = 40.0,     # m/s pour stabilite CFL numerique
        gravity: float = 9.81,
        thermal_expansion_beta: float = 0.0034  # 1/T_ambient
    ):
        self.max_particles = max_particles
        self.h = smoothing_length_h
        self.h2 = self.h ** 2
        self.ambient_temp = ambient_temp_k
        self.rho0 = rest_density
        self.cs = speed_of_sound
        self.g = gravity
        self.beta = thermal_expansion_beta

        # Constante de normalisation du noyau spline cubique en 3D
        self.kernel_norm = 1.0 / (math.pi * (self.h ** 3))
        self.grad_kernel_norm = 1.0 / (math.pi * (self.h ** 4))

        # Buffers particulaires
        self.num_active = 0
        self.pos = np.zeros((max_particles, 3), dtype=np.float32)      # [X, Y (Alti), Z]
        self.vel = np.zeros((max_particles, 3), dtype=np.float32)      # [U, V (W_up), W]
        self.temp = np.full(max_particles, ambient_temp_k, dtype=np.float32)
        self.density = np.full(max_particles, rest_density, dtype=np.float32)
        self.pressure = np.zeros(max_particles, dtype=np.float32)
        self.mass = np.full(max_particles, 15.0, dtype=np.float32)     # kg par particule fluide
        self.soot = np.zeros(max_particles, dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)

    def spawn_particles_from_flame_front(
        self,
        flame_sources: List[Tuple[float, float, float, float]],
        count_per_step: int = 40
    ):
        """
        flame_sources : Liste de (wx, wy_ground, wz, heat_intensity)
        """
        if not flame_sources:
            return

        for _ in range(count_per_step):
            if self.num_active >= self.max_particles:
                # Recyclage de la plus vieille particule
                idx = int(np.argmax(self.age[:self.num_active]))
            else:
                idx = self.num_active
                self.num_active += 1

            src = flame_sources[np.random.randint(len(flame_sources))]
            wx, wy_ground, wz, heat = src

            self.pos[idx] = [
                wx + np.random.uniform(-10.0, 10.0),
                wy_ground + np.random.uniform(2.0, 8.0),
                wz + np.random.uniform(-10.0, 10.0)
            ]
            self.vel[idx] = [
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(4.0, 12.0) * (heat / 2000.0),
                np.random.uniform(-0.5, 0.5)
            ]
            self.temp[idx] = min(1800.0, self.ambient_temp + heat * 0.45)
            self.density[idx] = self.rho0 * (self.ambient_temp / self.temp[idx])
            self.soot[idx] = min(1.0, heat / 3000.0)
            self.age[idx] = 0.0

    def step(self, dt: float = 0.15, wind_vec_3d: Tuple[float, float, float] = (10.0, 0.0, 5.0)):
        """
        Integration temporelle SPH Euler-Verlet :
        1. Densite & Pression SPH (Tait)
        2. Gradient de Pression + Flottabilite Thermique Boussinesq + Viscosite Monaghan
        3. Diffusion de Temperature et Refroidissement Radiatif
        4. Mise a jour Positions et Vitesse
        """
        N = self.num_active
        if N == 0:
            return

        pos = self.pos[:N]
        vel = self.vel[:N]
        temp = self.temp[:N]
        u_w, v_w, w_w = wind_vec_3d

        # 1. Calcul de la densite par lissage SPH
        # Matrice des distances r_ij
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # [N, N, 3]
        dist2 = np.sum(diff ** 2, axis=-1)                     # [N, N]
        dist = np.sqrt(dist2 + 1e-6)

        q = dist / self.h
        # Noyau Spline Cubique W(q)
        w_ij = np.where(
            q <= 1.0,
            1.0 - 1.5 * (q ** 2) + 0.75 * (q ** 3),
            np.where(q <= 2.0, 0.25 * ((2.0 - q) ** 3), 0.0)
        ) * self.kernel_norm

        self.density[:N] = np.sum(self.mass[:N, np.newaxis] * w_ij, axis=1)
        self.density[:N] = np.clip(self.density[:N], 0.20, 3.0)

        # 2. Pression d'etat Tait
        self.pressure[:N] = (self.cs ** 2) * (self.density[:N] - self.rho0)

        # 3. Gradient de pression et Viscosite artificielle
        # dW/dr
        dw_dr = np.where(
            q <= 1.0,
            -3.0 * q + 2.25 * (q ** 2),
            np.where(q <= 2.0, -0.75 * ((2.0 - q) ** 2), 0.0)
        ) * (self.grad_kernel_norm / (dist + 1e-6))

        grad_w = diff * dw_dr[:, :, np.newaxis]  # [N, N, 3]

        p_rho2 = self.pressure[:N] / (self.density[:N] ** 2)
        p_term = p_rho2[:, np.newaxis] + p_rho2[np.newaxis, :]  # [N, N]

        f_press = -np.sum(self.mass[:N, np.newaxis, np.newaxis] * p_term[:, :, np.newaxis] * grad_w, axis=1)

        # Flottabilite thermique Boussinesq F_buoy = rho * beta * (T - T_inf) * g
        f_buoy_y = self.beta * (temp - self.ambient_temp) * self.g

        # Forcage du vent
        f_wind_x = 0.25 * (u_w - vel[:, 0])
        f_wind_z = 0.25 * (w_w - vel[:, 2])

        # Acceleration totale
        acc_x = f_press[:, 0] + f_wind_x
        acc_y = f_press[:, 1] + f_buoy_y
        acc_z = f_press[:, 2] + f_wind_z

        # Integration de la vitesse
        self.vel[:N, 0] += acc_x * dt
        self.vel[:N, 1] += acc_y * dt
        self.vel[:N, 2] += acc_z * dt

        # 4. Diffusion thermique & Refroidissement
        # dT/dt = -alpha_rad * (T - T_inf)
        self.temp[:N] -= 0.08 * (self.temp[:N] - self.ambient_temp) * dt
        self.soot[:N] *= (1.0 - 0.02 * dt)

        # 5. Deplacement des particules
        self.pos[:N] += self.vel[:N] * dt
        self.age[:N] += dt

        # Extinction et masquage strict des particules sortant de la parcelle MNT ([-domain/2, +domain/2])
        half_dom = 3200.0
        out_of_bounds = (
            (np.abs(self.pos[:N, 0]) >= half_dom - 15.0) |
            (np.abs(self.pos[:N, 2]) >= half_dom - 15.0) |
            (self.pos[:N, 1] > 2500.0) |
            (self.age[:N] > 180.0)
        )
        self.pos[:N, 1][out_of_bounds] = -1000.0
        self.temp[:N][out_of_bounds] = self.ambient_temp
        self.soot[:N][out_of_bounds] = 0.0

    def get_renderable_sph_state(self) -> Dict[str, Any]:
        """
        Exporte l'etat des particules SPH pour le moteur 3D Three.js.
        """
        N = self.num_active
        if N == 0:
            return {"count": 0, "positions": [], "temperatures": [], "soot": [], "densities": []}

        valid_mask = (self.pos[:N, 1] >= -50.0) & (self.soot[:N] > 0.01)
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) == 0:
            return {"count": 0, "positions": [], "temperatures": [], "soot": [], "densities": []}

        return {
            "count": len(valid_indices),
            "positions": self.pos[valid_indices].round(2).tolist(),
            "temperatures": self.temp[valid_indices].round(1).tolist(),
            "soot": self.soot[valid_indices].round(3).tolist(),
            "densities": self.density[valid_indices].round(3).tolist()
        }
