"""
src/physics/combustion.py
-------------------------
Moteur thermochimique de combustion solide et transfert thermique :
- Radiation non-linéaire de Stefan-Boltzmann : q_rad = eps * sigma_sb * (T^4 - T_inf^4)
- Cinétique de pyrolyse et perte de masse selon Arrhenius
- Effet inhibiteur de l'humidité du combustible (FMC / FMC_ext)
"""

from typing import Tuple
import torch
import torch.nn as nn


class ThermochemicalCombustionEngine(nn.Module):
    """
    Solveur thermochimique exact des lois de conservation de masse et d'énergie.
    """

    def __init__(
        self,
        ambient_temp_k: float = 298.15,
        ignition_temp_k: float = 600.0,
        stefan_boltzmann: float = 5.670374e-8,
        emissivity: float = 0.95,
        heat_capacity_air: float = 1005.0,  # J / (kg * K)
        air_density: float = 1.225,          # kg / m^3
    ):
        super().__init__()
        self.t_ambient = ambient_temp_k
        self.t_ignition = ignition_temp_k
        self.sigma_sb = stefan_boltzmann
        self.emissivity = emissivity
        self.cp = heat_capacity_air
        self.rho_air = air_density

    def compute_radiative_flux(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """
        Flux radiatif émis par la flamme en W/m^2 (Loi de Stefan-Boltzmann).
        """
        t_clamped = torch.clamp(temperature_k, min=self.t_ambient)
        return self.emissivity * self.sigma_sb * (t_clamped**4 - self.t_ambient**4)

    def compute_pyrolysis_rate(
        self,
        temperature_k: torch.Tensor,
        fuel_mass_kg_m2: torch.Tensor,
        fuel_moisture: torch.Tensor,
        extinction_moisture: float = 0.25,
        arrhenius_a: float = 1.5e3,
        activation_energy_over_r: float = 4000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule le taux de consommation du combustible dm_f/dt (kg/m^2/s) et le dégagement de chaleur.
        """
        # Facteur d'humidité : 1 si sec, 0 si FMC >= FMC_ext
        moisture_damping = torch.clamp(1.0 - (fuel_moisture / extinction_moisture), min=0.0)

        # Seuil d'activation d'Arrhenius
        active_combustion = (temperature_k >= self.t_ignition).float()
        arrhenius_factor = arrhenius_a * torch.exp(-activation_energy_over_r / torch.clamp(temperature_k, min=300.0))

        # Débit massique consommé dm/dt
        dm_dt = arrhenius_factor * fuel_mass_kg_m2 * moisture_damping * active_combustion

        # Chaleur libérée (W/m^2) avec H_c ~ 18.6 MJ/kg
        h_c = 18.6e6
        heat_release_rate = dm_dt * h_c

        return dm_dt, heat_release_rate
