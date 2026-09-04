"""
src/terrain/fuel_models.py
--------------------------
Modèles de combustibles standardisés (NFFL / Scott & Burgan 40) :
- Charge en combustible sec w_0 (kg/m^2)
- Rapport surface/volume sigma (1/m)
- Humidité d'extinction FMC_ext (fraction)
- Pouvoir calorifique inférieur H_c (kJ/kg)
"""

from typing import NamedTuple, Dict
import numpy as np
import torch


class StandardFuelClass(NamedTuple):
    name: str
    fuel_load_kg_m2: float       # Charge massique sèche w_0
    surface_to_volume_ratio: float # Sigma (1/m)
    extinction_moisture: float    # FMC_ext (0 à 1)
    heat_content_kj_kg: float     # Chaleur de combustion (approx 18600 kJ/kg)
    mean_bed_depth_m: float       # Épaisseur du lit de combustible


class FuelModelDatabase:
    """
    Base de données thermo-physique des types de végétation.
    """

    MODELS: Dict[int, StandardFuelClass] = {
        # 1. Herbacées & Friches sèches (Propagation très rapide)
        1: StandardFuelClass(
            name="Herbacées Sèches (Short Grass)",
            fuel_load_kg_m2=0.166,
            surface_to_volume_ratio=11480.0,
            extinction_moisture=0.12,
            heat_content_kj_kg=18600.0,
            mean_bed_depth_m=0.30,
        ),
        # 4. Maquis Méditerranéen dense / Garrigue / Chapparal (Très énergétique)
        4: StandardFuelClass(
            name="Garrigue / Maquis Dense (Chaparral)",
            fuel_load_kg_m2=1.120,
            surface_to_volume_ratio=6560.0,
            extinction_moisture=0.20,
            heat_content_kj_kg=19000.0,
            mean_bed_depth_m=1.80,
        ),
        # 8. Forêt de feuillus / Litière de feuilles mortes (Propagation lente)
        8: StandardFuelClass(
            name="Litière de Feuillus (Hardwood Litter)",
            fuel_load_kg_m2=0.336,
            surface_to_volume_ratio=6560.0,
            extinction_moisture=0.30,
            heat_content_kj_kg=18600.0,
            mean_bed_depth_m=0.06,
        ),
        # 10. Forêt de Résineux avec sous-bois dense (Feu de cime potentiel)
        10: StandardFuelClass(
            name="Forêt de Pins avec Sous-bois (Timber / Understory)",
            fuel_load_kg_m2=0.672,
            surface_to_volume_ratio=6560.0,
            extinction_moisture=0.25,
            heat_content_kj_kg=18600.0,
            mean_bed_depth_m=0.30,
        ),
        # 99. Incombustible (Eau, routes, rocher nu, zone urbaine)
        99: StandardFuelClass(
            name="Incombustible (Eau / Roches / Urbain)",
            fuel_load_kg_m2=0.0,
            surface_to_volume_ratio=1.0,
            extinction_moisture=0.01,
            heat_content_kj_kg=0.0,
            mean_bed_depth_m=0.0,
        ),
    }

    @classmethod
    def get(cls, fuel_id: int) -> StandardFuelClass:
        return cls.MODELS.get(fuel_id, cls.MODELS[4])
