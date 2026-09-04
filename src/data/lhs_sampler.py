"""
src/data/lhs_sampler.py
-----------------------
Générateur de plan d'expériences par Échantillonnage Hypercube Latin (LHS).
Permet de balayer l'espace des paramètres météo (vent U, V, W), topographiques (pente, Perlin)
et de combustible (Fuel Moisture Content) pour l'entraînement WRF-SFIRE.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import qmc


class LatinHypercubeExperimentSampler:
    """
    Génère un espace d'expérimentations orthogonales bien réparties pour la simulation.
    """

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], seed: int = 42):
        self.bounds = parameter_bounds
        self.param_names = list(parameter_bounds.keys())
        self.dim = len(self.param_names)
        self.sampler = qmc.LatinHypercube(d=self.dim, seed=seed)

    def sample_experiments(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Tire N configurations d'expériences physiques réparties selon l'espace LHS.
        """
        sample_unit = self.sampler.random(n=n_samples)
        
        lower_bounds = [self.bounds[p][0] for p in self.param_names]
        upper_bounds = [self.bounds[p][1] for p in self.param_names]
        
        scaled_samples = qmc.scale(sample_unit, lower_bounds, upper_bounds)
        
        experiments = []
        for row in scaled_samples:
            exp_dict = {name: float(val) for name, val in zip(self.param_names, row)}
            experiments.append(exp_dict)
            
        return experiments
