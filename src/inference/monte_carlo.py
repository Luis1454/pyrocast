"""
src/inference/monte_carlo.py
----------------------------
Moteur d'inférence probabiliste Monte-Carlo.
Exécute des simulations parallèles sur GPU avec perturbations stochastiques météo
pour calculer la carte de probabilité de risque d'incendie P(brûlé, t).
"""

from typing import Tuple
import torch
from ..models.mgnn_fmm import NeuralFMM_MGNN
from ..mesh.octree_graph import AdaptiveOctreeGraphBuilder


class MonteCarloWildfireSimulator:
    def __init__(
        self,
        model: NeuralFMM_MGNN,
        graph_builder: AdaptiveOctreeGraphBuilder,
        num_ensembles: int = 64,
        wind_uncertainty_std: float = 0.15,
    ):
        self.model = model.eval()
        self.graph_builder = graph_builder
        self.num_ensembles = num_ensembles
        self.wind_uncertainty_std = wind_uncertainty_std

    @torch.no_grad()
    def predict_probability_map(
        self,
        initial_state_dense: torch.Tensor,
        base_wind_forecast: torch.Tensor,
        steps: int = 10,
        ignition_threshold_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        initial_state_dense: [C, (Z), Y, X]
        base_wind_forecast: [steps, 2]
        Retourne : [steps, Y, X] représentant la probabilité [0, 1] de propagation.
        """
        device = initial_state_dense.device
        H, W = initial_state_dense.shape[-2], initial_state_dense.shape[-1]

        burn_accum = torch.zeros((steps, H, W), device=device)

        for ens in range(self.num_ensembles):
            current_state = initial_state_dense.clone()

            for t in range(steps):
                wind_perturbed = base_wind_forecast[t] + self.wind_uncertainty_std * torch.randn_like(
                    base_wind_forecast[t]
                )

                mesh = self.graph_builder.build_mesh(current_state)
                next_nodes = self.model(mesh, wind_perturbed)

                temp_field = current_state[0]
                is_burnt = (temp_field > ignition_threshold_norm).float()

                if is_burnt.dim() == 3:
                    is_burnt = is_burnt.max(dim=0).values

                burn_accum[t] += is_burnt

        return burn_accum / float(self.num_ensembles)
