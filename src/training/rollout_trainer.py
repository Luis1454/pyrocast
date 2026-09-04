"""
src/training/rollout_trainer.py
-------------------------------
Boucle d'entraînement autorégressive sur K pas de temps (Rollout).
Injecte du bruit stochastique en cours de propagation pour stabiliser l'apprentissage.
"""

from typing import Dict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..models.mgnn_fmm import NeuralFMM_MGNN
from ..mesh.octree_graph import AdaptiveOctreeGraphBuilder
from ..losses.physics_loss import CombinedPhysicsLoss


class RolloutTrainer:
    def __init__(
        self,
        model: NeuralFMM_MGNN,
        graph_builder: AdaptiveOctreeGraphBuilder,
        criterion: CombinedPhysicsLoss,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        noise_std: float = 0.01,
        rollout_horizon: int = 4,
    ):
        self.model = model
        self.graph_builder = graph_builder
        self.criterion = criterion
        self.noise_std = noise_std
        self.k_rollout = rollout_horizon

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)

    def train_step(self, trajectory_dense: torch.Tensor, wind_series: torch.Tensor) -> Dict[str, float]:
        """
        trajectory_dense: [T, C, (Z), Y, X] (Trajectoire temporelle dense)
        wind_series: [T, 2] (Vecteur vent temporel)
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        metrics_accum = {"loss_total": 0.0, "loss_l2": 0.0, "loss_h1": 0.0, "loss_spectral": 0.0}

        current_dense = trajectory_dense[0]

        for t in range(self.k_rollout):
            target_dense = trajectory_dense[t + 1]
            wind_t = wind_series[t + 1]

            if self.noise_std > 0 and t > 0:
                current_dense = current_dense + self.noise_std * torch.randn_like(current_dense)

            mesh = self.graph_builder.build_mesh(current_dense)
            pred_nodes = self.model(mesh, wind_t)

            # Échantillonnage de la cible aux coordonnées des nœuds du graphe
            coords_sample = mesh.node_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N, 3]
            target_5d = target_dense.unsqueeze(0) if target_dense.dim() == 4 else target_dense.unsqueeze(0).unsqueeze(2)
            target_nodes = F.grid_sample(target_5d, coords_sample, align_corners=True).squeeze(0).squeeze(1).squeeze(1).permute(1, 0)

            step_loss = F.mse_loss(pred_nodes, target_nodes)
            total_loss += step_loss

            metrics_accum["loss_total"] += step_loss.item()
            metrics_accum["loss_l2"] += step_loss.item()

            current_dense = target_dense.clone().detach()

        total_loss = total_loss / self.k_rollout
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return {k: v / self.k_rollout for k, v in metrics_accum.items()}
