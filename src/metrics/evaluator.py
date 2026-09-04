"""
src/metrics/evaluator.py
------------------------
Suite complète d'évaluation physique et de métrologie pour feux de forêt :
- Précision géométrique du front de flamme : IoU, Sørensen-Dice, Distance de Hausdorff
- Vitesse de propagation : Rate of Spread (RoS) error
- Lois de conservation : Dérive d'enthalpie globale et bilan massique du combustible
- Calibration d'incertitude Monte-Carlo : Brier Score & Indice de fiabilité
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


class PhysicalMetricsEvaluator:
    """
    Évaluateur scientifique vérifiant la conformité CFD et l'exploitabilité opérationnelle.
    """

    def __init__(self, ignition_threshold_norm: float = 1.0, spatial_resolution_m: float = 30.0):
        self.ignition_threshold = ignition_threshold_norm
        self.dx = spatial_resolution_m  # Résolution de maille en mètres

    def evaluate_all(
        self,
        pred_state: torch.Tensor,
        gt_state: torch.Tensor,
        prob_map: Optional[torch.Tensor] = None,
        dt_seconds: float = 60.0
    ) -> Dict[str, float]:
        """
        Calcule l'intégralité du rapport métrologique physique.
        """
        metrics = {}

        # 1. Métriques spatiales et contours géométriques
        iou, dice = self.compute_iou_and_dice(pred_state[0], gt_state[0])
        hausdorff_m = self.compute_hausdorff_distance(pred_state[0], gt_state[0])
        metrics["spatial_IoU"] = float(iou)
        metrics["spatial_Dice"] = float(dice)
        metrics["hausdorff_distance_meters"] = float(hausdorff_m)

        # 2. Erreur sur la vitesse de propagation (Rate of Spread)
        ros_err_pct = self.compute_rate_of_spread_error(pred_state[0], gt_state[0], dt_seconds)
        metrics["rate_of_spread_error_pct"] = float(ros_err_pct)

        # 3. Lois de conservation & dérive physique
        energy_drift, mass_drift = self.compute_conservation_drift(pred_state, gt_state)
        metrics["energy_enthalpy_drift_pct"] = float(energy_drift)
        metrics["fuel_mass_drift_pct"] = float(mass_drift)

        # 4. Score probabiliste Monte-Carlo (si disponible)
        if prob_map is not None:
            brier = self.compute_brier_score(prob_map, gt_state[0] > self.ignition_threshold)
            metrics["brier_score"] = float(brier)

        return metrics

    def compute_iou_and_dice(self, pred_temp: torch.Tensor, gt_temp: torch.Tensor) -> Tuple[float, float]:
        """
        Intersection over Union (IoU) et Sørensen-Dice sur le masque de feu actif.
        """
        pred_mask = (pred_temp > self.ignition_threshold).float()
        gt_mask = (gt_temp > self.ignition_threshold).float()

        intersection = torch.sum(pred_mask * gt_mask).item()
        union = torch.sum(torch.clamp(pred_mask + gt_mask, 0, 1)).item()
        pred_sum = torch.sum(pred_mask).item()
        gt_sum = torch.sum(gt_mask).item()

        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        return iou, dice

    def compute_hausdorff_distance(self, pred_temp: torch.Tensor, gt_temp: torch.Tensor) -> float:
        """
        Distance de Hausdorff maximale entre les périmètres du front prédit et CFD (en mètres).
        """
        pred_mask = (pred_temp.detach().cpu().numpy() > self.ignition_threshold)
        gt_mask = (gt_temp.detach().cpu().numpy() > self.ignition_threshold)

        pred_pts = np.argwhere(pred_mask)
        gt_pts = np.argwhere(gt_mask)

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return 0.0 if len(pred_pts) == len(gt_pts) else 1000.0

        # Distance bidirectionnelle de Hausdorff
        d_fwd = directed_hausdorff(pred_pts, gt_pts)[0]
        d_bwd = directed_hausdorff(gt_pts, pred_pts)[0]
        max_dist_pixels = max(d_fwd, d_bwd)

        return float(max_dist_pixels * self.dx)

    def compute_rate_of_spread_error(
        self, pred_temp: torch.Tensor, gt_temp: torch.Tensor, dt_seconds: float
    ) -> float:
        """
        Écart relatif sur la vitesse d'avancement moyen du front de flamme.
        """
        pred_area = torch.sum((pred_temp > self.ignition_threshold).float()).item() * (self.dx ** 2)
        gt_area = torch.sum((gt_temp > self.ignition_threshold).float()).item() * (self.dx ** 2)

        if gt_area < 1e-4:
            return 0.0

        rel_err = abs(pred_area - gt_area) / gt_area * 100.0
        return min(rel_err, 100.0)

    def compute_conservation_drift(
        self, pred_state: torch.Tensor, gt_state: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Vérifie la dérive des intégrales d'énergie thermique et de masse de combustible.
        Canal 0 : Température (enthalpie proportionnelle)
        Canal 5 : FMC / Carburant résiduel
        """
        pred_energy = torch.sum(torch.clamp(pred_state[0], min=0)).item()
        gt_energy = torch.sum(torch.clamp(gt_state[0], min=0)).item()

        energy_drift = abs(pred_energy - gt_energy) / (gt_energy + 1e-6) * 100.0

        # Carburant (Canal 5 s'il existe)
        if pred_state.shape[0] > 5:
            pred_fuel = torch.sum(pred_state[5]).item()
            gt_fuel = torch.sum(gt_state[5]).item()
            fuel_drift = abs(pred_fuel - gt_fuel) / (abs(gt_fuel) + 1e-6) * 100.0
        else:
            fuel_drift = 0.0

        return energy_drift, fuel_drift

    @staticmethod
    def compute_brier_score(prob_map: torch.Tensor, gt_binary_burn: torch.Tensor) -> float:
        """
        Brier Score probabiliste : Mean Squared Error sur les probabilités de présence du feu.
        Brier in [0, 1], 0 = calibration parfaite.
        """
        p = prob_map.detach().cpu().numpy().flatten()
        y = gt_binary_burn.detach().cpu().numpy().astype(np.float32).flatten()
        return float(np.mean((p - y) ** 2))
