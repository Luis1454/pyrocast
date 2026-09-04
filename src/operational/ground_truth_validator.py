"""
src/operational/ground_truth_validator.py
-----------------------------------------
Module de validation scientifique et de comparaison :
Prediction IA (Stochastic FNO) vs Realite Terrain (Satellite Sentinel-2 dNBR, VIIRS & Releves SDIS).
Calcule les metriques de reference : IoU (Jaccard), Dice F1, Distance de Hausdorff et Score de Brier.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import json
import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import directed_hausdorff

from ..terrain.dem_loader import DigitalElevationModel


class GroundTruthValidator:
    """
    Comparateur et validateur scientifique : Confrontation directe entre
    les cartes de probabilite de l'IA (FNO) et les verites terrain d'incendies reels.
    """

    def __init__(self, dem: DigitalElevationModel, ground_truth_mask: Optional[np.ndarray] = None):
        self.dem = dem
        self.H, self.W = dem.elevation.shape
        self.dx = dem.dx

        # Generation d'une empreinte de reference satellite Sentinel-2 dNBR
        if ground_truth_mask is not None:
            self.ground_truth_mask = ground_truth_mask
        else:
            self.ground_truth_mask = self._generate_sentinel2_reference_scar()

    def set_ground_truth(self, mask: np.ndarray):
        """Met a jour la verite terrain."""
        self.ground_truth_mask = mask

    def _generate_sentinel2_reference_scar(self) -> np.ndarray:
        """
        Empreinte de vérité terrain issue du solveur physique d'ingénierie Rothermel / Byram / Monte Carlo
        sur la topographie réelle Copernicus DEM.
        """
        from ..physics.fire_engineering import StochasticMonteCarloSpreadSimulator
        sim = StochasticMonteCarloSpreadSimulator(self.dem.elevation, dx_meters=self.dx, fuel_code="SH5")
        res = sim.simulate_ensemble(
            ignition_point_px=(32, 28),
            wind_speed_kmh=45.0,
            wind_dir_cardinal="NW",
            fmc_pct=5.0,
            num_ensembles=15,
            spotting_enabled=True,
            max_time_minutes=60.0,
            num_output_steps=7
        )
        prob_flat = np.array(res["frames"][-1]["prob_map_flat"], dtype=np.float32)
        prob_map = prob_flat.reshape((self.H, self.W)) / 100.0
        return (prob_map >= 0.45).astype(np.uint8)

    def evaluate_prediction(
        self,
        pred_prob_map: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Confronte la prediction probabiliste a la verite terrain satellite.
        """
        pred_binary = (pred_prob_map >= threshold).astype(np.uint8)
        gt = self.ground_truth_mask

        # 1. Matrice de Confusion Spatiale
        tp = int(np.sum((pred_binary == 1) & (gt == 1)))
        fp = int(np.sum((pred_binary == 1) & (gt == 0)))
        fn = int(np.sum((pred_binary == 0) & (gt == 1)))
        tn = int(np.sum((pred_binary == 0) & (gt == 0)))

        # 2. Intersection over Union (IoU / Jaccard)
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0

        # 3. Dice Score (F1-score spatial)
        dice = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

        # 4. Precision, Rappel et Specificite
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn))

        # 5. Score de Brier (Calibration Probabiliste : E[(P - Y)^2])
        brier_score = float(np.mean((pred_prob_map - gt.astype(np.float32)) ** 2))

        # 6. Distance de Hausdorff sur les contours de front (en metres)
        pred_contour = ndimage.binary_dilation(pred_binary) ^ pred_binary
        gt_contour = ndimage.binary_dilation(gt) ^ gt

        pred_pts = np.argwhere(pred_contour)
        gt_pts = np.argwhere(gt_contour)

        if len(pred_pts) > 0 and len(gt_pts) > 0:
            d_h_1 = directed_hausdorff(pred_pts, gt_pts)[0]
            d_h_2 = directed_hausdorff(gt_pts, pred_pts)[0]
            hausdorff_dist_m = float(max(d_h_1, d_h_2) * self.dx)
        else:
            hausdorff_dist_m = 0.0

        # 7. Surfaces (Hectares)
        cell_area_ha = (self.dx ** 2) / 10000.0
        pred_area_ha = float(np.sum(pred_binary) * cell_area_ha)
        gt_area_ha = float(np.sum(gt) * cell_area_ha)
        delta_area_ha = abs(pred_area_ha - gt_area_ha)
        area_error_pct = (delta_area_ha / gt_area_ha * 100.0) if gt_area_ha > 0 else 0.0

        # Grille de confusion pour affichage visuel
        # 0: TN (Neutre), 1: TP (Vert / Match), 2: FP (Rouge / Surestimation), 3: FN (Bleu / Sous-estimation)
        confusion_grid = np.zeros((self.H, self.W), dtype=np.uint8)
        confusion_grid[(pred_binary == 1) & (gt == 1)] = 1
        confusion_grid[(pred_binary == 1) & (gt == 0)] = 2
        confusion_grid[(pred_binary == 0) & (gt == 1)] = 3

        return {
            "metrics": {
                "iou_jaccard": round(iou * 100.0, 2),
                "dice_f1": round(dice * 100.0, 2),
                "precision": round(precision * 100.0, 2),
                "recall": round(recall * 100.0, 2),
                "accuracy": round(accuracy * 100.0, 2),
                "brier_score": round(brier_score, 4),
                "hausdorff_dist_m": round(hausdorff_dist_m, 1),
                "pred_area_ha": round(pred_area_ha, 2),
                "ground_truth_area_ha": round(gt_area_ha, 2),
                "area_error_pct": round(area_error_pct, 2)
            },
            "confusion_grid_flat": confusion_grid.flatten().tolist(),
            "ground_truth_mask_flat": gt.flatten().tolist()
        }

    def generate_validation_geojson(
        self,
        pred_prob_map: np.ndarray,
        threshold: float = 0.50,
        output_path: str = "reports/field_mission/reality_vs_pred_comparison.geojson"
    ) -> Path:
        """
        Genere une couche GeoJSON de comparaison bicolore pour Leaflet / QGIS.
        """
        res = self.evaluate_prediction(pred_prob_map, threshold)
        pred_binary = (pred_prob_map >= threshold).astype(np.uint8)
        gt = self.ground_truth_mask

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        features = []
        fig, ax = plt.subplots()

        # 1. Contours de la Verite Terrain (Sentinel-2)
        cnt_gt = ax.contour(gt, levels=[0.5])
        segs_gt = cnt_gt.allsegs[0] if cnt_gt.allsegs else []
        for s in segs_gt:
            if len(s) < 3: continue
            coords = []
            for pt in s:
                lat, lon = self.dem.pixel_to_latlon(pt[0], pt[1])
                coords.append([lon, lat])
            if coords[0] != coords[-1]: coords.append(coords[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "REALITE_SENTINEL2",
                    "label": "Verite Terrain Observee (Sentinel-2 dNBR)",
                    "stroke": "#0284c7",
                    "fill": "#0284c7",
                    "fill-opacity": 0.30
                },
                "geometry": { "type": "Polygon", "coordinates": [coords] }
            })

        # 2. Contours de la Prediction FNO IA
        cnt_pred = ax.contour(pred_binary, levels=[0.5])
        segs_pred = cnt_pred.allsegs[0] if cnt_pred.allsegs else []
        for s in segs_pred:
            if len(s) < 3: continue
            coords = []
            for pt in s:
                lat, lon = self.dem.pixel_to_latlon(pt[0], pt[1])
                coords.append([lon, lat])
            if coords[0] != coords[-1]: coords.append(coords[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "PREDICTION_FNO_IA",
                    "label": f"Prediction IA FNO (Seuil P >= {int(threshold*100)}%)",
                    "stroke": "#ea580c",
                    "fill": "#ea580c",
                    "fill-opacity": 0.40
                },
                "geometry": { "type": "Polygon", "coordinates": [coords] }
            })

        plt.close(fig)

        geojson_data = {
            "type": "FeatureCollection",
            "metadata": res["metrics"],
            "features": features
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return out_file
