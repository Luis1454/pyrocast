"""
tests/test_ground_truth_validator.py
------------------------------------
Tests unitaires pour le module de validation scientifique et de comparaison
Prediction IA (FNO) vs Realite Terrain Satellite (Sentinel-2 / VIIRS / SDIS).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
from src.terrain.dem_loader import DigitalElevationModel
from src.operational.ground_truth_validator import GroundTruthValidator


def test_ground_truth_metrics():
    print("=" * 80)
    print("[*] TEST DU MODULE DE COMPARAISON : PREDICTION IA VS REALITE TERRAIN")
    print("=" * 80)

    elev_dummy = np.ones((64, 64), dtype=np.float32) * 350.0
    dem = DigitalElevationModel(elev_dummy, resolution_meters=30.0, origin_lat_lon=(43.5250, 5.4420))
    validator = GroundTruthValidator(dem)

    # Prediction synthetique simulant un modele FNO performant
    gt = validator.ground_truth_mask
    pred_prob = np.zeros((64, 64), dtype=np.float32)
    pred_prob[gt == 1] = 0.88
    # Bruit leger sur la frontiere
    pred_prob[30:34, 40:44] = 0.65

    res = validator.evaluate_prediction(pred_prob, threshold=0.50)
    metrics = res["metrics"]

    print(f"Intersection over Union (IoU): {metrics['iou_jaccard']} %")
    print(f"Dice F1-Score                : {metrics['dice_f1']} %")
    print(f"Precision / Rappel           : {metrics['precision']} % / {metrics['recall']} %")
    print(f"Score de Brier (Calibration) : {metrics['brier_score']}")
    print(f"Distance de Hausdorff        : {metrics['hausdorff_dist_m']} m")
    print(f"Surface Predite / Reelle     : {metrics['pred_area_ha']} Ha / {metrics['ground_truth_area_ha']} Ha")

    assert metrics["iou_jaccard"] > 75.0, "L'IoU est anormalement basse."
    assert metrics["dice_f1"] > 85.0, "Le Dice score est anormalement bas."
    assert metrics["brier_score"] < 0.15, "Le score de Brier est trop eleve."

    geojson_path = validator.generate_validation_geojson(pred_prob, threshold=0.50)
    assert geojson_path.exists()
    print(f"[OK] Export GeoJSON de comparaison valide : {geojson_path}")

    print("[OK] Module de validation scientifique valide a 100%.")


if __name__ == "__main__":
    test_ground_truth_metrics()
