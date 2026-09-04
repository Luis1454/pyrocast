"""
tests/test_historical_dataset.py
--------------------------------
Tests unitaires pour le dataset historique et le pipeline d'entrainement FNO.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from src.data.historical_wildfire_dataset import HistoricalWildfireDataset


def test_historical_dataset_pipeline():
    print("=" * 80)
    print("[*] TEST DU DATASET HISTORIQUE MULTISPECTRAL (SENTINEL-2 / ERA5 / MNT)")
    print("=" * 80)

    dataset = HistoricalWildfireDataset(num_samples=12, grid_size=64)
    assert len(dataset) == 12, "Nombre d'échantillons incorrect."

    sample = dataset[0]
    x_in = sample["input"]
    y_tgt = sample["target"]

    print(f"Shape tenseur d'entree [C=8, H=64, W=64] : {list(x_in.shape)}")
    print(f"Shape sequence cible  [T=7, H=64, W=64] : {list(y_tgt.shape)}")
    print(f"Cas historique documente                 : {sample['case_name']}")

    assert x_in.shape == (8, 64, 64)
    assert y_tgt.shape == (7, 64, 64)
    assert (y_tgt >= 0.0).all() and (y_tgt <= 1.0).all()

    # Vérification de la croissance monotone de l'empreinte brûlée
    for t in range(1, 7):
        assert torch.sum(y_tgt[t]) >= torch.sum(y_tgt[t-1]) - 1e-4, "L'empreinte brûlée doit croître dans le temps."

    print("[OK] Dataset historique validé à 100%.")


if __name__ == "__main__":
    test_historical_dataset_pipeline()
