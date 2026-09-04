"""
tests/test_pipeline.py
----------------------
Validation unitaire et d'intégration du moteur FireMap :
- Constructeur Octree -> Graphe adaptatif
- Réseau Neural FMM / Multiscale GNN
- CombinedPhysicsLoss (L2 + Sobolev H1 + Spectral FFT)
"""

import torch
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.losses.physics_loss import CombinedPhysicsLoss


def test_complete_firemap_pipeline():
    print("=== [1] Génération d'un front thermique raide WRF-SFIRE synthétique ===")
    B, C, D, H, W = 1, 7, 16, 64, 64
    x = torch.linspace(-2, 2, W)
    y = torch.linspace(-2, 2, H)
    z = torch.linspace(-1, 1, D)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")

    # Front circulaire raide
    r = torch.sqrt(grid_x**2 + grid_y**2)
    flame = torch.exp(-((r - 0.7) ** 2) / 0.03) * 2.5

    gt_state = torch.zeros((C, D, H, W))
    gt_state[0] = flame
    gt_state[2] = 0.5 * torch.sin(grid_y) # Vent U
    gt_state[3] = 0.3 * torch.cos(grid_x) # Vent V

    print(f"Dimension de l'état 3D dense : {list(gt_state.shape)}")

    print("\n=== [2] Test du maillage adaptatif Octree ===")
    builder = AdaptiveOctreeGraphBuilder(
        max_depth=3, min_depth=1, grad_threshold=0.3, temp_ignition_threshold=1.0, r_near=0.15, k_far=4
    )
    mesh = builder.build_mesh(gt_state)

    total_dense_pts = D * H * W
    active_nodes = mesh.node_coords.shape[0]
    compression = (1.0 - (active_nodes / total_dense_pts)) * 100

    print(f"Points de grille cartésienne 3D : {total_dense_pts}")
    print(f"Nœuds actifs dans l'Octree       : {active_nodes}")
    print(f"Taux de compression mémoire      : {compression:.2f}%")
    print(f"Arêtes Near-field (Convection)   : {mesh.edge_index_near.shape[1]}")
    print(f"Arêtes Far-field (Multipôles FMM): {mesh.edge_index_far.shape[1]}")

    print("\n=== [3] Test du modèle Neural FMM / MGNN ===")
    model = NeuralFMM_MGNN(in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3)
    wind_forcing = torch.tensor([1.2, -0.4, 0.15])  # 3D (u, v, w)
    
    out_nodes = model(mesh, wind_forcing)
    print(f"Forme des nœuds de sortie prédits : {list(out_nodes.shape)}")

    print("\n=== [4] Test de la CombinedPhysicsLoss (Sobolev H1 + Spectral FFT) ===")
    criterion = CombinedPhysicsLoss(lambda_l2=1.0, lambda_h1=0.5, lambda_spec=0.2, dim=3)
    
    pred_dense = gt_state.unsqueeze(0).clone()
    pred_dense = pred_dense + 0.05 * torch.randn_like(pred_dense)
    pred_dense.requires_grad_(True)

    loss, metrics = criterion(pred_dense, gt_state.unsqueeze(0))
    loss.backward()

    print(f"Perte totale        : {metrics['loss_total']:.6f}")
    print(f"  - L2 (MSE)        : {metrics['loss_l2']:.6f}")
    print(f"  - Sobolev H1      : {metrics['loss_h1']:.6f} (Anti-dissipation front de flamme)")
    print(f"  - Spectrale FFT   : {metrics['loss_spectral']:.6f} (Conservation des hautes fréquences)")
    print(f"Norme du gradient backprop : {pred_dense.grad.norm().item():.6f}")

    assert not torch.isnan(loss), "Erreur: NaN dans la fonction de perte"
    assert pred_dense.grad is not None, "Erreur: Backpropagation échouée"
    print("\n[OK] TOUS LES TESTS UNITAIRES ET D'INTEGRATION SONT VALIDES AVEC SUCCES.")


if __name__ == "__main__":
    test_complete_firemap_pipeline()
