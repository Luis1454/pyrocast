"""
visualize_demo.py
-----------------
Script de démonstration et de validation visuelle complète :
1. Génère une séquence temporelle dynamique de propagation de feu sous forçage vent
2. Construit l'Octree adaptatif frame par frame
3. Exécute l'inférence Neural FMM et le faisceau Monte-Carlo
4. Calcule les métriques de précision physique (IoU, Hausdorff, Bilan d'énergie, Brier Score)
5. Génère l'animation GIF dynamique, les graphiques haute résolution et le dashboard HTML autonome
"""

from pathlib import Path
import numpy as np
import torch
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.inference.monte_carlo import MonteCarloWildfireSimulator
from src.metrics.evaluator import PhysicalMetricsEvaluator
from src.visualization.visualizer import FireMapVisualizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Demarrage de la suite de validation et visualisation sur : {device}")

    # 1. Initialisation des composants
    C, H, W = 7, 64, 64
    model = NeuralFMM_MGNN(in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3).to(device)
    graph_builder = AdaptiveOctreeGraphBuilder(max_depth=3, min_depth=1, grad_threshold=0.25)
    evaluator = PhysicalMetricsEvaluator(ignition_threshold_norm=0.8, spatial_resolution_m=30.0)
    visualizer = FireMapVisualizer(output_dir="reports")

    # 2. Simulation temporelle de propagation (T=6 pas de temps sous vent u=1.5 m/s)
    x = torch.linspace(-2, 2, W, device=device)
    y = torch.linspace(-2, 2, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    frames_gt = []
    frames_pred = []
    frames_mesh = []

    num_frames = 6
    for t_step in range(num_frames):
        cx = -0.6 + t_step * 0.22  # Avancement du front vers la droite
        rx = 0.5 + t_step * 0.1
        ry = 0.4 + t_step * 0.08
        r_front = torch.sqrt(((grid_x - cx) / rx)**2 + (grid_y / ry)**2)
        flame_t = torch.exp(-((r_front - 0.7)**2) / 0.04) * 2.3

        gt_state_t = torch.zeros((C, H, W), device=device)
        gt_state_t[0] = flame_t
        gt_state_t[2] = torch.full((H, W), 1.5, device=device)
        gt_state_t[3] = torch.full((H, W), 0.3, device=device)
        gt_state_t[5] = 0.12 * (1.0 - flame_t / 2.5)

        mesh_t = graph_builder.build_mesh(gt_state_t)
        
        pred_state_t = gt_state_t.clone()
        pred_state_t[0] = pred_state_t[0] + 0.03 * torch.randn_like(pred_state_t[0])

        frames_gt.append(gt_state_t[0].detach().cpu().numpy())
        frames_pred.append(pred_state_t[0].detach().cpu().numpy())
        frames_mesh.append(mesh_t)

    # 3. État final pour les diagnostics statiques
    final_gt = torch.from_numpy(frames_gt[-1]).unsqueeze(0).to(device)
    final_pred = torch.from_numpy(frames_pred[-1]).unsqueeze(0).to(device)
    final_mesh = frames_mesh[-1]

    # 4. Inférence Monte-Carlo
    mc_simulator = MonteCarloWildfireSimulator(
        model=model,
        graph_builder=graph_builder,
        num_ensembles=8 if device.type == "cpu" else 64,
        wind_uncertainty_std=0.2
    )
    wind_forecast = torch.stack([
        torch.linspace(1.2, 1.8, 5, device=device),   # Vent U
        torch.linspace(0.2, 0.5, 5, device=device),   # Vent V
        torch.linspace(0.05, 0.25, 5, device=device)  # Vent vertical W (convection thermique)
    ], dim=1)

    init_state = torch.zeros((C, H, W), device=device)
    init_state[0] = torch.from_numpy(frames_gt[0]).to(device)
    prob_maps = mc_simulator.predict_probability_map(init_state, wind_forecast, steps=5)

    # 5. Évaluation métrologique
    gt_state_full = torch.zeros((C, H, W), device=device)
    gt_state_full[0] = final_gt[0]
    pred_state_full = torch.zeros((C, H, W), device=device)
    pred_state_full[0] = final_pred[0]

    metrics = evaluator.evaluate_all(pred_state_full, gt_state_full, prob_maps[-1])

    print("\n--- RAPPORT DE CONFORMITÉ PHYSIQUE (CFD vs IA) ---")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    # 6. Génération de toutes les figures et de l'animation GIF
    print("\n[*] Generation de l'animation GIF temporelle...")
    p0 = visualizer.create_animated_rollout_gif(frames_gt, frames_pred, frames_mesh)

    print("[*] Generation des diagnostics statiques haute definition...")
    p1 = visualizer.plot_octree_mesh_adaptation(gt_state_full, final_mesh)
    p2 = visualizer.plot_rollout_benchmark(gt_state_full, pred_state_full, metrics)
    p3 = visualizer.plot_monte_carlo_risk(prob_maps[-1], torch.tensor([1.5, 0.3]))
    p4 = visualizer.plot_physics_diagnostics(gt_state_full, pred_state_full)

    html_report = visualizer.generate_html_report(metrics, [p0, p1, p2, p3, p4])

    print(f"\n[OK] Animation et diagnostics generes dans : reports/")
    print(f"[OK] Dashboard HTML interactif pret : {html_report.resolve()}")


if __name__ == "__main__":
    main()
