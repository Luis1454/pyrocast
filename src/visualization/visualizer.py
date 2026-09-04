"""
src/visualization/visualizer.py
-------------------------------
Suite de visualisation scientifique et d'aide à la décision opérationnelle :
- Rendu de la maille adaptative Octree vs Champ Thermique
- Comparaison spatiale Rollout (CFD WRF-SFIRE vs Neural FMM) + Cartes d'erreur
- Isochrones et Cartes de Risque Probabilistes Monte-Carlo avec forçage vent
- Spectre d'énergie turbulent E(k) et conservation des gradients raides
- Génération de dashboard interactif HTML autonome
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Mode sans interface graphique pour serveur/headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..mesh.octree_graph import GraphMesh


class FireMapVisualizer:
    """
    Générateur de rendus visuels haute définition et de diagnostics physiques.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colormap feu physique : Noir -> Rouge sombre -> Orange -> Jaune vif -> Blanc
        self.fire_cmap = LinearSegmentedColormap.from_list(
            "fire_phys", ["#111111", "#800c00", "#e65100", "#ffb300", "#fff9c4", "#ffffff"]
        )
        self.risk_cmap = plt.cm.inferno

    def plot_octree_mesh_adaptation(
        self,
        dense_state: torch.Tensor,
        mesh: GraphMesh,
        filename: str = "01_octree_adaptation.png"
    ) -> Path:
        """
        Démontre comment l'Octree raffine la maille uniquement sur le front de flamme.
        """
        temp_field = dense_state[0]
        if temp_field.dim() == 3:
            temp_field = temp_field.max(dim=0).values  # Projection 2D si 3D
        temp_np = temp_field.detach().cpu().numpy()

        coords = mesh.node_coords.detach().cpu().numpy()
        levels = mesh.node_levels.detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0e1117")

        # 1. Champ thermique continu
        im1 = ax1.imshow(temp_np, cmap=self.fire_cmap, origin="lower")
        ax1.set_title("Champ Thermique Continu (T / WRF-SFIRE)", color="white", fontsize=12, pad=10)
        ax1.set_xlabel("Axe X (pixels)", color="white")
        ax1.set_ylabel("Axe Y (pixels)", color="white")
        ax1.tick_params(colors="white")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(colors="white")
        cbar1.set_label("Température normalisée", color="white")

        # 2. Structure du maillage Octree adaptatif
        H, W = temp_np.shape
        x_pix = (coords[:, 0] + 1.0) / 2.0 * (W - 1)
        y_pix = (coords[:, 1] + 1.0) / 2.0 * (H - 1)

        scatter = ax2.scatter(
            x_pix, y_pix, c=levels, cmap="plasma", s=8 + (levels ** 1.8) * 3, alpha=0.85, edgecolors="none"
        )
        ax2.set_title(f"Maillage Adaptatif Octree (N={len(coords)} nœuds)", color="white", fontsize=12, pad=10)
        ax2.set_xlim(0, W - 1)
        ax2.set_ylim(0, H - 1)
        ax2.set_facecolor("#0a0d14")
        ax2.set_xlabel("Axe X (pixels)", color="white")
        ax2.set_ylabel("Axe Y (pixels)", color="white")
        ax2.tick_params(colors="white")
        cbar2 = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(colors="white")
        cbar2.set_label("Profondeur Octree (1=grossier, Max=fin)", color="white")

        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        return save_path

    def plot_rollout_benchmark(
        self,
        gt_state: torch.Tensor,
        pred_state: torch.Tensor,
        metrics: Dict[str, float],
        filename: str = "02_rollout_benchmark.png"
    ) -> Path:
        """
        Comparaison spatiale CFD vs IA avec carte d'erreur absolue et métriques.
        """
        gt_2d = gt_state[0] if gt_state.dim() == 2 else (gt_state[0, 0] if gt_state.dim() == 4 else gt_state[0])
        pred_2d = pred_state[0] if pred_state.dim() == 2 else (pred_state[0, 0] if pred_state.dim() == 4 else pred_state[0])

        gt_np = gt_2d.detach().cpu().numpy()
        pred_np = pred_2d.detach().cpu().numpy()
        err_np = np.abs(pred_np - gt_np)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0e1117")

        # Vérité terrain CFD
        im1 = ax1.imshow(gt_np, cmap=self.fire_cmap, origin="lower")
        ax1.set_title("Ground Truth (CFD WRF-SFIRE)", color="white", fontsize=12, pad=10)
        ax1.tick_params(colors="white")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Prédiction Neural FMM
        im2 = ax2.imshow(pred_np, cmap=self.fire_cmap, origin="lower")
        ax2.set_title("Prédiction (Neural FMM / MGNN)", color="white", fontsize=12, pad=10)
        ax2.tick_params(colors="white")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Erreur absolue
        im3 = ax3.imshow(err_np, cmap="magma", origin="lower")
        ax3.set_title(
            f"Erreur Absolue |CFD - IA|\nIoU={metrics.get('spatial_IoU', 0.0):.3f} | Dice={metrics.get('spatial_Dice', 0.0):.3f}",
            color="white",
            fontsize=11,
            pad=10
        )
        ax3.tick_params(colors="white")
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Ecart absolu (K)", color="white")

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#000000")
            ax.set_xlabel("X (pixels)", color="white")
            ax.set_ylabel("Y (pixels)", color="white")

        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        return save_path

    def plot_monte_carlo_risk(
        self,
        prob_map: torch.Tensor,
        wind_vector: Optional[torch.Tensor] = None,
        filename: str = "03_monte_carlo_hazard.png"
    ) -> Path:
        """
        Carte de probabilité de risque P(feu) issue du faisceau Monte-Carlo GPU avec isochrones.
        """
        # Si tenseur temporel [T, H, W], prendre le pas final
        prob_2d = prob_map[-1] if prob_map.dim() == 3 else prob_map
        p_np = prob_2d.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="#0e1117")
        ax.set_facecolor("#0a0d14")

        # Rendu de la densité de probabilité
        im = ax.imshow(p_np, cmap=self.risk_cmap, origin="lower", vmin=0.0, vmax=1.0)
        
        # Isochrones de confiance : 20%, 50%, 80% de probabilité
        contours = ax.contour(p_np, levels=[0.2, 0.5, 0.8], colors=["#00e5ff", "#76ff03", "#ffff00"], linewidths=1.5)
        ax.clabel(contours, inline=True, fontsize=9, fmt="P=%.1f")

        # Flèche de direction moyenne du vent
        if wind_vector is not None:
            w_u = float(wind_vector[0].item())
            w_v = float(wind_vector[1].item())
            H, W = p_np.shape
            ax.arrow(
                W * 0.15, H * 0.85, w_u * 8.0, w_v * 8.0,
                head_width=2.5, head_length=3.5, fc="#00e5ff", ec="#00e5ff", width=0.8
            )
            ax.text(W * 0.15, H * 0.85 + 4, "Vecteur Vent", color="#00e5ff", fontsize=10, weight="bold")

        ax.set_title("Carte de Risque Probabiliste (Faisceau Monte-Carlo GPU)", color="white", fontsize=13, pad=12)
        ax.set_xlabel("Coordonnée X (pixels)", color="white")
        ax.set_ylabel("Coordonnée Y (pixels)", color="white")
        ax.tick_params(colors="white")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white")
        cbar.set_label("Probabilité de passage du feu P(brûlé)", color="white", fontsize=11)

        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        return save_path

    def plot_physics_diagnostics(
        self,
        gt_state: torch.Tensor,
        pred_state: torch.Tensor,
        filename: str = "04_physics_diagnostics.png"
    ) -> Path:
        """
        Vérification de la cascade d'énergie turbulente FFT et de la raideur du gradient Sobolev.
        """
        gt_2d = gt_state[0] if gt_state.dim() == 2 else (gt_state[0, 0] if gt_state.dim() == 4 else gt_state[0])
        pred_2d = pred_state[0] if pred_state.dim() == 2 else (pred_state[0, 0] if pred_state.dim() == 4 else pred_state[0])

        # 1. FFT Radiale 2D (Spectre d'énergie E(k))
        def get_radial_spectrum(field_2d):
            fft = np.fft.fft2(field_2d.detach().cpu().numpy())
            fft_shift = np.fft.fftshift(fft)
            psd = np.abs(fft_shift) ** 2
            H, W = psd.shape
            cy, cx = H // 2, W // 2
            y, x = np.ogrid[:H, :W]
            r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
            k_bins = np.bincount(r.ravel(), psd.ravel())
            k_counts = np.bincount(r.ravel())
            return k_bins[1:min(cx, cy)] / np.maximum(k_counts[1:min(cx, cy)], 1)

        spec_gt = get_radial_spectrum(gt_2d)
        spec_pred = get_radial_spectrum(pred_2d)
        k_axis = np.arange(1, len(spec_gt) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0e1117")

        # Spectre de Fourier
        ax1.loglog(k_axis, spec_gt, label="CFD Ground Truth (WRF-SFIRE)", color="#00e5ff", lw=2)
        ax1.loglog(k_axis, spec_pred, "--", label="Neural FMM (Notre modèle)", color="#ff5252", lw=2)
        # Pente théorique de Kolmogorov k^(-5/3)
        kolm = spec_gt[0] * (k_axis ** (-5.0 / 3.0))
        ax1.loglog(k_axis, kolm, ":", label="Cascade Kolmogorov k^(-5/3)", color="#ffffff", alpha=0.5)

        ax1.set_title("Spectre d'Énergie Turbulent E(k)", color="white", fontsize=12, pad=10)
        ax1.set_xlabel("Nombre d'onde spatial k", color="white")
        ax1.set_ylabel("Densité spectrale d'énergie", color="white")
        ax1.grid(True, which="both", ls="--", alpha=0.2)
        ax1.legend(facecolor="#1a1f2c", edgecolor="none", labelcolor="white")
        ax1.set_facecolor("#0a0d14")
        ax1.tick_params(colors="white")

        # Distribution des gradients spatiaux ||grad(T)|| (Sobolev H1)
        gy_gt, gx_gt = np.gradient(gt_2d.detach().cpu().numpy())
        gy_pr, gx_pr = np.gradient(pred_2d.detach().cpu().numpy())
        grad_gt = np.sqrt(gx_gt**2 + gy_gt**2).ravel()
        grad_pr = np.sqrt(gx_pr**2 + gy_pr**2).ravel()

        ax2.hist(grad_gt, bins=50, alpha=0.6, label="CFD Gradients", color="#00e5ff", density=True)
        ax2.hist(grad_pr, bins=50, alpha=0.6, label="Neural FMM Gradients", color="#ff5252", density=True)
        ax2.set_yscale("log")
        ax2.set_title("Conservation des Gradients Raides ||∇T||", color="white", fontsize=12, pad=10)
        ax2.set_xlabel("Magnitude du gradient spatial", color="white")
        ax2.set_ylabel("Densité de probabilité (log)", color="white")
        ax2.grid(True, which="both", ls="--", alpha=0.2)
        ax2.legend(facecolor="#1a1f2c", edgecolor="none", labelcolor="white")
        ax2.set_facecolor("#0a0d14")
        ax2.tick_params(colors="white")

        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        return save_path

    def create_animated_rollout_gif(
        self,
        frames_gt: List[np.ndarray],
        frames_pred: List[np.ndarray],
        frames_mesh: List[GraphMesh],
        filename: str = "00_fire_propagation_animated.gif",
        fps: int = 3
    ) -> Path:
        """
        Génère une animation GIF montrant la propagation temporelle du front et l'adaptation de l'Octree.
        """
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0e1117")

        def update(frame_idx):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            gt_np = frames_gt[frame_idx]
            pred_np = frames_pred[frame_idx]
            mesh = frames_mesh[frame_idx]

            # 1. CFD Ground Truth
            ax1.imshow(gt_np, cmap=self.fire_cmap, origin="lower", vmin=0, vmax=2.5)
            ax1.set_title(f"CFD Ground Truth (t = {frame_idx * 60}s)", color="white", fontsize=12)
            ax1.set_facecolor("#0a0d14")
            ax1.tick_params(colors="white")

            # 2. Neural FMM Prediction
            ax2.imshow(pred_np, cmap=self.fire_cmap, origin="lower", vmin=0, vmax=2.5)
            ax2.set_title(f"Neural FMM Prediction (t = {frame_idx * 60}s)", color="white", fontsize=12)
            ax2.set_facecolor("#0a0d14")
            ax2.tick_params(colors="white")

            # 3. Dynamic Octree Mesh
            coords = mesh.node_coords.detach().cpu().numpy()
            levels = mesh.node_levels.detach().cpu().numpy()
            H, W = gt_np.shape
            x_pix = (coords[:, 0] + 1.0) / 2.0 * (W - 1)
            y_pix = (coords[:, 1] + 1.0) / 2.0 * (H - 1)

            ax3.scatter(x_pix, y_pix, c=levels, cmap="plasma", s=8 + (levels ** 1.8) * 3, alpha=0.85)
            ax3.set_xlim(0, W - 1)
            ax3.set_ylim(0, H - 1)
            ax3.set_title(f"Octree Dynamique (N={len(coords)} nœuds)", color="white", fontsize=12)
            ax3.set_facecolor("#0a0d14")
            ax3.tick_params(colors="white")

            for ax in (ax1, ax2, ax3):
                ax.set_xlabel("X (pixels)", color="white")
                ax.set_ylabel("Y (pixels)", color="white")

            plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=len(frames_gt), repeat=True)
        save_path = self.output_dir / filename
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close()
        return save_path

    def generate_html_report(self, metrics: Dict[str, float], plots: List[Path], filename: str = "index.html") -> Path:
        """
        Génère un dashboard HTML interactif autonome pour restitution client / opérationnelle.
        """
        html_path = self.output_dir / filename
        
        cards_html = ""
        for k, v in metrics.items():
            title_clean = k.replace("_", " ").title()
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-title">{title_clean}</div>
                <div class="metric-val">{v:.4f}</div>
            </div>
            """

        images_html = ""
        for p in plots:
            images_html += f"""
            <div class="plot-container">
                <h3>{p.stem.replace('_', ' ').title()}</h3>
                <img src="{p.name}" alt="{p.stem}">
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>FireMap - Rapport de Validation Métrologique CFD/IA</title>
    <style>
        body {{
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            border-bottom: 1px solid #1e293b;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #ff5722;
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            color: #94a3b8;
            margin: 6px 0 0 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: #131c2e;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
        }}
        .metric-title {{
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-val {{
            font-size: 24px;
            font-weight: bold;
            color: #38bdf8;
            margin-top: 8px;
        }}
        .plot-container {{
            background: #131c2e;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .plot-container h3 {{
            margin-top: 0;
            color: #f8fafc;
        }}
        .plot-container img {{
            width: 100%;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FireMap : Validation Physique & Métrologique</h1>
        <p>Moteur Neural FMM / Multi-Scale GNN sur Maillage Adaptatif Octree (Benchmark vs WRF-SFIRE CFD)</p>
    </div>

    <h2>1. Indicateurs Métrologiques Clés (Vérification & Validation)</h2>
    <div class="metrics-grid">
        {cards_html}
    </div>

    <h2>2. Visualisations Spatiales et Diagnostics Physiques</h2>
    {images_html}
</body>
</html>"""
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path
