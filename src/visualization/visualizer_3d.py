"""
src/visualization/visualizer_3d.py
----------------------------------
Visualisation volumique 3D haute fidélité pour solveur CFD / FMM :
- Coupes orthogonales 3D (Plan vertical X-Z de panache, plan Y-Z, plan sol X-Y)
- Rendu 3D isovolume du front de flamme avec champ vectoriel de vitesse (u, v, w)
- Décomposition 3D des mailles de l'Octree adaptatif
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Volumetric3DVisualizer:
    """
    Moteur de rendu 3D scientifique pour l'aéro-thermodynamique des feux de forêts.
    """

    def __init__(self, output_dir: str = "reports/3d_cfd"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_3d_orthogonal_slices(
        self,
        temperature_3d: torch.Tensor,
        u_3d: torch.Tensor,
        w_3d: torch.Tensor,
        dx: float = 20.0,
        dz: float = 15.0,
        filename: str = "01_cfd_3d_orthogonal_slices.png"
    ) -> Path:
        """
        Affiche les coupes orthogonales 3D révélant la structure verticale du panache ascendant.
        """
        T = temperature_3d.detach().cpu().numpy()
        U = u_3d.detach().cpu().numpy()
        W = w_3d.detach().cpu().numpy()
        D, H, W_dim = T.shape

        mid_y = H // 2
        mid_z = 1  # Couche proche du sol

        fig = plt.figure(figsize=(15, 6), facecolor="#0e1117")
        
        # 1. Coupe Verticale X-Z (Vue en élévation du panache de feu)
        ax1 = fig.add_subplot(1, 2, 1, facecolor="#0a0d14")
        x_m = np.linspace(0, (W_dim - 1) * dx, W_dim)
        z_m = np.linspace(0, (D - 1) * dz, D)
        X_grid, Z_grid = np.meshgrid(x_m, z_m)

        slice_xz = T[:, mid_y, :]
        im1 = ax1.pcolormesh(X_grid, Z_grid, slice_xz, cmap="inferno", shading="auto", vmin=300, vmax=1800)
        
        # Vecteurs vitesse (U, W) dans le plan vertical
        step_x, step_z = max(1, W_dim // 16), max(1, D // 8)
        ax1.quiver(
            X_grid[::step_z, ::step_x], Z_grid[::step_z, ::step_x],
            U[::step_z, mid_y, ::step_x], W[::step_z, mid_y, ::step_x],
            color="#00e5ff", alpha=0.9, scale=120.0
        )
        ax1.set_title("COUPE VERTICALE X-Z DU PANACHE THERMIQUE (CFD 3D)\nFlottabilité & Vitesse Ascendante W(z)", color="white", fontsize=11, pad=10)
        ax1.set_xlabel("Distance Horizontale X (m)", color="white")
        ax1.set_ylabel("Altitude Z (m)", color="white")
        ax1.tick_params(colors="white")

        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label("Température du Gaz T (K)", color="white")
        cb1.ax.tick_params(colors="white")

        # 2. Coupe Horizontale X-Y (Vue du dessus au sol z=1)
        ax2 = fig.add_subplot(1, 2, 2, facecolor="#0a0d14")
        y_m = np.linspace(0, (H - 1) * dx, H)
        X_xy, Y_xy = np.meshgrid(x_m, y_m)

        slice_xy = T[mid_z, :, :]
        im2 = ax2.pcolormesh(X_xy, Y_xy, slice_xy, cmap="inferno", shading="auto", vmin=300, vmax=1800)
        ax2.set_title(f"COUPE HORIZONTALE X-Y AU SOL (Altitude z={mid_z * dz:.0f}m)\nFront de Flamme et Aspiration In-draft", color="white", fontsize=11, pad=10)
        ax2.set_xlabel("Distance X (m)", color="white")
        ax2.set_ylabel("Distance Y (m)", color="white")
        ax2.tick_params(colors="white")

        cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("Température au Sol (K)", color="white")
        cb2.ax.tick_params(colors="white")

        plt.tight_layout()
        out_file = self.output_dir / filename
        plt.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

        return out_file

    def plot_3d_volumetric_plume(
        self,
        temperature_3d: torch.Tensor,
        u_3d: torch.Tensor,
        v_3d: torch.Tensor,
        w_3d: torch.Tensor,
        dx: float = 20.0,
        dz: float = 15.0,
        filename: str = "02_cfd_3d_volumetric_plume.png"
    ) -> Path:
        """
        Rendu 3D de l'isovolume de combustion et des vecteurs 3D de pyro-convection.
        """
        T = temperature_3d.detach().cpu().numpy()
        U = u_3d.detach().cpu().numpy()
        V = v_3d.detach().cpu().numpy()
        W = w_3d.detach().cpu().numpy()
        D, H, W_dim = T.shape

        fig = plt.figure(figsize=(12, 10), facecolor="#0e1117")
        ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="#0a0d14")

        # Sélection des voxels en feu (T >= 600K)
        flame_mask = T >= 600.0
        z_idx, y_idx, x_idx = np.where(flame_mask)

        if len(z_idx) > 0:
            x_pts = x_idx * dx
            y_pts = y_idx * dx
            z_pts = z_idx * dz
            t_pts = T[z_idx, y_idx, x_idx]

            scatter = ax.scatter(
                x_pts, y_pts, z_pts,
                c=t_pts, cmap="hot", s=25, alpha=0.75, edgecolors="none", vmin=600, vmax=1800
            )

            # Vecteurs d'ascendance 3D
            step = max(1, len(z_idx) // 40)
            ax.quiver(
                x_pts[::step], y_pts[::step], z_pts[::step],
                U[z_idx[::step], y_idx[::step], x_idx[::step]],
                V[z_idx[::step], y_idx[::step], x_idx[::step]],
                W[z_idx[::step], y_idx[::step], x_idx[::step]],
                length=40.0, normalize=True, color="#00e5ff", alpha=0.85
            )

        ax.set_title("STRUCTURE VOLUMIQUE 3D DU PANACHE DE FEU & PYRO-CONVECTION\nIso-volume T >= 600K avec Vecteurs Vitesse 3D (u, v, w)", color="white", fontsize=12, pad=15)
        ax.set_xlabel("X (mètres)", color="white")
        ax.set_ylabel("Y (mètres)", color="white")
        ax.set_zlabel("Altitude Z (mètres)", color="white")

        ax.tick_params(colors="white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Orientation de la caméra 3D
        ax.view_init(elev=28, azim=-55)

        out_file = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

        return out_file

    def plot_3d_octree_mesh(
        self,
        node_coords_3d: torch.Tensor,
        node_depths: torch.Tensor,
        filename: str = "03_octree_3d_decomposition.png"
    ) -> Path:
        """
        Visualise la distribution 3D spatiale des cellules actives de l'Octree.
        """
        coords = node_coords_3d.detach().cpu().numpy()
        depths = node_depths.detach().cpu().numpy()

        fig = plt.figure(figsize=(11, 9), facecolor="#0e1117")
        ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="#0a0d14")

        colors = ["#4ade80", "#fbbf24", "#ef4444"]
        for d in np.unique(depths):
            mask = depths == d
            ax.scatter(
                coords[mask, 0], coords[mask, 1], coords[mask, 2],
                s=15 + d * 15, alpha=0.65, label=f"Octree Niveau {d} (Raffinement {2**d}x)"
            )

        ax.set_title("MAILLAGE ADAPTATIF 3D DE L'OCTREE DYNAMIQUE\nConcentration Spatiale des Cellules sur le Panache", color="white", fontsize=12, pad=15)
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_zlabel("Z (m)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e293b", edgecolor="none", labelcolor="white")

        ax.view_init(elev=25, azim=-60)
        out_file = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

        return out_file
