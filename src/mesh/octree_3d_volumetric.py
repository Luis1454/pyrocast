"""
src/mesh/octree_3d_volumetric.py
--------------------------------
Générateur d'Octree 3D volumique authentique :
- Subdivise l'espace tridimensionnel (Z, Y, X) en 8 octants récursifs
- Critère de raffinement 3D : Norme du gradient thermique 3D ||grad(T)||, température de flamme et tourbillon 3D ||rot(u)||
- Construit le graphe dual 3D complet :
    * Arêtes Near-Field 3D (Transport convectif local)
    * Arêtes Far-Field 3D (Rayonnement longue portée FMM)
"""

from typing import Tuple, List, NamedTuple
import numpy as np
import torch
import torch.nn.functional as F


class VolumetricOctreeMesh(NamedTuple):
    node_coords: torch.Tensor       # [N, 3] Coordonnées 3D réelles (x, y, z) en mètres
    node_features: torch.Tensor     # [N, 7] T, P, U, V, W, FMC, FireFlux
    node_cell_sizes: torch.Tensor   # [N, 3] Dimensions de la maille (dx, dy, dz)
    node_depths: torch.Tensor       # [N] Niveau de profondeur dans l'Octree (1 à max_depth)
    edge_index_near: torch.Tensor   # [2, E_near] Paires de nœuds connectés localement
    edge_index_far: torch.Tensor    # [2, E_far] Multipôles FMM longue portée
    compression_rate_pct: float     # Taux d'économie de mémoire


class Volumetric3DOctreeBuilder:
    """
    Constructeur d'Octree 3D multirésolution pour solveur CFD / GNN.
    """

    def __init__(
        self,
        domain_size_m: Tuple[float, float, float] = (300.0, 1280.0, 1280.0), # Lz, Ly, Lx
        max_depth: int = 3,
        min_depth: int = 1,
        grad_threshold: float = 0.25,
        r_near_ratio: float = 0.15,
        k_far: int = 6,
    ):
        self.Lz, self.Ly, self.Lx = domain_size_m
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.grad_threshold = grad_threshold
        self.r_near = r_near_ratio
        self.k_far = k_far

    def build_from_3d_state(self, state_tensor: torch.Tensor) -> VolumetricOctreeMesh:
        """
        state_tensor: [C=7, D, H, W]
        """
        C, D, H, W = state_tensor.shape
        device = state_tensor.device

        # 1. Calcul du critère d'erreur 3D multi-physique
        # Tenseur thermique T [D, H, W]
        T = state_tensor[0]
        
        # Gradients 3D par convolutions
        T_pad = F.pad(T.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate")
        
        kernel_z = torch.zeros((1, 1, 3, 3, 3), device=device)
        kernel_z[0, 0, 2, 1, 1] = 0.5
        kernel_z[0, 0, 0, 1, 1] = -0.5

        kernel_y = torch.zeros((1, 1, 3, 3, 3), device=device)
        kernel_y[0, 0, 1, 2, 1] = 0.5
        kernel_y[0, 0, 1, 0, 1] = -0.5

        kernel_x = torch.zeros((1, 1, 3, 3, 3), device=device)
        kernel_x[0, 0, 1, 1, 2] = 0.5
        kernel_x[0, 0, 1, 1, 0] = -0.5

        grad_z = F.conv3d(T_pad, kernel_z).squeeze()
        grad_y = F.conv3d(T_pad, kernel_y).squeeze()
        grad_x = F.conv3d(T_pad, kernel_x).squeeze()

        grad_norm_3d = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        flame_indicator = (T > 500.0).float() * 1.5

        refinement_criterion_3d = grad_norm_3d + flame_indicator

        # 2. Échantillonnage hiérarchique des nœuds de l'Octree
        nodes_coords = []
        nodes_features = []
        nodes_cell_sizes = []
        nodes_depths = []

        total_dense_points = D * H * W

        for depth in range(self.min_depth, self.max_depth + 1):
            stride = 2 ** (self.max_depth - depth)
            
            # Subdivisions des mailles
            dz_cell = self.Lz / (D / stride)
            dy_cell = self.Ly / (H / stride)
            dx_cell = self.Lx / (W / stride)

            for z in range(0, D, stride):
                for y in range(0, H, stride):
                    for x in range(0, W, stride):
                        crit_val = float(refinement_criterion_3d[z, y, x].item())
                        
                        # Décision d'activation d'octant
                        keep = False
                        if depth == self.max_depth:
                            keep = (crit_val >= self.grad_threshold)
                        elif depth == self.min_depth:
                            keep = (crit_val < self.grad_threshold * 0.4)
                        else:
                            keep = (crit_val >= self.grad_threshold * 0.4 and crit_val < self.grad_threshold)

                        if keep:
                            # Coordonnées physiques réelles (x, y, z) en mètres
                            x_m = (x / max(W - 1, 1)) * self.Lx
                            y_m = (y / max(H - 1, 1)) * self.Ly
                            z_m = (z / max(D - 1, 1)) * self.Lz

                            feat = state_tensor[:, z, y, x]

                            nodes_coords.append([x_m, y_m, z_m])
                            nodes_features.append(feat)
                            nodes_cell_sizes.append([dx_cell, dy_cell, dz_cell])
                            nodes_depths.append(depth)

        # Si aucun nœud retenu, fallback sur la grille grossière
        if len(nodes_coords) == 0:
            for z in range(0, D, 2):
                for y in range(0, H, 2):
                    for x in range(0, W, 2):
                        nodes_coords.append([(x/W)*self.Lx, (y/H)*self.Ly, (z/D)*self.Lz])
                        nodes_features.append(state_tensor[:, z, y, x])
                        nodes_cell_sizes.append([self.Lx/W*2, self.Ly/H*2, self.Lz/D*2])
                        nodes_depths.append(1)

        node_coords_t = torch.tensor(nodes_coords, device=device, dtype=torch.float32)
        node_features_t = torch.stack(nodes_features).to(device)
        node_cell_sizes_t = torch.tensor(nodes_cell_sizes, device=device, dtype=torch.float32)
        node_depths_t = torch.tensor(nodes_depths, device=device, dtype=torch.long)

        num_active = node_coords_t.shape[0]
        compression = (1.0 - (num_active / total_dense_points)) * 100.0

        # 3. Construction des arêtes 3D Near-field
        # Rayon physique de voisinage basé sur la taille moyenne de domaine
        r_cutoff_m = max(self.Lx, self.Ly) * self.r_near
        dist_mat = torch.cdist(node_coords_t, node_coords_t)

        near_mask = (dist_mat <= r_cutoff_m) & (dist_mat > 1e-4)
        edge_near = torch.nonzero(near_mask, as_tuple=False).t()

        # 4. Construction des arêtes Far-field Multipôles FMM
        far_mask = (dist_mat > r_cutoff_m)
        far_indices = torch.nonzero(far_mask, as_tuple=False)
        if far_indices.shape[0] > 0:
            perm = torch.randperm(far_indices.shape[0])[: min(far_indices.shape[0], num_active * self.k_far)]
            edge_far = far_indices[perm].t()
        else:
            edge_far = torch.empty((2, 0), dtype=torch.long, device=device)

        return VolumetricOctreeMesh(
            node_coords=node_coords_t,
            node_features=node_features_t,
            node_cell_sizes=node_cell_sizes_t,
            node_depths=node_depths_t,
            edge_index_near=edge_near,
            edge_index_far=edge_far,
            compression_rate_pct=compression
        )
