"""
src/mesh/octree_graph.py
------------------------
Conversion haute performance d'une grille dense en graphe Octree multi-échelle.
- Haute résolution locale (niveau fin) sur le front thermique.
- Basse résolution (niveaux grossiers) en zone froide ou consumée.
- Arêtes locales (Near-field) et arêtes multipôles (Far-field FMM).
"""

from typing import NamedTuple, Tuple, List
import torch
import torch.nn.functional as F


class GraphMesh(NamedTuple):
    node_features: torch.Tensor    # [N, C]
    node_coords: torch.Tensor      # [N, 3] normalisées [-1, 1]
    node_levels: torch.Tensor      # [N] Profondeur de l'Octree
    edge_index_near: torch.Tensor  # [2, E_near] Transport local (Convection/Diffusion)
    edge_index_far: torch.Tensor   # [2, E_far] Rayonnement FMM (M2L)


class AdaptiveOctreeGraphBuilder:
    """
    Constructeur d'Octree guidé par l'erreur physique et le gradient thermique.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_depth: int = 1,
        grad_threshold: float = 0.25,
        temp_ignition_threshold: float = 1.0,
        r_near: float = 0.12,
        k_far: int = 6,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.grad_threshold = grad_threshold
        self.temp_ignition_threshold = temp_ignition_threshold
        self.r_near = r_near
        self.k_far = k_far

    @staticmethod
    def _compute_spatial_gradients(tensor: torch.Tensor) -> torch.Tensor:
        """Différences finies 2D ou 3D pour estimer ||grad(T)||."""
        D = tensor.shape[2]
        if D == 1:
            temp_2d = tensor[:, :, 0]  # [1, 1, H, W]
            grad_y = temp_2d[:, :, 2:, 1:-1] - temp_2d[:, :, :-2, 1:-1]
            grad_x = temp_2d[:, :, 1:-1, 2:] - temp_2d[:, :, 1:-1, :-2]
            grad_mag = torch.sqrt(grad_y**2 + grad_x**2 + 1e-8)
            return F.pad(grad_mag, (1, 1, 1, 1), mode="replicate").unsqueeze(2)
        else:
            grad_z = tensor[:, :, 2:, 1:-1, 1:-1] - tensor[:, :, :-2, 1:-1, 1:-1]
            grad_y = tensor[:, :, 1:-1, 2:, 1:-1] - tensor[:, :, 1:-1, :-2, 1:-1]
            grad_x = tensor[:, :, 1:-1, 1:-1, 2:] - tensor[:, :, 1:-1, 1:-1, :-2]
            grad_mag = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + 1e-8)
            return F.pad(grad_mag, (1, 1, 1, 1, 1, 1), mode="replicate")

    def build_mesh(self, state_dense: torch.Tensor) -> GraphMesh:
        """
        state_dense: [C, D, H, W] ou [C, H, W]
        """
        if state_dense.dim() == 3:
            state_dense = state_dense.unsqueeze(1)  # [C, 1, H, W]

        C, D, H, W = state_dense.shape
        device = state_dense.device

        temp = state_dense[0:1].unsqueeze(0)  # [1, 1, D, H, W]
        grad_mag = self._compute_spatial_gradients(temp)[0, 0]  # [D, H, W]
        temp_val = temp[0, 0]  # [D, H, W]

        # Critère d'activation : fort gradient OU température proche/au-dessus de l'ignition
        refinement_criterion = grad_mag + 1.2 * (temp_val > self.temp_ignition_threshold).float()

        coords_list, feats_list, levels_list = [], [], []

        for depth in range(self.min_depth, self.max_depth + 1):
            stride = 2 ** (self.max_depth - depth)
            sub_crit = refinement_criterion[::stride, ::stride, ::stride]
            sub_state = state_dense[:, ::stride, ::stride, ::stride]

            if depth == self.max_depth:
                mask = sub_crit >= self.grad_threshold
            elif depth == self.min_depth:
                mask = sub_crit < (self.grad_threshold / 2.0)
            else:
                mask = (sub_crit >= self.grad_threshold / (2 ** (self.max_depth - depth))) & (
                    sub_crit < self.grad_threshold
                )

            if not mask.any():
                continue

            indices = torch.nonzero(mask, as_tuple=False)
            z_n = (indices[:, 0].float() * stride) / max(D - 1, 1) * 2.0 - 1.0
            y_n = (indices[:, 1].float() * stride) / max(H - 1, 1) * 2.0 - 1.0
            x_n = (indices[:, 2].float() * stride) / max(W - 1, 1) * 2.0 - 1.0

            coords = torch.stack([x_n, y_n, z_n], dim=-1)
            feats = sub_state[:, indices[:, 0], indices[:, 1], indices[:, 2]].permute(1, 0)

            coords_list.append(coords)
            feats_list.append(feats)
            levels_list.append(torch.full((coords.shape[0],), depth, device=device, dtype=torch.long))

        if not coords_list:
            # Fallback en grille grossière uniforme si aucun gradient
            stride = 2 ** (self.max_depth - self.min_depth)
            sub_state = state_dense[:, ::stride, ::stride, ::stride]
            d_s, h_s, w_s = sub_state.shape[1:]
            gz, gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, d_s, device=device),
                torch.linspace(-1, 1, h_s, device=device),
                torch.linspace(-1, 1, w_s, device=device),
                indexing="ij",
            )
            all_coords = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
            all_features = sub_state.reshape(C, -1).permute(1, 0)
            all_levels = torch.full((all_coords.shape[0],), self.min_depth, device=device, dtype=torch.long)
        else:
            all_coords = torch.cat(coords_list, dim=0)
            all_features = torch.cat(feats_list, dim=0)
            all_levels = torch.cat(levels_list, dim=0)

        # Arêtes Near-field et Far-field
        edge_near = self._build_radius_edges(all_coords, self.r_near)
        edge_far = self._build_fmm_multipole_edges(all_coords, all_levels, self.k_far)

        return GraphMesh(
            node_features=all_features,
            node_coords=all_coords,
            node_levels=all_levels,
            edge_index_near=edge_near,
            edge_index_far=edge_far,
        )

    @staticmethod
    def _build_radius_edges(coords: torch.Tensor, radius: float) -> torch.Tensor:
        dist = torch.cdist(coords, coords, p=2.0)
        adj = (dist < radius) & (dist > 1e-6)
        return torch.nonzero(adj).t().contiguous()

    @staticmethod
    def _build_fmm_multipole_edges(coords: torch.Tensor, levels: torch.Tensor, k: int) -> torch.Tensor:
        coarse_mask = levels <= levels.median()
        fine_mask = levels > levels.median()

        if not coarse_mask.any() or not fine_mask.any():
            dist = torch.cdist(coords, coords, p=2.0)
            _, topk = torch.topk(dist, k=min(k, coords.shape[0]), largest=False)
            src = torch.arange(coords.shape[0], device=coords.device).unsqueeze(1).expand(-1, topk.shape[1]).reshape(-1)
            dst = topk.reshape(-1)
            return torch.stack([src, dst], dim=0)

        fine_idx = torch.nonzero(fine_mask).squeeze(-1)
        coarse_idx = torch.nonzero(coarse_mask).squeeze(-1)

        dist_cross = torch.cdist(coords[fine_idx], coords[coarse_idx], p=2.0)
        _, topk_coarse = torch.topk(dist_cross, k=min(k, coarse_idx.shape[0]), largest=False)

        src = fine_idx.unsqueeze(1).expand(-1, topk_coarse.shape[1]).reshape(-1)
        dst = coarse_idx[topk_coarse.reshape(-1)]
        return torch.stack([src, dst], dim=0)
