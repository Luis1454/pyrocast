"""
src/models/mgnn_fmm.py
----------------------
Modèle Multiscale Graph Neural Network (MGNN) couplé au Neural Fast Multipole Method (FMM).
Résout la propagation non-linéaire du front de flamme en complexité O(N).
"""

import torch
import torch.nn as nn
from .fmm_layers import NearFieldConv, NeuralFMMMultipoleLayer
from ..mesh.octree_graph import GraphMesh


class NeuralFMM_MGNN(nn.Module):
    """
    Réseau Hybride pour la propagation thermodynamique de feux de forêt.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 128,
        out_channels: int = 7,
        num_layers: int = 4,
        forcing_dim: int = 3,  # Champ de vent 3D dynamique (U, V, W)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Encodage nœuds + coordonnées 3D (x,y,z) + forçage météo 3D (u,v,w)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels + 3 + forcing_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Blocs alternés Near-field (Convection/Diffusion 3D) + Far-field FMM (Rayonnement O(N))
        # edge_attr_dim = 5 : [dx, dy, dz, ||dr||, u_3d . dr_unit (flux convectif directionnel)]
        self.near_convs = nn.ModuleList([
            NearFieldConv(hidden_dim, hidden_dim, edge_attr_dim=5) for _ in range(num_layers)
        ])
        self.far_fmms = nn.ModuleList([
            NeuralFMMMultipoleLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Décodeur Euler résiduel
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, mesh: GraphMesh, wind_forcing_3d: torch.Tensor) -> torch.Tensor:
        """
        mesh: Structure du graphe Octree (nœuds 3D et arêtes)
        wind_forcing_3d: Champ de vent 3D [3] (global) ou [N, 3] (champ 3D hétérogène par nœud)
        """
        x = mesh.node_features
        coords = mesh.node_coords

        if wind_forcing_3d.dim() == 1:
            # Si forçage atmosphérique global 3D (U, V, W) -> expansion sur les N nœuds
            forcing = wind_forcing_3d.unsqueeze(0).expand(x.shape[0], -1)
        else:
            forcing = wind_forcing_3d

        h = self.node_encoder(torch.cat([x, coords, forcing], dim=-1))

        # 1. Attributs des arêtes Near-field 3D
        src_n, dst_n = mesh.edge_index_near[0], mesh.edge_index_near[1]
        dr_near = coords[dst_n] - coords[src_n]  # [E, 3]
        dist_near = torch.norm(dr_near, dim=-1, keepdim=True)  # [E, 1]
        unit_dr = dr_near / (dist_near + 1e-6)  # Vecteur unitaire directionnel

        # Projection convective 3D : u_local . r_unit (transport advectif le long de l'arête)
        # On utilise le vecteur vitesse 3D du nœud source (canaux 2,3,4 de x ou forcing)
        wind_src = forcing[src_n]  # [E, 3]
        convective_flux = torch.sum(wind_src * unit_dr, dim=-1, keepdim=True)  # [E, 1]

        edge_attr_near = torch.cat([dr_near, dist_near, convective_flux], dim=-1)  # [E, 5]

        # 2. Attributs des arêtes Far-field 3D
        src_f, dst_f = mesh.edge_index_far[0], mesh.edge_index_far[1]
        dist_far = torch.norm(coords[dst_f] - coords[src_f], dim=-1)

        # 3. Message Passing multi-échelle
        for near_op, far_op, norm in zip(self.near_convs, self.far_fmms, self.norms):
            h_near = near_op(h, mesh.edge_index_near, edge_attr_near)
            h_far = far_op(h, mesh.edge_index_far, dist_far)
            h = norm(h + h_near + h_far)

        delta_state = self.decoder(h)
        return x + delta_state
