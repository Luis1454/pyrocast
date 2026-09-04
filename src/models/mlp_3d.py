"""
src/models/mlp_3d.py
--------------------
Modèle Neural Implicit Representation / MLP 3D Spatio-Temporel pour la Thermodynamique :
f_theta(x, y, z, t, u_wind, v_wind, w_wind) -> (T_norm, P_norm, u_norm, v_norm, w_norm, soot_norm, hrr_norm)
- Encodage positionnel harmonique de Fourier (Fourier Features)
- Réseau Dense Multi-couches avec connexions résiduelles (Skip Connections)
- Normalisation adimensionnelle standardisée pour convergence rapide et stable
- Évaluation continue sur coordonnées arbitraires (x, y, z, t) avec conversion physique SI
"""

import math
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureEncoder(nn.Module):
    """
    Encodeur positionnel harmonique de Fourier pour injecter les hautes fréquences spatiales.
    gamma(p) = [p, sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]
    """

    def __init__(self, in_dim: int = 4, num_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Fréquences logarithmiques : 2^0, 2^1, ..., 2^{L-1}
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        self.out_dim = in_dim + 2 * in_dim * num_frequencies if include_input else 2 * in_dim * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords : [..., in_dim]
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi  # [..., in_dim, num_freq]
        sin_feat = torch.sin(scaled).flatten(start_dim=-2)
        cos_feat = torch.cos(scaled).flatten(start_dim=-2)

        if self.include_input:
            return torch.cat([coords, sin_feat, cos_feat], dim=-1)
        return torch.cat([sin_feat, cos_feat], dim=-1)


class ThermodynamicMLP3D(nn.Module):
    """
    MLP 3D Continu pour les champs thermo-dynamiques couplés feu-atmosphère :
    Entrées : Coordonnées 3D + Temps [x, y, z, t] (normalisées) et Forçage Météo 3D [u_w, v_w, w_w]
    Sorties Adimensionnelles : [T_norm, P_norm, u_norm, v_norm, w_norm, soot_norm, hrr_norm] in [-1, 1] ou [0, 1]
    """

    def __init__(
        self,
        coord_dim: int = 4,      # x, y, z, t normalisés
        forcing_dim: int = 3,    # u, v, w météo
        out_dim: int = 7,        # T, P, u, v, w, soot, hrr
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_frequencies: int = 6,
        skip_connection_layer: int = 2
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.forcing_dim = forcing_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.skip_layer = skip_connection_layer

        self.encoder = FourierFeatureEncoder(in_dim=coord_dim, num_frequencies=num_frequencies)
        in_features = self.encoder.out_dim + forcing_dim

        self.input_layer = nn.Linear(in_features, hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == skip_connection_layer:
                self.middle_layers.append(nn.Linear(hidden_dim + in_features, hidden_dim))
            else:
                self.middle_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Tête de sortie standardisée
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

        # Constantes physiques de dé-normalisation SI
        self.register_buffer("temp_ambient", torch.tensor(298.15))
        self.register_buffer("temp_scale", torch.tensor(1500.0))
        self.register_buffer("p_ambient", torch.tensor(101325.0))
        self.register_buffer("p_scale", torch.tensor(500.0)) # Variations hydrodynamiques
        self.register_buffer("vel_scale", torch.tensor(30.0))
        self.register_buffer("soot_scale", torch.tensor(5.0))
        self.register_buffer("hrr_scale", torch.tensor(250.0))

    def forward(
        self,
        coords: torch.Tensor,
        wind_forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retourne le vecteur adimensionnel normalisé standardisé [..., 7]
        """
        encoded_coords = self.encoder(coords)

        if wind_forcing is not None:
            if wind_forcing.dim() == 1:
                wind_expanded = wind_forcing.view(*([1] * (coords.dim() - 1)), -1).expand(*coords.shape[:-1], -1)
            else:
                wind_expanded = wind_forcing
            feat_in = torch.cat([encoded_coords, wind_expanded], dim=-1)
        else:
            dummy_wind = torch.zeros(*coords.shape[:-1], self.forcing_dim, device=coords.device)
            feat_in = torch.cat([encoded_coords, dummy_wind], dim=-1)

        h = F.gelu(self.input_layer(feat_in))
        h = self.layer_norms[0](h)

        for i, layer in enumerate(self.middle_layers):
            if i == self.skip_layer:
                h = torch.cat([h, feat_in], dim=-1)
            h = F.gelu(layer(h))
            h = self.layer_norms[i + 1](h)

        raw_out = self.output_head(h)
        return raw_out

    def forward_physical(
        self,
        coords: torch.Tensor,
        wind_forcing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retourne les grandeurs physiques dé-normalisées en unités SI :
        [T(K), P(Pa), u(m/s), v(m/s), w(m/s), soot(g/m3), hrr(kW/m3)]
        """
        norm_out = self.forward(coords, wind_forcing)
        phys = torch.zeros_like(norm_out)

        phys[..., 0] = self.temp_ambient + F.relu(norm_out[..., 0]) * self.temp_scale
        phys[..., 1] = self.p_ambient + norm_out[..., 1] * self.p_scale
        phys[..., 2] = norm_out[..., 2] * self.vel_scale
        phys[..., 3] = norm_out[..., 3] * self.vel_scale
        phys[..., 4] = F.relu(norm_out[..., 4]) * self.vel_scale # Updraft >= 0
        phys[..., 5] = F.relu(norm_out[..., 5]) * self.soot_scale
        phys[..., 6] = F.relu(norm_out[..., 6]) * self.hrr_scale

        return phys

    @torch.no_grad()
    def evaluate_grid_3d(
        self,
        grid_shape: Tuple[int, int, int] = (16, 32, 32), # (D, H, W)
        domain_bounds: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0),
        time_val: float = 0.5,
        wind_vec: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        """
        Évalue le champ continu 3D sur une grille dense régulière discrétisée en unités physiques.
        """
        self.eval()
        D, H, W = grid_shape
        xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds

        xs = torch.linspace(xmin, xmax, W, device=device)
        ys = torch.linspace(ymin, ymax, H, device=device)
        zs = torch.linspace(zmin, zmax, D, device=device)

        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        grid_t = torch.full_like(grid_z, time_val)

        coords = torch.stack([grid_x, grid_y, grid_z, grid_t], dim=-1).view(-1, 4)
        
        if wind_vec is None:
            wind_vec = torch.tensor([1.5, -0.8, 0.0], device=device)

        preds_phys = self.forward_physical(coords, wind_vec) # [D*H*W, 7]
        preds_3d = preds_phys.view(D, H, W, self.out_dim)

        return {
            "temperature_k": preds_3d[..., 0],
            "pressure": preds_3d[..., 1],
            "u": preds_3d[..., 2],
            "v": preds_3d[..., 3],
            "w": preds_3d[..., 4],
            "soot_density": preds_3d[..., 5],
            "hrr": preds_3d[..., 6],
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z
        }
