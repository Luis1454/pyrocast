"""
src/models/fmm_layers.py
------------------------
Opérateurs neuronaux du Fast Multipole Method :
- Local-to-Local (L2L) : Convection/Diffusion par passage de message local.
- Multipole-to-Local (M2L) : Rayonnement thermique longue portée avec complexité O(N).
"""

import torch
import torch.nn as nn


class NearFieldConv(nn.Module):
    """
    Opérateur L2L : Message passing local avec encodage des vitesses de vent et distances relatives.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int = 4):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_attr_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Agrégation de flux local
        agg = torch.zeros(x.shape[0], messages.shape[1], device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages)

        return self.update_mlp(torch.cat([x, agg], dim=-1))


class NeuralFMMMultipoleLayer(nn.Module):
    """
    Opérateur M2L : Agrégation par attention multi-têtes entre nœuds multipôles
    et nœuds récepteurs pour calculer le rayonnement global en O(N).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self, x: torch.Tensor, edge_index_far: torch.Tensor, dist_far: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index_far[0], edge_index_far[1]

        q = self.q_proj(x[dst]).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x[src]).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x[src]).view(-1, self.num_heads, self.head_dim)

        # Atténuation radiative thermique ~ 1 / (r^2 + eps)
        decay = 1.0 / (dist_far.unsqueeze(-1).unsqueeze(-1) ** 2 + 1e-4)
        attn_scores = (q * k).sum(dim=-1, keepdim=True) * self.scale * decay
        attn_weights = torch.sigmoid(attn_scores)

        far_messages = (attn_weights * v).view(-1, self.num_heads * self.head_dim)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, far_messages)

        return self.out_proj(agg)
