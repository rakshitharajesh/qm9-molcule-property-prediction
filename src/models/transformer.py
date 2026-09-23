"""Attention-based / graph transformer encoders.

Two models, deliberately building on each other so the comparison tells a
story rather than just producing two more numbers in a table:

- GATv2Regressor: local attention only (each atom still only attends to its
  bonded neighbors, but with learned attention weights instead of GCN's
  fixed averaging). This is the natural "attention added to a classical
  GNN" stepping stone, not yet a real graph transformer.

- GraphGPSRegressor: an actual graph transformer, using PyG's GPSConv
  (Rampasek et al. 2022, "Recipe for a General, Powerful, Scalable Graph
  Transformer"). Each layer combines (a) a local edge-aware MPNN (GINEConv)
  so bonded-neighbor structure is still modeled directly, with (b) full
  multi-head self-attention across every atom in the molecule regardless of
  distance. Crucially, nodes are also given a random-walk structural
  encoding before the first layer - without this, global attention has no
  way to tell graph structure apart from an unordered set of atoms, and
  usually loses to a plain GNN. This is the single most common reason a
  from-scratch "graph transformer" ends up underperforming GCN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GPSConv
from torch_geometric.transforms import AddRandomWalkPE

from .common import RegressionHead, pool_graph

PE_WALK_LENGTH = 16
PE_ATTR_NAME = "pe"


def add_structural_encoding(graphs: list, walk_length: int = PE_WALK_LENGTH) -> list:
    """Attach a random-walk structural/positional encoding to every graph.
    Call this once on train/val/test lists right after loading, before
    building DataLoaders, so it isn't recomputed every epoch.

    Falls back to zeros for the rare graph where PE computation fails (e.g.
    a single isolated atom with no edges) rather than crashing the run.
    """
    transform = AddRandomWalkPE(walk_length=walk_length, attr_name=PE_ATTR_NAME)
    out = []
    n_failed = 0
    for g in graphs:
        try:
            g = transform(g)
        except Exception:
            g[PE_ATTR_NAME] = torch.zeros(g.x.size(0), walk_length)
            n_failed += 1
        out.append(g)
    if n_failed:
        print(f"[add_structural_encoding] fell back to zero-PE for {n_failed}/{len(graphs)} graphs")
    return out


class GATv2Regressor(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_targets,
                 num_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads,
                          edge_dim=edge_dim, dropout=dropout)
            )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = RegressionHead(hidden_channels * 2, hidden_channels, num_targets, dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = pool_graph(x, batch)
        return self.head(graph_embedding, mc_dropout=mc_dropout)


class GraphGPSRegressor(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_targets,
                 num_layers=4, heads=4, dropout=0.1, pe_dim=PE_WALK_LENGTH):
        super().__init__()
        if hidden_channels - pe_dim < 8:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must leave room for both the "
                f"positional encoding (pe_dim={pe_dim}) and actual node features "
                f"(need >=8 dims). Either raise --hidden or lower pe_dim - and make "
                f"sure add_structural_encoding() was called with the same walk_length "
                f"as pe_dim here, or the two will silently disagree."
            )
        self.pe_dim = pe_dim
        self.input_proj = nn.Linear(in_channels, hidden_channels - pe_dim)
        self.pe_proj = nn.Linear(pe_dim, pe_dim)  # small learned reweighting of the raw PE
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=edge_dim,
            )
            self.convs.append(
                GPSConv(hidden_channels, local_conv, heads=heads, dropout=dropout)
            )
        self.dropout = dropout
        self.head = RegressionHead(hidden_channels * 2, hidden_channels, num_targets, dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        pe = getattr(data, PE_ATTR_NAME, None)
        if pe is None:
            raise ValueError(
                "Graphs have no positional encoding - call "
                "add_structural_encoding(graphs) before building the DataLoader."
            )
        x = torch.cat([self.input_proj(x), self.pe_proj(pe)], dim=-1)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = pool_graph(x, batch)
        return self.head(graph_embedding, mc_dropout=mc_dropout)


def build_transformer(name: str, in_channels: int, edge_dim: int, num_targets: int,
                       hidden_channels: int = 64, num_layers: int = 4, dropout: float = 0.1,
                       pe_dim: int = PE_WALK_LENGTH):
    name = name.lower()
    if name in ("gat", "gatv2"):
        return GATv2Regressor(in_channels, edge_dim, hidden_channels, num_targets,
                               num_layers, dropout=dropout)
    if name in ("gps", "graphgps", "transformer"):
        return GraphGPSRegressor(in_channels, edge_dim, hidden_channels, num_targets,
                                  num_layers, dropout=dropout, pe_dim=pe_dim)
    raise ValueError(f"Unknown transformer variant: {name}")
