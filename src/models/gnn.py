"""Classical GNN encoders: GCN, edge-aware GIN (GINE), and an edge-conditioned
MPNN (NNConv, the architecture from Gilmer et al. 2017's original QM9 paper).

All three share the RegressionHead/pool_graph plumbing from common.py so the
only thing that differs between them - and between them and the transformer
in transformer.py - is the message-passing layer itself. That's what makes
the "classical GNN vs graph transformer" comparison meaningful rather than
an apples-to-oranges mess of different heads/pooling/training setups.

Note on edge features: plain GCNConv has no mechanism to use bond-type
information (it only propagates based on the adjacency structure) - this is
a real, worth-reporting limitation of GCN for molecular graphs, not an
oversight here. GINE and NNConv both consume edge_attr.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, NNConv

from .common import RegressionHead, pool_graph


class GCNRegressor(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_targets,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = RegressionHead(hidden_channels * 2, hidden_channels, num_targets, dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = pool_graph(x, batch)
        return self.head(graph_embedding, mc_dropout=mc_dropout)


class GINRegressor(nn.Module):
    """Edge-aware GIN (GINEConv). GIN's sum-aggregation + MLP update is
    strictly more expressive than GCN's mean aggregation (it can distinguish
    graph structures GCN provably cannot), and GINEConv additionally folds
    in bond features via a learned projection."""

    def __init__(self, in_channels, edge_dim, hidden_channels, num_targets,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = RegressionHead(hidden_channels * 2, hidden_channels, num_targets, dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = pool_graph(x, batch)
        return self.head(graph_embedding, mc_dropout=mc_dropout)


class MPNNRegressor(nn.Module):
    """Edge-conditioned MPNN (NNConv) - the message-passing scheme from the
    original Gilmer et al. 2017 QM9 paper: each edge's features are passed
    through a small network to produce a per-edge weight matrix, so the
    message an atom sends depends on the bond connecting it, not just on
    its own features. This is the most expressive of the three classical
    models but also the most parameter-heavy (the edge network outputs a
    full hidden x hidden matrix per edge), so keep hidden_channels modest.
    """

    def __init__(self, in_channels, edge_dim, hidden_channels, num_targets,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels * hidden_channels),
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn, aggr="mean"))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.head = RegressionHead(hidden_channels * 2, hidden_channels, num_targets, dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)
        h = x
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, edge_attr)
            m = norm(m)
            m = F.relu(m)
            h = self.gru(m, h)  # GRU-based update, as in the original MPNN paper
            h = F.dropout(h, p=self.dropout, training=self.training)
        graph_embedding = pool_graph(h, batch)
        return self.head(graph_embedding, mc_dropout=mc_dropout)


def build_classical_gnn(name: str, in_channels: int, edge_dim: int, num_targets: int,
                         hidden_channels: int = 64, num_layers: int = 4, dropout: float = 0.1):
    name = name.lower()
    if name == "gcn":
        return GCNRegressor(in_channels, edge_dim, hidden_channels, num_targets, num_layers, dropout)
    if name == "gin":
        return GINRegressor(in_channels, edge_dim, hidden_channels, num_targets, num_layers, dropout)
    if name == "mpnn":
        return MPNNRegressor(in_channels, edge_dim, hidden_channels, num_targets,
                              min(num_layers, 3), dropout)
    raise ValueError(f"Unknown classical GNN: {name}")
