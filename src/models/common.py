"""Shared building blocks so every model (classical GNN or transformer)
exposes the same interface: forward(batch) -> (num_graphs, num_targets)
tensor. Keeping this identical across models is what makes the comparison
fair - same pooling, same head, same training loop, only the encoder layers
differ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool


def pool_graph(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean + sum pooling concatenated. Mean pooling normalizes for molecule
    size, sum pooling keeps a signal of size/extensive properties (relevant
    for e.g. polarizability, which scales roughly with molecule size) -
    together they give the head more to work with than either alone."""
    return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)


class RegressionHead(nn.Module):
    """Graph-embedding -> per-target scalar MLP head, shared by every
    encoder. Dropout stays active at eval time when `mc_dropout=True` is
    passed to forward - this is what MC-dropout uncertainty (see
    evaluate.py) relies on.
    """

    # plain 3 layer mlp narrowing the pooled embeddings down to 5 targets

    def __init__(self, in_dim: int, hidden_dim: int, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, graph_embedding: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        if mc_dropout:
            was_training = self.training
            self.train()  # force dropout on even in eval()
            out = self.net(graph_embedding)
            self.train(was_training)
            return out
        return self.net(graph_embedding)
