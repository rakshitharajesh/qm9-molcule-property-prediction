"""Shared inference helper used by both train.py (for early-stopping metrics)
and evaluate.py (for the final results table), so there's exactly one place
that knows how to run a torch model over a loader and return raw-scale
predictions.
"""
from __future__ import annotations

import torch

from .data import TargetScaler


@torch.no_grad()
def run_inference(model, loader, scaler: TargetScaler, device, mc_dropout: bool = False):
    """Returns (y_true_raw, y_pred_raw) as (N, num_targets) tensors on CPU,
    in original physical units (mu in Debye, homo/lumo/gap in Hartree, etc).
    """
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        pred_scaled = model(batch, mc_dropout=mc_dropout)
        pred_raw = scaler.inverse_transform(pred_scaled.cpu())
        ys.append(batch.y.cpu())
        preds.append(pred_raw)
    return torch.cat(ys, dim=0), torch.cat(preds, dim=0)


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true - y_pred).abs().mean().item()
