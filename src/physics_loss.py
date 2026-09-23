"""Physics-consistency loss: predicted gap should equal predicted LUMO minus
predicted HOMO. This only makes chemical sense in the *original* (unscaled)
units - each target is independently standardized for training, so the
linear relationship gap = lumo - homo does not hold between the
standardized values. Always inverse_transform predictions before calling
this.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gap_consistency_loss(
    pred_raw: torch.Tensor, target_names: list[str],
    homo_key: str = "homo", lumo_key: str = "lumo", gap_key: str = "gap",
) -> torch.Tensor:
    """pred_raw: (batch, num_targets) predictions already inverse-transformed
    back to physical units. Returns 0 if homo/lumo/gap aren't all in
    target_names (so this is a safe no-op if you train on a different
    target subset)."""
    if not all(k in target_names for k in (homo_key, lumo_key, gap_key)):
        return torch.tensor(0.0, device=pred_raw.device)
    homo = pred_raw[:, target_names.index(homo_key)]
    lumo = pred_raw[:, target_names.index(lumo_key)]
    gap = pred_raw[:, target_names.index(gap_key)]
    return F.mse_loss(gap, lumo - homo)
