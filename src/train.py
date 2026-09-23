"""One shared training loop for every torch model (GCN/GIN/MPNN/GATv2/GPS).
Using the same loop for all of them is what makes "compare classical GNNs
vs graph transformers" a fair comparison rather than an artifact of one
model getting a better training recipe than another.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from .data import TargetScaler
from .physics_loss import gap_consistency_loss
from .predict_utils import mean_absolute_error, run_inference


def train_torch_model(
    model,
    train_loader,
    val_loader,
    scaler: TargetScaler,
    target_names: list[str],
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    use_physics_loss: bool = False,
    physics_loss_weight: float = 0.1,
    checkpoint_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Trains `model` in place (restoring the best-val-MAE weights at the
    end) and returns a history dict for plotting. Early stopping and LR
    reduction are both driven by mean validation MAE in raw physical units
    (not the scaled training loss), since that's the number you'll actually
    report.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, patience // 3)
    )

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_mae": [], "lr": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_scaled = model(batch)
            y_scaled = scaler.transform(batch.y.to(device))
            loss = F.mse_loss(pred_scaled, y_scaled)

            if use_physics_loss:
                pred_raw = scaler.inverse_transform(pred_scaled)
                phys = gap_consistency_loss(pred_raw, target_names)
                loss = loss + physics_loss_weight * phys

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_seen += batch.num_graphs
        train_loss = total_loss / max(n_seen, 1)

        y_true, y_pred = run_inference(model, val_loader, scaler, device)
        val_mae = mean_absolute_error(y_true, y_pred)
        scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or improved):
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val_MAE {val_mae:.4f} | best {best_val_mae:.4f} | "
                f"{time.time() - t0:.1f}s{'  *' if improved else ''}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_mae"] = best_val_mae
    return history
