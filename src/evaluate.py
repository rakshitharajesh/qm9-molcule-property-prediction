"""Evaluation metrics, results-table building, and MC-dropout uncertainty -
shared across baselines, classical GNNs, and the transformer so every model
in the comparison is scored identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .predict_utils import run_inference


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    """One row per target (MAE, RMSE, R2) plus a final 'macro_avg' row.
    Never collapse this into one pooled number for the write-up - QM9
    targets are on very different scales, so a single blended metric hides
    whichever target the model is doing worst on.
    """
    rows = []
    for i, name in enumerate(target_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        rows.append({
            "target": name,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": mean_squared_error(yt, yp) ** 0.5,
            "R2": r2_score(yt, yp),
        })
    df = pd.DataFrame(rows)
    macro = {"target": "macro_avg", "MAE": df["MAE"].mean(), "RMSE": df["RMSE"].mean(), "R2": df["R2"].mean()}
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def evaluate_torch_model(model, loader, scaler, target_names, device="cpu") -> pd.DataFrame:
    y_true, y_pred = run_inference(model, loader, scaler, device)
    return per_target_metrics(y_true.numpy(), y_pred.numpy(), target_names)


def evaluate_sklearn_model(model, X, y_true: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    y_pred = model.predict(X)
    return per_target_metrics(y_true, y_pred, target_names)


def mc_dropout_predict(model, loader, scaler, device="cpu", n_samples: int = 20):
    """Runs `n_samples` stochastic forward passes (dropout kept active) and
    returns (y_true, mean_pred, std_pred) - std_pred is the per-molecule,
    per-target uncertainty estimate. Higher std = model is less confident;
    use it to build a confidence-vs-coverage curve (accuracy on the most
    confident 90%/80%/... of test molecules) as described in the project
    plan.
    """
    all_preds = []
    y_true = None
    for _ in range(n_samples):
        yt, yp = run_inference(model, loader, scaler, device, mc_dropout=True)
        y_true = yt
        all_preds.append(yp.unsqueeze(0))
    stacked = torch.cat(all_preds, dim=0)  # (n_samples, N, num_targets)
    return y_true, stacked.mean(dim=0), stacked.std(dim=0)


def coverage_accuracy_curve(y_true, mean_pred, std_pred, target_idx: int, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """For one target: sort test molecules by predicted uncertainty (std),
    keep only the most confident `fraction`, and report MAE on that subset.
    A model whose uncertainty estimate is meaningful should show MAE drop
    as fraction shrinks (abstaining on the least confident predictions
    should improve accuracy on what's left)."""
    yt = y_true[:, target_idx].numpy()
    yp = mean_pred[:, target_idx].numpy()
    unc = std_pred[:, target_idx].numpy()
    order = np.argsort(unc)  # most confident (lowest std) first
    rows = []
    n = len(yt)
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        idx = order[:k]
        rows.append({
            "coverage": frac,
            "n_molecules": k,
            "MAE": mean_absolute_error(yt[idx], yp[idx]),
        })
    return pd.DataFrame(rows)


def build_results_table(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {"model": ..., "split": "random"/"scaffold", **macro_metrics}
    Convenience for assembling the final model-comparison table from
    multiple evaluate_* calls across your experiment script."""
    return pd.DataFrame(rows)
