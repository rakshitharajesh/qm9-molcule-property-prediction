#!/usr/bin/env python3
"""
Re-evaluates the saved checkpoints in results/ and writes a detailed,
per-target breakdown that the combined table (macro-averages only) doesn't
include:

- results/per_target_metrics.csv   MAE / RMSE / R2 per model x split x target
- results/physics_consistency.csv  mean |gap - (lumo - homo)| of predictions
- results/coverage_curves.csv      MC-dropout confidence-vs-coverage (gap target)
- results/predictions_<split>.npz  raw test-set predictions for parity plots

Must be run with the same --hidden / --layers the checkpoints were trained
with (defaults match run_experiment.py). Baselines are retrained here since
sklearn models aren't checkpointed.

    python scripts/eval_checkpoints.py --csv data/qm9_full.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data import DEFAULT_TARGETS, TargetScaler, load_qm9_csv, random_split, scaffold_split
from src.evaluate import coverage_accuracy_curve, mc_dropout_predict, per_target_metrics
from src.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.baselines import build_baseline_models, featurize_smiles_list
from src.models.gnn import build_classical_gnn
from src.models.transformer import add_structural_encoding, build_transformer
from src.predict_utils import run_inference

CLASSICAL_GNNS = ["gcn", "gin", "mpnn"]
TRANSFORMERS = ["gatv2", "gps"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/qm9_full.csv")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--mc-samples", type=int, default=20)
    return p.parse_args()


def gap_violation(pred: np.ndarray, targets: list[str]) -> float:
    h, l, g = (targets.index(k) for k in ("homo", "lumo", "gap"))
    return float(np.abs(pred[:, g] - (pred[:, l] - pred[:, h])).mean())


def main():
    args = parse_args()
    targets = DEFAULT_TARGETS
    graphs = load_qm9_csv(args.csv, target_cols=targets)
    pe_dim = max(4, min(16, args.hidden // 4))

    metric_rows, phys_rows, cov_rows = [], [], []
    for split_name, splitter in [("random", random_split), ("scaffold", scaffold_split)]:
        sp = splitter(graphs)
        print(f"\n== {split_name}: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")
        scaler = TargetScaler().fit(sp.train)
        y_test = torch.cat([g.y for g in sp.test]).numpy()
        preds = {"y_true": y_test}

        if not args.skip_baselines:
            X_train = featurize_smiles_list([g.smiles for g in sp.train])
            X_test = featurize_smiles_list([g.smiles for g in sp.test])
            y_train = torch.cat([g.y for g in sp.train]).numpy()
            for name, model in build_baseline_models().items():
                print(f"  fitting {name}")
                model.fit(X_train, y_train)
                preds[name] = model.predict(X_test)

        test_pe = add_structural_encoding(sp.test, walk_length=pe_dim)
        loader = DataLoader(sp.test, batch_size=args.batch_size)
        loader_pe = DataLoader(test_pe, batch_size=args.batch_size)
        for name in CLASSICAL_GNNS + TRANSFORMERS:
            ckpt = os.path.join(args.results_dir, f"{name}_{split_name}.pt")
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt} not found")
                continue
            if name in TRANSFORMERS:
                model = build_transformer(name, ATOM_FEATURE_DIM, BOND_FEATURE_DIM, len(targets),
                                           args.hidden, args.layers, pe_dim=pe_dim)
                dl = loader_pe
            else:
                model = build_classical_gnn(name, ATOM_FEATURE_DIM, BOND_FEATURE_DIM, len(targets),
                                             args.hidden, args.layers)
                dl = loader
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.to(args.device)
            _, yp = run_inference(model, dl, scaler, args.device)
            preds[name] = yp.numpy()

            yt, mean_p, std_p = mc_dropout_predict(model, dl, scaler, args.device, args.mc_samples)
            cov = coverage_accuracy_curve(yt, mean_p, std_p, targets.index("gap"),
                                          fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1))
            cov.insert(0, "split", split_name)
            cov.insert(0, "model", name)
            cov_rows.append(cov)
            print(f"  evaluated {name}")

        for name, yp in preds.items():
            if name == "y_true":
                continue
            df = per_target_metrics(y_test, yp, targets)
            df.insert(0, "split", split_name)
            df.insert(0, "model", name)
            metric_rows.append(df)
            phys_rows.append({"model": name, "split": split_name,
                              "gap_violation_Ha": gap_violation(yp, targets)})
        phys_rows.append({"model": "ground_truth", "split": split_name,
                          "gap_violation_Ha": gap_violation(y_test, targets)})
        np.savez_compressed(os.path.join(args.results_dir, f"predictions_{split_name}.npz"), **preds)

    out = args.results_dir
    pd.concat(metric_rows).to_csv(os.path.join(out, "per_target_metrics.csv"), index=False)
    pd.DataFrame(phys_rows).to_csv(os.path.join(out, "physics_consistency.csv"), index=False)
    if cov_rows:
        pd.concat(cov_rows).to_csv(os.path.join(out, "coverage_curves.csv"), index=False)
    print("\nwrote per_target_metrics.csv, physics_consistency.csv, coverage_curves.csv")


if __name__ == "__main__":
    main()
