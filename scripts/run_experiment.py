#!/usr/bin/env python3
"""
Main experiment driver: trains every baseline, classical GNN, and graph
transformer on both a random split and a scaffold split of QM9, and writes
one combined comparison table to results/.

Usage (on your GPU machine, with the full QM9 CSV):
    python scripts/run_experiment.py --csv data/qm9.csv --epochs 150 --device cuda

Quick correctness check (a few seconds, tiny 5-row sample, CPU):
    python scripts/run_experiment.py --quick

See README.md for the full walkthrough, what each flag does, and what to
change for the physics-consistency and uncertainty ablations.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader

from src.data import DEFAULT_TARGETS, TargetScaler, load_qm9_csv, random_split, scaffold_split
from src.evaluate import build_results_table, evaluate_sklearn_model, evaluate_torch_model
from src.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.baselines import build_baseline_models, featurize_smiles_list
from src.models.gnn import build_classical_gnn
from src.models.transformer import add_structural_encoding, build_transformer
from src.train import train_torch_model

CLASSICAL_GNNS = ["gcn", "gin", "mpnn"]
TRANSFORMERS = ["gatv2", "gps"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/sample_qm9_5rows.csv",
                    help="Path to the QM9 CSV (mol_id, smiles, + target columns). "
                         "Defaults to the tiny 5-row sample for a quick pipeline check - "
                         "point this at your real ~134k-row file for the actual project run.")
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--physics-loss", action="store_true",
                    help="Add the HOMO/LUMO/gap consistency penalty during training.")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--max-rows", type=int, default=None,
                    help="Only load the first N rows - handy for a fast dry run on the real CSV.")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--quick", action="store_true",
                    help="Overrides several flags for a ~10-second sanity check: tiny sample "
                         "CSV, 3 epochs, tiny hidden size. Use this to confirm the whole "
                         "pipeline runs on your machine before committing to a real run.")
    args = p.parse_args()
    if args.quick:
        args.csv = "data/sample_qm9_5rows.csv"
        args.epochs = 3
        args.patience = 3
        args.hidden = 16
        args.layers = 2
        args.batch_size = 2
    return args


def run_split(name: str, graphs, args, target_names) -> list[dict]:
    """Trains and evaluates every model on one split (either 'random' or
    'scaffold') and returns a list of result rows for the combined table."""
    print(f"\n{'=' * 70}\n{name.upper()} SPLIT  (train={len(graphs.train)}, "
          f"val={len(graphs.val)}, test={len(graphs.test)})\n{'=' * 70}")

    rows = []
    scaler = TargetScaler().fit(graphs.train)

    # ---- baselines (non-graph) ----
    if not args.skip_baselines:
        X_train = featurize_smiles_list([g.smiles for g in graphs.train])
        X_test = featurize_smiles_list([g.smiles for g in graphs.test])
        y_train = torch.cat([g.y for g in graphs.train]).numpy()
        y_test = torch.cat([g.y for g in graphs.test]).numpy()
        for model_name, model in build_baseline_models().items():
            print(f"\n--- baseline: {model_name} ({name} split) ---")
            model.fit(X_train, y_train)
            df = evaluate_sklearn_model(model, X_test, y_test, target_names)
            macro = df.iloc[-1]
            print(df.to_string(index=False))
            rows.append({"model": model_name, "family": "baseline", "split": name,
                         "MAE": macro["MAE"], "RMSE": macro["RMSE"], "R2": macro["R2"]})

    # ---- classical GNNs + transformers (share the same torch loop) ----
    train_loader = DataLoader(graphs.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(graphs.val, batch_size=args.batch_size)
    test_loader = DataLoader(graphs.test, batch_size=args.batch_size)

    torch_model_names = list(CLASSICAL_GNNS)
    if not args.skip_transformers:
        torch_model_names += TRANSFORMERS

    # Transformers need the random-walk positional encoding attached to
    # every graph before batching - classical GNNs ignore the extra
    # attribute, so it's safe to add it once for everyone.
    if not args.skip_transformers:
        # Positional-encoding width must leave room for real node features
        # inside GraphGPS's hidden_channels - scale it down for small
        # --hidden values (e.g. --quick) instead of hardcoding 16.
        pe_dim = max(4, min(16, args.hidden // 4))
        graphs_pe = graphs.__class__(
            train=add_structural_encoding(graphs.train, walk_length=pe_dim),
            val=add_structural_encoding(graphs.val, walk_length=pe_dim),
            test=add_structural_encoding(graphs.test, walk_length=pe_dim),
        )
        train_loader_pe = DataLoader(graphs_pe.train, batch_size=args.batch_size, shuffle=True)
        val_loader_pe = DataLoader(graphs_pe.val, batch_size=args.batch_size)
        test_loader_pe = DataLoader(graphs_pe.test, batch_size=args.batch_size)

    for model_name in torch_model_names:
        print(f"\n--- {model_name} ({name} split) ---")
        is_transformer = model_name in TRANSFORMERS
        tl, vl, tel = (train_loader_pe, val_loader_pe, test_loader_pe) if is_transformer \
            else (train_loader, val_loader, test_loader)

        if is_transformer:
            model = build_transformer(model_name, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                                       len(target_names), args.hidden, args.layers,
                                       pe_dim=pe_dim)
        else:
            model = build_classical_gnn(model_name, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                                         len(target_names), args.hidden, args.layers)

        n_params = sum(p.numel() for p in model.parameters())
        ckpt_path = os.path.join(args.out_dir, f"{model_name}_{name}.pt")
        history = train_torch_model(
            model, tl, vl, scaler, target_names, device=args.device,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            use_physics_loss=args.physics_loss, checkpoint_path=ckpt_path,
        )
        df = evaluate_torch_model(model, tel, scaler, target_names, device=args.device)
        macro = df.iloc[-1]
        print(df.to_string(index=False))
        rows.append({
            "model": model_name, "family": "transformer" if is_transformer else "gnn",
            "split": name, "MAE": macro["MAE"], "RMSE": macro["RMSE"], "R2": macro["R2"],
            "params": n_params, "best_val_mae": history["best_val_mae"],
        })

    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    graphs = load_qm9_csv(args.csv, target_cols=args.targets, max_rows=args.max_rows)
    if len(graphs) < 10:
        print(f"\n[!] Only {len(graphs)} molecules loaded - this looks like the tiny sample "
              f"file, not your full QM9 CSV. Metrics below are meaningless (e.g. R2 needs "
              f">=2 test samples); this run just proves the pipeline executes end to end. "
              f"Point --csv at your real ~134k-row file for the actual project results.\n")

    random_splits = random_split(graphs)
    scaffold_splits = scaffold_split(graphs)

    all_rows = []
    all_rows += run_split("random", random_splits, args, args.targets)
    all_rows += run_split("scaffold", scaffold_splits, args, args.targets)

    results_df = build_results_table(all_rows)
    out_path = os.path.join(args.out_dir, "combined_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 70}\nFULL COMPARISON TABLE  (saved to {out_path})\n{'=' * 70}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
