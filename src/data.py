"""
Dataset loading, splitting, and target scaling for the QM9 multi-property
regression project.

Primary path (this file): read a QM9 CSV (mol_id, smiles, mu, alpha, homo,
lumo, gap, ...) - see scripts/xyz_to_csv.py - and build graphs with RDKit
via featurize.py.

Alternate path (not used by default): torch_geometric.datasets.QM9
auto-downloads a version of QM9 that also carries 3D atomic coordinates,
which is what you'd want if you later add a 3D-aware model (SchNet/DimeNet).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

from .featurize import mol_to_graph

# Silence RDKit's C++ warnings (invalid valence etc.) - we handle failures
# ourselves by skipping unparsable rows.
RDLogger.DisableLog("rdApp.*")

DEFAULT_TARGETS = ["mu", "alpha", "homo", "lumo", "gap"]


def load_qm9_csv(
    csv_path: str,
    smiles_col: str = "smiles",
    target_cols: list[str] = DEFAULT_TARGETS,
    add_hs: bool = True,
    max_rows: int | None = None,
) -> list[Data]:
    """Load a QM9-style CSV and convert every row into a PyG Data graph.

    Rows RDKit can't parse are skipped (and counted) rather than crashing
    the whole load - with the real QM9 CSV this should be ~0 rows, but it's
    a cheap safety net.
    """
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    missing = [c for c in [smiles_col] + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    graphs = []
    n_failed = 0
    for _, row in df.iterrows():
        g = mol_to_graph(row[smiles_col], add_hs=add_hs)
        if g is None:
            n_failed += 1
            continue
        g.y = torch.tensor([[row[c] for c in target_cols]], dtype=torch.float)
        g.scaffold = get_scaffold(row[smiles_col])
        graphs.append(g)

    if n_failed:
        print(f"[load_qm9_csv] skipped {n_failed}/{len(df)} rows RDKit could not parse")
    print(f"[load_qm9_csv] loaded {len(graphs)} graphs, targets={target_cols}")
    return graphs


def get_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES, used to group structurally-related
    molecules for scaffold splitting. Falls back to the original SMILES if
    scaffold extraction fails (e.g. acyclic molecules get an empty scaffold,
    which is fine - they'll just all share the "" scaffold bucket).
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return smiles


@dataclass
class Splits:
    train: list[Data]
    val: list[Data]
    test: list[Data]


def _ensure_nonempty(train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> None:
    """Safety net for tiny datasets (e.g. a smoke-test sample of a handful
    of molecules) where int(frac * n) rounds a split down to zero. At real
    QM9 scale (~134k rows) this never triggers - frac_val/frac_test alone
    already give thousands of molecules. Mutates the lists in place, moving
    from the end of train_idx (as a scaffold_split caller, note this can
    split a scaffold group across splits in this tiny-N degenerate case
    only - never at real dataset sizes)."""
    for target in (val_idx, test_idx):
        if len(target) == 0 and len(train_idx) > 1:
            target.append(train_idx.pop())


def random_split(
    graphs: list[Data], frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42
) -> Splits:
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    idx = list(range(len(graphs)))
    random.Random(seed).shuffle(idx)
    n_train = int(frac_train * len(idx))
    n_val = int(frac_val * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    _ensure_nonempty(train_idx, val_idx, test_idx)
    return Splits(
        train=[graphs[i] for i in train_idx],
        val=[graphs[i] for i in val_idx],
        test=[graphs[i] for i in test_idx],
    )


def scaffold_split(
    graphs: list[Data], frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42
) -> Splits:
    """Standard scaffold split (as used in MoleculeNet/DeepChem): group
    molecules by Bemis-Murcko scaffold, then greedily fill train with the
    biggest scaffold groups first, then val, then test.

    This means every molecule sharing a scaffold ends up in the same split,
    so the test set contains structurally novel molecules the model has
    never seen the "core" of - a much harder and more honest generalization
    test than a random split. Expect meaningfully worse metrics here than
    under random_split; report both, that gap is itself a project finding.
    """
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(graphs):
        scaffold = getattr(g, "scaffold", None) or get_scaffold(g.smiles)
        scaffold_to_indices[scaffold].append(i)

    # Sort groups deterministically but shuffle within same-size groups so
    # the split isn't sensitive to CSV row order.
    groups = list(scaffold_to_indices.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n = len(graphs)
    n_train_target = frac_train * n
    n_val_target = frac_val * n

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) + len(group) <= n_train_target or not train_idx:
            train_idx += group
        elif len(val_idx) + len(group) <= n_val_target or not val_idx:
            val_idx += group
        else:
            test_idx += group

    _ensure_nonempty(train_idx, val_idx, test_idx)

    return Splits(
        train=[graphs[i] for i in train_idx],
        val=[graphs[i] for i in val_idx],
        test=[graphs[i] for i in test_idx],
    )


class TargetScaler:
    """Per-target standardization (zero mean, unit variance), fit on the
    training split only. QM9 targets live on very different scales (e.g.
    homo/lumo/gap are ~0.1-0.5 Hartree, alpha is ~10-100), so training
    without this will let the loss be dominated by whichever target has the
    largest raw magnitude.
    """

    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, graphs: list[Data]):
        y = torch.cat([g.y for g in graphs], dim=0)
        self.mean = y.mean(dim=0, keepdim=True)
        self.std = y.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std.to(y.device) + self.mean.to(y.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, sd):
        self.mean, self.std = sd["mean"], sd["std"]
