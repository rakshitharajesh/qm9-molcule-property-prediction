"""
SMILES -> molecular graph featurization for QM9.

QM9 molecules contain only H, C, N, O, F, so the atom feature vocabulary
below is deliberately small and QM9-specific (with a safety "other" bucket
in case you later point this at a different dataset).

Design note: hydrogens are made explicit (Chem.AddHs) so that the graph
matches what QM9's 3D structures actually contain (every atom, including
H). This roughly doubles node count for typical QM9 molecules (up to ~29
heavy atoms -> up to ~29 heavy + attached H). If you want a lighter/faster
graph, set `add_hs=False` in mol_to_graph and rely on the num_hs feature
below to encode implicit hydrogens instead - this is a legitimate ablation
to report (does explicit-H graph help vs. hurt).
"""
from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_LIST = ["H", "C", "N", "O", "F"]  # QM9 vocabulary; "other" is the safety bucket
NUM_ATOM_TYPES = len(ATOM_LIST) + 1

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]  # + "other"

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def _one_hot(value, choices) -> list[float]:
    # length is always len(choices) + 1: one slot per listed choice, plus a
    # trailing "other" bucket for anything not in the list.
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)  # "other" bucket
    vec[idx] = 1.0
    return vec


# Feature dims are derived from the same _one_hot logic used at featurization
# time (rather than hand-counted) so they can never silently drift out of
# sync with atom_features()/bond_features() below.
_DEGREE_CHOICES = [0, 1, 2, 3, 4]
_CHARGE_CHOICES = [-1, 0, 1]
_NUM_H_CHOICES = [0, 1, 2, 3, 4]

ATOM_FEATURE_DIM = (
    (len(ATOM_LIST) + 1)          # one-hot atomic symbol (+ other)
    + (len(_DEGREE_CHOICES) + 1)  # degree 0-4 one-hot (+ other, unused)
    + (len(_CHARGE_CHOICES) + 1)  # formal charge {-1, 0, 1} one-hot (+ other)
    + (len(HYBRIDIZATIONS) + 1)   # hybridization one-hot (+ other)
    + 1                           # is aromatic
    + 1                           # is in ring
    + (len(_NUM_H_CHOICES) + 1)   # total attached H count 0-4 one-hot (+ other)
)  # = 28

BOND_FEATURE_DIM = (len(BOND_TYPES) + 1) + 2  # bond type one-hot (+other) + conjugated + in-ring = 7


def atom_features(atom: Chem.Atom) -> list[float]:
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_LIST)
    feats += _one_hot(min(atom.GetDegree(), 4), _DEGREE_CHOICES)
    charge = atom.GetFormalCharge()
    feats += _one_hot(charge if charge in (-1, 0, 1) else "other", _CHARGE_CHOICES)
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats += _one_hot(min(atom.GetTotalNumHs(includeNeighbors=True), 4), _NUM_H_CHOICES)
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    feats = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats

# adding H atoms - which are left implicit in SMILE strings
# 3d molecule must also represent the H atoms
def mol_from_smiles(smiles: str, add_hs: bool = True) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol


def mol_to_graph(smiles: str, add_hs: bool = True) -> Data | None:
    """Convert a SMILES string into a PyG Data object with x, edge_index, edge_attr.

    Returns None if RDKit fails to parse the SMILES (log and skip these -
    with QM9 this should be essentially never, but real-world SMILES columns
    can contain a handful of bad rows).
    """
    mol = mol_from_smiles(smiles, add_hs=add_hs)
    if mol is None:
        return None

    xs = [atom_features(atom) for atom in mol.GetAtoms()]
    if len(xs) == 0:
        return None
    x = torch.tensor(xs, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        # single-atom molecule (e.g. methane's carbon with no bonds after
        # some edge case) - give it an empty-but-correctly-shaped edge set
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
