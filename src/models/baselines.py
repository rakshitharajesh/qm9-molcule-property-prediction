"""
Non-graph baselines: RDKit descriptors + Morgan fingerprints -> classical
regressors. These exist to prove the GNNs/transformer are actually earning
their extra complexity - report them in the same results table as the graph
models, not as a footnote.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# A compact, chemically-meaningful descriptor set. Deliberately excludes
# Descriptors that are near-duplicates of each other to keep collinearity
# manageable for the linear baseline.
DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "RingCount": Descriptors.RingCount,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "MolLogP": Descriptors.MolLogP,
}


def featurize_smiles_list(
    smiles_list: list[str], radius: int = 2, n_bits: int = 1024
) -> np.ndarray:
    """Descriptors + Morgan (ECFP) fingerprint, concatenated, for each SMILES.

    Rows for unparsable SMILES are zero-filled rather than dropped, so the
    output always has the same length/order as the input list - callers
    that also build graphs from the same SMILES list can align by index.
    """
    n_desc = len(DESCRIPTOR_FUNCS)
    feats = np.zeros((len(smiles_list), n_desc + n_bits), dtype=np.float32)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc = [f(mol) for f in DESCRIPTOR_FUNCS.values()]
        fp = fp_gen.GetFingerprintAsNumPy(mol)
        feats[i, :n_desc] = desc
        feats[i, n_desc:] = fp.astype(np.float32)
    return feats


def build_baseline_models(seed: int = 42) -> dict:
    """Returns a dict of un-fit sklearn-API regressors, each handling all
    targets at once (multi-output regression)."""
    return {
        "ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed
        ),  # natively supports multi-output
        "xgboost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            )
        ),
    }
