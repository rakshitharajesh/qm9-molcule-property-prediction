# Project Context and Design Notes

[`README.md`](./README.md) covers how to run the code, and
[`RESULTS.md`](./RESULTS.md) covers what the experiments found. This file
explains **why the project is shaped the way it is**, meaning the decisions
behind the pipeline and what to keep in mind when extending it.

## 1. Problem statement

Quantum-chemical property calculation (e.g. DFT) is accurate but
computationally expensive. This project asks whether machine learning can
estimate molecular properties directly from structure, fast enough to screen
candidate molecules before running expensive calculations.

- **Input:** molecular structure as a SMILES string, converted to a graph.
- **Output:** several quantum-chemical properties, predicted as continuous values.
- **Approach:** compare classical molecular-descriptor models, classical graph
  neural networks, and graph transformers under identical conditions.
- **Task type:** multi-output regression.

## 2. Targets

Five linked properties, predicted jointly by one shared encoder (not five
separate models):

| Target | Meaning | Unit | Why it matters |
|---|---|---|---|
| `mu` | Dipole moment | Debye | Charge separation |
| `alpha` | Isotropic polarizability | bohr³ | How easily the electron cloud distorts |
| `homo` | Highest occupied molecular orbital energy | Hartree | Electron donation / reactivity |
| `lumo` | Lowest unoccupied molecular orbital energy | Hartree | Electron acceptance |
| `gap` | `lumo − homo` | Hartree | Electronic stability / excitation energy |

The `gap = lumo − homo` relationship holds only in **raw physical units**.
Each target is z-scored independently for training, which breaks the linear
relationship. `src/physics_loss.py` and `src/train.py` therefore
inverse-transform predictions before computing the consistency penalty. Don't
"simplify" this by computing the loss on scaled predictions.

## 3. Dataset

QM9 (Ramakrishnan et al., 2014): about 134k small organic molecules
containing only H, C, N, O and F, each with DFT-computed properties.
`scripts/xyz_to_csv.py` converts the raw per-molecule `.xyz` files into a flat
CSV (`mol_id, smiles, mu, alpha, homo, lumo, gap`). It uses the
DFT-relaxed SMILES, because that is the structure the labels were computed
for. About 0.6% of rows fail RDKit parsing and are skipped, leaving 133,139
molecules.

`data/sample_qm9_5rows.csv` holds 5 real rows for smoke-testing only. Its
metrics are meaningless.

## 4. Research angles

This is not just "apply GNNs to QM9". The project covers five specific angles:

1. **Multi-property learning.** One shared graph encoder predicts all five
   targets together.
2. **Physics consistency.** Predicted HOMO/LUMO/gap should satisfy
   `gap ≈ lumo − homo`, optionally enforced with a training-time penalty.
3. **Scaffold generalization.** Models are evaluated on structurally novel
   molecules (a Bemis-Murcko scaffold split) as well as on a random split.
4. **Uncertainty awareness.** MC-dropout gives each prediction a confidence
   score, so low-confidence molecules can be flagged.
5. **Classical GNNs vs. graph transformers.** A controlled comparison where
   only the encoder changes.

## 5. Why the model ladder looks like this

A plain GNN (GCN, GIN, MPNN) lets each atom see only its bonded neighbours in
each layer, so information from a distant atom takes many layers to arrive
(over-squashing). A graph transformer lets every atom attend to every other
atom in one layer. But graphs have no inherent node ordering, so the model
needs a **structural/positional encoding** (here a random-walk PE) to tell
graph structure apart from an unordered bag of atoms. Skipping that encoding
is the most common reason a from-scratch graph transformer loses to a plain
GCN. That is why `GraphGPSRegressor` builds it in through PyG's `GPSConv` and
`AddRandomWalkPE`.

The five neural models form a deliberate ladder. Each step answers "what does
the next mechanism buy us?":

1. **GCN**: fixed, degree-normalized neighbour averaging with no bond-type
   awareness, a real limitation for molecules.
2. **GIN (GINEConv)**: sum aggregation plus an MLP update (more expressive
   than mean aggregation), and it uses bond features.
3. **MPNN (NNConv + GRU)**: edge-conditioned convolution from Gilmer et al.
   (2017). The most expressive classical option, with the most parameters.
4. **GATv2**: learned attention over bonded neighbours only. A stepping
   stone, not yet a graph transformer.
5. **GraphGPS**: local MPNN plus global multi-head self-attention plus a
   random-walk structural encoding.

All five share one training loop, one evaluation module, and one
pooling/head design (`src/models/common.py`). This keeps the comparison from
being confounded by one model getting a better training recipe than another.

## 6. Implementation notes and gotchas

- **Feature dimensions are derived, not hand-counted.** `_one_hot()` always
  allocates `len(choices) + 1` slots (the extra one is an "other" bucket).
  `ATOM_FEATURE_DIM = 28` and `BOND_FEATURE_DIM = 7` are computed from the
  same choice lists used at featurization time. If you add features, follow
  that pattern.
- **Tiny-dataset guard.** On very small inputs, `int(frac * n)` can round a
  split down to zero molecules. `_ensure_nonempty()` in `src/data.py` handles
  this and does nothing at real QM9 scale.
- **Scaffold split sizes.** The greedy fill (largest scaffold groups first)
  uses QM9's many single-molecule scaffolds to hit the 80/10/10 targets
  almost exactly, while still keeping every scaffold in one split.
- **Positional-encoding width.** GraphGPS reserves `pe_dim` channels of the
  hidden size for the random-walk PE. `run_experiment.py` scales `pe_dim`
  with `--hidden`, so small models still work.

## 7. Possible extensions

- **3D-aware models** (SchNet, DimeNet++, PaiNN). QM9 labels come from 3D
  geometries that the 2D-graph models never see.
- **Explainability** (GNNExplainer or RDKit atom-attribution maps).
- **Deep ensembles** as a better-calibrated alternative to MC-dropout.
- **Physics-loss ablation.** Compare `--physics-loss` on vs. off for
  consistency and accuracy.
- **Data-efficiency curves.** Retrain with `--max-rows` at several sizes and
  plot MAE against training-set size for each model family.
- **Multiple seeds and tuning per model**, to put error bars on the comparison.
