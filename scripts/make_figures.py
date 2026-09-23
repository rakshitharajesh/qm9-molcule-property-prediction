#!/usr/bin/env python3
"""
Generates every chart and diagram in docs/figures/ from the CSVs in results/.

    python scripts/run_experiment.py ...      # -> results/combined_results.csv
    python scripts/eval_checkpoints.py ...    # -> per-target / physics / coverage CSVs
    python scripts/make_figures.py

Charts that depend on a missing CSV are skipped with a message, so this
works with just combined_results.csv too.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")

# Palette: fixed categorical order (validated colorblind-safe for 3 series,
# adjacent-safe up to 8), recessive neutrals for text/grid.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FAMILY_COLOR = {"baseline": BLUE, "gnn": ORANGE, "transformer": AQUA}
FAMILY_LABEL = {"baseline": "Descriptor baselines", "gnn": "Classical GNNs",
                "transformer": "Attention / graph transformers"}
SPLIT_COLOR = {"random": BLUE, "scaffold": ORANGE}
MODEL_ORDER = ["ridge", "random_forest", "xgboost", "gcn", "gin", "mpnn", "gatv2", "gps"]
MODEL_LABEL = {"ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost",
               "gcn": "GCN", "gin": "GIN", "mpnn": "MPNN", "gatv2": "GATv2", "gps": "GraphGPS"}
MODEL_FAMILY = {"ridge": "baseline", "random_forest": "baseline", "xgboost": "baseline",
                "gcn": "gnn", "gin": "gnn", "mpnn": "gnn", "gatv2": "transformer", "gps": "transformer"}
HARTREE_TO_EV = 27.2114
HEATMAP_LABEL = {"mu": "μ\n(D)", "alpha": "α\n(bohr³)", "homo": "HOMO\n(eV)",
                 "lumo": "LUMO\n(eV)", "gap": "Gap\n(eV)"}
TARGET_LABEL = {"mu": "μ (D)", "alpha": "α (a₀³)", "homo": "HOMO (Ha)",
                "lumo": "LUMO (Ha)", "gap": "Gap (Ha)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "figure.dpi": 100,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------- charts

def chart_split_comparison(df):
    """Grouped horizontal bars: macro MAE and R2 on random vs scaffold split."""
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    y = np.arange(len(models))[::-1]
    h = 0.36
    for ax, metric, better in [(axes[0], "MAE", "lower is better"), (axes[1], "R2", "higher is better")]:
        for k, split in enumerate(["random", "scaffold"]):
            sub = df[df.split == split].set_index("model").reindex(models)
            vals = sub[metric].values
            ax.barh(y + (h / 2 if k == 0 else -h / 2), vals, height=h - 0.04,
                    color=SPLIT_COLOR[split], label=f"{split.capitalize()} split")
            for yi, v in zip(y + (h / 2 if k == 0 else -h / 2), vals):
                ax.text(v + (0.01 if metric == "MAE" else 0.008), yi, f"{v:.2f}", va="center",
                        fontsize=9, color=INK2)
        ax.set_title(f"Macro-averaged {metric.replace('R2', 'R²')}  ({better})", pad=22)
        ax.grid(axis="y", visible=False)
        if metric == "R2":
            ax.set_xlim(0, 1.08)
        else:
            ax.set_xlim(0, df.MAE.max() * 1.15)
    axes[0].set_yticks(y, [MODEL_LABEL[m] for m in models])
    for yi, m in zip(y, models):
        axes[0].get_yticklabels()[list(y).index(yi)].set_color(INK)
    # family separators
    for ax in axes:
        for boundary in (4.5, 1.5):
            ax.axhline(boundary, color=GRID, lw=1)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, -0.2), ncol=2)
    fig.text(0.01, -0.02, "Macro average across μ, α, HOMO, LUMO, gap in raw units - dominated by α and μ; "
             "see per-target figure for the orbital energies.", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "01_random_vs_scaffold.png")


def chart_generalization_gap(df):
    """How much each model's MAE grows when tested on unseen scaffolds."""
    piv = df.pivot(index="model", columns="split", values="MAE")
    models = [m for m in MODEL_ORDER if m in piv.index]
    pct = ((piv.scaffold - piv.random) / piv.random * 100).reindex(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    colors = [FAMILY_COLOR[MODEL_FAMILY[m]] for m in models]
    ax.bar(x, pct.values, width=0.6, color=colors)
    ax.axhline(0, color=MUTED, lw=1)
    for xi, v in zip(x, pct.values):
        ax.text(xi, v + (3 if v >= 0 else -3), f"{v:+.0f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
    ax.set_xticks(x, [MODEL_LABEL[m].replace(" ", "\n") for m in models])
    ax.set_ylabel("MAE change, random → scaffold (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Generalization gap: error increase on unseen molecular scaffolds", pad=26)
    subtitle(ax, "Higher bar = model relied more on having seen similar scaffolds during training")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles, FAMILY_LABEL.values(), loc="upper left", ncol=3, bbox_to_anchor=(0, -0.1))
    lo = min(0, pct.min()) - 12
    ax.set_ylim(lo, pct.max() + 15)
    save(fig, "02_generalization_gap.png")


def chart_params_vs_mae(df):
    torch_df = df[df.params.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split in zip(axes, ["random", "scaffold"]):
        sub = torch_df[torch_df.split == split]
        base = df[(df.split == split) & (df.model == "xgboost")]
        for _, r in sub.iterrows():
            ax.scatter(r.params, r.MAE, s=120, color=FAMILY_COLOR[r.family],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(MODEL_LABEL[r.model], (r.params, r.MAE), xytext=(8, 6),
                        textcoords="offset points", fontsize=10, color=INK)
        if len(base):
            ax.axhline(base.MAE.iloc[0], color=BLUE, lw=1.5, ls=(0, (4, 3)))
            ax.text(0.99, base.MAE.iloc[0], "XGBoost baseline ", transform=ax.get_yaxis_transform(),
                    color=INK2, fontsize=9, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_title(f"{split.capitalize()} split", pad=10)
    axes[0].set_ylabel("Test MAE (macro avg)")
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=10, color=FAMILY_COLOR[f])
               for f in ("gnn", "transformer")]
    axes[0].legend(handles, [FAMILY_LABEL["gnn"], FAMILY_LABEL["transformer"]], loc="upper left",
                   bbox_to_anchor=(0, -0.14), ncol=2)
    fig.suptitle("Model size vs. accuracy - GraphGPS matches MPNN with ~5x fewer parameters",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "03_params_vs_mae.png")


def chart_per_target_heatmap(pt):
    targets = ["mu", "alpha", "homo", "lumo", "gap"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ[1:])
    for ax, split in zip(axes, ["random", "scaffold"]):
        sub = pt[(pt.split == split) & (pt.target.isin(targets))]
        piv = sub.pivot(index="model", columns="target", values="MAE")
        models = [m for m in MODEL_ORDER if m in piv.index]
        piv = piv.reindex(index=models, columns=targets)
        rel = piv / piv.min(axis=0)  # 1.0 = best model for that target
        im = ax.imshow(rel.values, cmap=cmap, vmin=1, vmax=min(3.0, rel.values.max()), aspect="auto")
        for i in range(rel.shape[0]):
            for j in range(rel.shape[1]):
                v = rel.values[i, j]
                raw = piv.values[i, j]
                if targets[j] in ("homo", "lumo", "gap"):
                    raw = raw * HARTREE_TO_EV  # display orbital energies in eV
                txt = f"{raw:.2f}" if targets[j] in ("mu", "alpha") else f"{raw:.3f}"
                dark = v > 1.9
                ax.text(j, i, txt + ("\n★" if v == 1 else ""), ha="center", va="center", fontsize=9,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(targets)), [HEATMAP_LABEL[t] for t in targets])
        ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
        ax.tick_params(axis="y", labelleft=(split == "random"))
        ax.grid(False)
        ax.set_title(f"{split.capitalize()} split", pad=10)
        for b in (2.5, 5.5):
            ax.axhline(b, color=SURFACE, lw=3)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("MAE relative to best model on that property (1.0 = best)", color=INK2)
    cbar.outline.set_visible(False)
    fig.suptitle("Per-property test MAE  ·  darker = worse relative to the best model  ·  ★ = best",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    save(fig, "04_per_target_mae_heatmap.png")


def chart_parity(split="random"):
    path = os.path.join(RES, f"predictions_{split}.npz")
    if not os.path.exists(path):
        print(f"[skip] predictions_{split}.npz not found")
        return
    d = np.load(path)
    targets = ["mu", "alpha", "homo", "lumo", "gap"]
    gi = targets.index("gap")
    picks = [m for m in ("xgboost", "mpnn", "gps") if m in d.files]
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5), sharex=True, sharey=True)
    yt = d["y_true"][:, gi] * 27.2114  # Hartree -> eV for readability
    lo, hi = np.percentile(yt, [0.2, 99.8])
    for ax, m in zip(np.atleast_1d(axes), picks):
        yp = d[m][:, gi] * 27.2114
        ax.hexbin(yt, yp, gridsize=70, extent=(lo, hi, lo, hi), mincnt=1, bins="log",
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ[1:]))
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        mae = np.abs(yp - yt).mean()
        ss_res = ((yp - yt) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
        ax.set_title(MODEL_LABEL[m], pad=8)
        ax.text(0.04, 0.96, f"MAE {mae:.2f} eV\nR² {1 - ss_res / ss_tot:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, color=INK)
        ax.set_xlabel("DFT HOMO-LUMO gap (eV)")
        ax.set_aspect("equal")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Predicted gap (eV)")
    fig.suptitle(f"Predicted vs. true HOMO-LUMO gap - best of each family ({split} split test set)",
                 x=0.01, y=1.04, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, f"05_parity_gap_{split}.png")


def chart_physics(phys):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    models = [m for m in MODEL_ORDER if m in set(phys.model)]
    x = np.arange(len(models))
    h = 0.36
    for k, split in enumerate(["random", "scaffold"]):
        sub = phys[phys.split == split].set_index("model").reindex(models)
        v = sub.gap_violation_Ha.values * 1000 * 27.2114  # meV
        ax.bar(x + (k - 0.5) * h, v, width=h - 0.04, color=SPLIT_COLOR[split],
               label=f"{split.capitalize()} split")
    ax.set_xticks(x, [MODEL_LABEL[m].replace(" ", "\n") for m in models])
    ax.set_yscale("log")
    ax.set_ylabel("Mean |gap − (LUMO − HOMO)|  (meV, log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Physics consistency: how far predictions break gap = LUMO − HOMO", pad=26)
    subtitle(ax, "No physics-loss penalty used. Lower = more self-consistent. XGBoost fits each target independently.")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    save(fig, "06_physics_consistency.png")


def chart_coverage(cov):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    order = [m for m in ["gcn", "gin", "mpnn", "gatv2", "gps"] if m in set(cov.model)]
    colors = dict(zip(["gcn", "gin", "mpnn", "gatv2", "gps"], [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
    for ax, split in zip(axes, ["random", "scaffold"]):
        for m in order:
            sub = cov[(cov.split == split) & (cov.model == m)].sort_values("coverage")
            mae_ev = sub.MAE.values * 27.2114
            ax.plot(sub.coverage * 100, mae_ev, color=colors[m], lw=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABEL[m])
        ax.invert_xaxis()
        ax.set_xlabel("Coverage: % of test molecules kept (most confident first)")
        ax.set_title(f"{split.capitalize()} split", pad=10)
    axes[0].set_ylabel("Gap MAE on kept molecules (eV)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=5)
    fig.suptitle("MC-dropout uncertainty: dropping the least-confident molecules lowers gap error",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "07_uncertainty_coverage.png")


# ------------------------------------------------------------------- diagrams

def _box(ax, x, y, w, h, text, fc, ec=None, fontsize=11, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or fc, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=color, weight=weight, wrap=True)


def _arrow(ax, p, q, color=MUTED, style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.0))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.0); ax.axis("off")
    light = {"blue": "#e3eefb", "orange": "#fde7dc", "aqua": "#dcf3ea", "gray": "#f0efec"}
    _box(ax, 0.2, 2.0, 2.0, 1.2, "QM9\n~134k molecules\nSMILES + DFT labels", light["gray"], fontsize=10)
    _box(ax, 2.8, 2.0, 2.2, 1.2, "RDKit featurize\natoms → 28-d nodes\nbonds → 7-d edges", light["gray"], fontsize=10)
    _box(ax, 5.6, 3.1, 2.0, 0.9, "Random split\n80 / 10 / 10", light["gray"], fontsize=10)
    _box(ax, 5.6, 1.2, 2.0, 0.9, "Scaffold split\n(unseen ring systems)", light["gray"], fontsize=10)
    _arrow(ax, (2.2, 2.6), (2.8, 2.6))
    _arrow(ax, (5.0, 2.8), (5.6, 3.5)); _arrow(ax, (5.0, 2.4), (5.6, 1.65))
    # model families
    _box(ax, 8.3, 3.7, 3.2, 1.1, "Descriptor baselines\nFingerprints + descriptors →\nRidge · Random Forest · XGBoost",
         light["blue"], ec=BLUE, fontsize=9.5)
    _box(ax, 8.3, 2.05, 3.2, 1.1, "Classical GNNs\nlocal message passing →\nGCN · GIN · MPNN",
         light["orange"], ec=ORANGE, fontsize=9.5)
    _box(ax, 8.3, 0.4, 3.2, 1.1, "Attention / graph transformers\nGATv2 (local) ·\nGraphGPS (local + global + RWPE)",
         light["aqua"], ec=AQUA, fontsize=9.5)
    for yy in (4.25, 2.6, 0.95):
        _arrow(ax, (7.6, 2.6), (8.3, yy))
    _box(ax, 12.2, 1.6, 2.6, 2.0, "Shared evaluation\nper-target MAE · RMSE · R²\nscaffold generalization\n"
         "physics consistency\nMC-dropout uncertainty", light["gray"], fontsize=9.5)
    for yy in (4.25, 2.6, 0.95):
        _arrow(ax, (11.5, yy), (12.2, 2.6))
    ax.text(0.2, 5.7, "Project pipeline", fontsize=15, weight="bold", color=INK)
    ax.text(0.2, 5.3, "One multi-task head predicts μ, α, HOMO, LUMO and gap together; "
            "every torch model shares the same training loop.", fontsize=10.5, color=INK2)
    save(fig, "10_pipeline_diagram.png")


def _molecule(ax, cx, cy, s=1.0):
    """A small 7-atom toy molecular graph; returns node positions and edges."""
    pts = np.array([[0, 0], [1, 0.55], [2, 0], [2, -1.1], [1, -1.65], [0, -1.1], [3.0, 0.55]]) * 0.55 * s
    pts = pts - pts.mean(axis=0) + np.array([cx, cy])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (2, 6)]
    return pts, edges


def _draw_mol(ax, pts, edges, node_colors, edge_color=MUTED, lw=2):
    for i, j in edges:
        ax.plot(*zip(pts[i], pts[j]), color=edge_color, lw=lw, zorder=1)
    for p, c in zip(pts, node_colors):
        ax.add_patch(Circle(p, 0.13, fc=c, ec=SURFACE, lw=2, zorder=3))


def diagram_how_models_see():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    titles = ["Descriptor baseline", "Classical GNN (message passing)", "Graph transformer (GraphGPS)"]
    captions = [
        "Molecule is hashed into a fixed\nfingerprint + ~11 descriptors.\nThe model never sees the graph -\n"
        "only which substructures exist.",
        "Each layer, every atom aggregates\nmessages from bonded neighbours.\nk layers → k-bond receptive field;\n"
        "far atoms interact only indirectly.",
        "Each layer = local message passing\n+ global self-attention over all atoms.\nRandom-walk encodings tell attention\n"
        "where each atom sits in the graph.",
    ]
    for ax, t, c in zip(axes, titles, captions):
        ax.set_xlim(-2, 2); ax.set_ylim(-2.6, 1.8); ax.axis("off"); ax.set_aspect("equal")
        ax.text(-1.9, 1.6, t, fontsize=13, weight="bold", color=INK)
        ax.text(-1.9, -1.7, c, fontsize=9.5, color=INK2, va="top")

    # baseline: molecule -> bit vector
    ax = axes[0]
    pts, edges = _molecule(ax, -0.9, 0.35, 0.8)
    _draw_mol(ax, pts, edges, [MUTED] * 7)
    _arrow(ax, (-0.1, 0.35), (0.35, 0.35))
    bits = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    for k, b in enumerate(bits):
        ax.add_patch(plt.Rectangle((0.45 + (k % 4) * 0.3, 0.8 - (k // 4) * 0.35), 0.26, 0.3,
                                   fc=BLUE if b else "#e3eefb", ec=SURFACE, lw=1.5))
    ax.text(1.05, -0.45, "fingerprint bits", ha="center", fontsize=9, color=INK2)

    # GNN: highlight 1-hop neighbourhood of centre atom with arrows in
    ax = axes[1]
    pts, edges = _molecule(ax, 0, 0.1, 1.2)
    colors = [MUTED] * 7
    center, nbrs = 2, [1, 3, 6]
    colors[center] = ORANGE
    for n in nbrs:
        colors[n] = "#f6b394"
    _draw_mol(ax, pts, edges, colors)
    for n in nbrs:
        _arrow(ax, pts[n], pts[center], color=ORANGE, lw=2)

    # transformer: local bonds + dashed attention to every atom
    ax = axes[2]
    pts, edges = _molecule(ax, 0, 0.1, 1.2)
    colors = [AQUA if i != 5 else "#0f7a54" for i in range(7)]
    for i in range(7):
        if i != 5:
            ax.add_patch(FancyArrowPatch(pts[5], pts[i], arrowstyle="-", color=AQUA, lw=1.4,
                                         ls=(0, (3, 2)), connectionstyle="arc3,rad=-0.3", zorder=0))
    _draw_mol(ax, pts, edges, colors)
    ax.text(0, -1.35, "dark atom attends to every other atom (dashed)", fontsize=9, color=INK2, ha="center")
    save(fig, "11_how_models_see_a_molecule.png")


def diagram_model_ladder():
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
    steps = [
        ("GCN", "Degree-normalised mean\nof neighbours.\nIgnores bond type.", ORANGE, "#fde7dc"),
        ("GIN (GINE)", "Sum aggregation + MLP.\nWL-test expressive.\nAdds bond features.", ORANGE, "#fde7dc"),
        ("MPNN", "Bond-conditioned weight\nmatrix per edge + GRU.\n(Gilmer et al. 2017)", ORANGE, "#fde7dc"),
        ("GATv2", "Learned attention over\nbonded neighbours only.\nStill local.", AQUA, "#dcf3ea"),
        ("GraphGPS", "Local GINE + global\nmulti-head attention\n+ random-walk PE.", AQUA, "#dcf3ea"),
    ]
    w, gap = 2.55, 0.35
    for k, (name, desc, ec, fc) in enumerate(steps):
        x = 0.25 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 2.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, 2.7, name, ha="center", fontsize=13, weight="bold", color=INK)
        ax.text(x + w / 2, 1.75, desc, ha="center", va="center", fontsize=9.5, color=INK2)
        if k < len(steps) - 1:
            _arrow(ax, (x + w + 0.03, 1.8), (x + w + gap - 0.03, 1.8), color=INK2)
    ax.text(0.25, 4.15, "The model ladder - each step adds one mechanism", fontsize=15,
            weight="bold", color=INK)
    ax.text(0.25, 3.65, "Classical GNNs (orange) → attention-based models (aqua). "
            "Same head, pooling and training loop throughout, so only the encoder changes.",
            fontsize=10.5, color=INK2)
    ax.annotate("", xy=(14.7, 0.25), xytext=(0.25, 0.25),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(7.5, 0.0, "more expressive receptive field  →", ha="center", fontsize=9.5, color=MUTED)
    save(fig, "12_model_ladder.png")


def diagram_scaffold_split():
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
    rng = np.random.default_rng(7)
    # six toy scaffolds, each with its own colour AND marker so identity isn't colour-only
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#4a3aa7"]
    marks = ["o", "s", "^", "D", "P", "v"]
    names = ["benzene", "pyridine", "cyclohexane", "furan", "oxirane", "acyclic"]
    per = 9
    boxes = [("Train (80%)", 0.2, 5.4), ("Val (10%)", 5.9, 1.9), ("Test (10%)", 8.1, 1.9)]
    for ax, mode in zip(axes, ["random", "scaffold"]):
        ax.set_xlim(0, 10.2); ax.set_ylim(1.1, 4.7); ax.axis("off")
        for label, x0, w in boxes:
            ax.add_patch(FancyBboxPatch((x0, 2.3), w, 1.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc="#f0efec", ec="#f0efec"))
            ax.text(x0 + w / 2, 4.35, label, ha="center", fontsize=11, weight="bold", color=INK)
        bins = {0: [], 1: [], 2: []}
        for s_ in range(6):
            for _ in range(per):
                if mode == "random":
                    b = int(rng.choice(3, p=[0.72, 0.14, 0.14]))
                else:
                    b = 0 if s_ < 4 else (1 if s_ == 4 else 2)
                bins[b].append(s_)
        for b, (_, x0, w) in enumerate(boxes):
            ncol = int((w - 0.2) // 0.42)
            for k, s_ in enumerate(sorted(bins[b])):
                cx = x0 + 0.35 + (k % ncol) * 0.42
                cy = 3.75 - (k // ncol) * 0.42
                ax.plot(cx, cy, marker=marks[s_], color=cols[s_], markersize=11,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, ls="")
        ax.set_title("Random split" if mode == "random" else "Scaffold split (Bemis-Murcko)", pad=2)
    axes[0].text(0.2, 2.1, "The same ring systems appear in train and test\n→ test measures interpolation",
                 fontsize=10, color=INK2, va="top")
    axes[1].text(0.2, 2.1, "Each scaffold lives in exactly one split\n→ test measures generalization to unseen chemistry",
                 fontsize=10, color=INK2, va="top")
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", markersize=10) for m, c in zip(marks, cols)]
    fig.legend(handles, names, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Why two splits? Each marker is a molecule, shaped and coloured by its scaffold",
                 x=0.01, y=1.1, ha="left", fontsize=14, fontweight="bold", color=INK)
    save(fig, "13_scaffold_split_diagram.png")


def main():
    combined = load("combined_results.csv")
    pt = load("per_target_metrics.csv")
    if pt is not None:
        # eval_checkpoints.py covers every model (incl. retrained baselines), so
        # prefer its macro rows; keep params from the training run's table.
        macro = pt[pt.target == "macro_avg"].drop(columns="target").copy()
        macro["family"] = macro.model.map(MODEL_FAMILY)
        if combined is not None:
            macro = macro.merge(combined[["model", "split", "params"]], on=["model", "split"], how="left")
        else:
            macro["params"] = np.nan
        combined = macro
    if combined is not None:
        chart_split_comparison(combined)
        chart_generalization_gap(combined)
        chart_params_vs_mae(combined)
    if pt is not None:
        chart_per_target_heatmap(pt)
    chart_parity("random")
    chart_parity("scaffold")
    phys = load("physics_consistency.csv")
    if phys is not None:
        chart_physics(phys[phys.model != "ground_truth"])
    cov = load("coverage_curves.csv")
    if cov is not None:
        chart_coverage(cov)
    diagram_pipeline()
    diagram_how_models_see()
    diagram_model_ladder()
    diagram_scaffold_split()


if __name__ == "__main__":
    main()
