"""Replot existing development predictions; never fit or select a model."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
sources = {}


def read(relative):
    p = PROJECT / relative
    sources[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


cohort = read("results/reliability/tables/development_cohort.csv")
assert len(cohort) == 3184 and cohort.subject_id.nunique() == 810
fig, axes = plt.subplots(2, 1, figsize=(3.05, 3.1), constrained_layout=True)
axes[0].hist(cohort.sbp_mmHg, bins=np.arange(60, 221, 10), alpha=.65, color="#285f85", label="SBP")
axes[0].hist(cohort.dbp_mmHg, bins=np.arange(30, 151, 10), alpha=.65, color="#cf7b32", label="DBP")
axes[0].set(xlabel="Reference blood pressure (mmHg)", ylabel="Records", title="(a) Development BP distribution")
axes[0].legend(frameon=False)
for shift, col, color, label in [(-.18, "ecg_quality_label", "#285f85", "ECG"),
                                 (.18, "ppg_quality_label", "#cf7b32", "PPG")]:
    counts = cohort[col].value_counts().reindex(range(4), fill_value=0)
    axes[1].bar(np.arange(4)+shift, counts, .36, color=color, label=label)
axes[1].set(xticks=range(4), xlabel="Manually annotated poor windows (0–3)", ylabel="Records", title="(b) Manual quality annotations")
axes[1].legend(frameon=False)
save(fig, "figure1_cohort")

contrasts = read("results/reliability/tables/fusion_contrasts.csv")
primary = contrasts[contrasts.primary].copy()
assert len(primary) == 4 and np.allclose(primary.adjusted_confidence, .9875)
fig, ax = plt.subplots(figsize=(3.05, 2.0), constrained_layout=True)
labels = []
for y, (cond, target) in enumerate([("clean", "SBP"), ("clean", "DBP"),
                                    ("known_corruption_mean", "SBP"), ("known_corruption_mean", "DBP")]):
    r = primary[(primary.condition == cond) & (primary.target == target)].iloc[0]
    ax.errorbar(r.delta_MAE, y, xerr=[[r.delta_MAE-r.adjusted_lower], [r.adjusted_upper-r.delta_MAE]],
                fmt="o", color="#285f85", capsize=3, markersize=4)
    labels.append(("Clean" if cond == "clean" else "Known noise")+" / "+target)
ax.axvline(0, color=".45", lw=.8, ls="--")
ax.set(yticks=range(4), yticklabels=labels, xlabel="MAE difference (mmHg): full risk − BP-only", xlim=(-.021, .002))
ax.invert_yaxis()
ax.grid(axis="x", alpha=.2)
save(fig, "figure2_primary")

sel = read("results/reliability/tables/selective_risk_coverage.csv")
clean = sel[sel.condition == "clean"]
means = clean.groupby(["selector", "validation_target_coverage", "target"], as_index=False).mean(numeric_only=True)
fig, axes = plt.subplots(2, 1, figsize=(3.05, 3.15), constrained_layout=True)
for ax, target in zip(axes, ["SBP", "DBP"]):
    for selector, color, marker in [("risk", "#285f85", "o"), ("quality", "#cf7b32", "s")]:
        g = means[(means.selector == selector) & (means.target == target)].sort_values("actual_record_coverage")
        ax.plot(100*g.actual_record_coverage, g.selective_MAE, marker=marker, color=color, label=selector.capitalize())
        if selector == "risk":
            ax.plot(100*g.actual_record_coverage, g.random_rejection_MAE_mean, color=".45", ls="--", label="Random (risk counts)")
    ax.set(xlabel="Actual retained records (%)", ylabel=f"{target} MAE (mmHg)", xlim=(45, 103))
    ax.grid(alpha=.2)
axes[0].legend(frameon=False, fontsize=6.5)
save(fig, "figure3_retention")

pred = read("results/reliability/predictions/adaptive_risk_fusion_seed_17_clean.csv")
assert pred.record_id.nunique() == len(cohort)
fig, axes = plt.subplots(2, 1, figsize=(3.05, 3.2), constrained_layout=True)
ba = {}
for ax, target in zip(axes, ["SBP", "DBP"]):
    true = pred[target.lower()+"_mmHg"].to_numpy()
    estimate = pred[target.lower()+"_predicted"].to_numpy()
    error = estimate-true
    mean, sd = error.mean(), error.std(ddof=1)
    ba[target] = {"seed": 17, "mean_error": mean, "sd_error": sd,
                  "lower_limit": mean-1.96*sd, "upper_limit": mean+1.96*sd}
    ax.scatter((estimate+true)/2, error, s=2, alpha=.15, color="#285f85", rasterized=True)
    for value, style in [(mean, "-"), (mean-1.96*sd, "--"), (mean+1.96*sd, "--")]:
        ax.axhline(value, ls=style, color="#8d3c2e", lw=.8)
    ax.set(xlabel="Mean of estimate and reference (mmHg)", ylabel=f"{target} error (mmHg)")
    ax.text(.02, .97, f"Bias {mean:.2f}; limits [{mean-1.96*sd:.2f}, {mean+1.96*sd:.2f}]",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1})
save(fig, "figure4_agreement")
(ROOT / "figure_provenance.json").write_text(json.dumps({"source_sha256": sources, "bland_altman_seed17": ba,
                                                          "dpi": 400, "fitting_performed": False}, indent=2))
print("Created four figures as PNG/PDF from verified saved data.")
