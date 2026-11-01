from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset_loader import QUALITY, TARGETS, read_waveform
from .evaluate import regression_metrics

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True, "pdf.fonttype": 42})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataset_figures(root, cohort, arrays, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, target, label in zip(axes, TARGETS, ["SBP", "DBP"], strict=True):
        ax.hist(cohort[target], bins=25, color="steelblue", edgecolor="white")
        ax.set(xlabel=f"{label} (mmHg)", ylabel="Records", title=f"{label} distribution: usable cohort")
    save(fig, directory, "target_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, modality in zip(axes, QUALITY, ["ECG", "PPG"], strict=True):
        counts = cohort[col].value_counts().reindex(range(4), fill_value=0)
        ax.bar(counts.index, counts.values, color="teal")
        ax.set(xlabel="Manually annotated poor windows", ylabel="Records", title=f"{modality} quality", xticks=range(4))
    save(fig, directory, "manual_quality_distribution")
    fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    for c, modality in enumerate(["ECG", "PPG"]):
        for j, label in enumerate([0, 3]):
            subset = cohort[cohort[QUALITY[c]] == label]
            if subset.empty:
                continue
            row = subset.iloc[0]
            raw = read_waveform(root, row.wfdb_record_path)[:3000, c].astype(float)
            raw = (raw - np.median(raw)) / max(raw.std(), 1e-8)
            axes[c, j].plot(np.arange(len(raw))/500, raw, linewidth=.6)
            axes[c, j].set(title=f"{modality}, manual poor windows={label}, {row.record_id}",
                           xlabel="Time (s; first 6 s)", ylabel="Standardized digital amplitude (a.u.)")
    save(fig, directory, "waveform_quality_examples")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for c, modality in enumerate(["ECG", "PPG"]):
        for feature, title in [(0, "band power"), (1, "baseline drift"), (2, "high-frequency noise")]:
            values = [arrays["sqi"][cohort[QUALITY[c]].to_numpy() == q, c, feature].mean() for q in range(4)]
            axes[c].plot(range(4), values, "o-", label=title)
        axes[c].set(title=modality, xlabel="Manually annotated poor windows", ylabel="Fraction of spectral power", xticks=range(4))
        axes[c].legend(fontsize=8)
    save(fig, directory, "sqi_drift_noise_analysis")


def result_figures(output, summary, predictions, histories):
    output = Path(output)
    directory = output / "figures"
    names = summary.model.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for metric in ["MAE", "RMSE"]:
        fig, ax = plt.subplots(figsize=(11, 4))
        positions = np.arange(len(names))
        for c, target in enumerate(["SBP", "DBP"]):
            ax.bar(positions + (c-.5)*.36, summary[f"{target}_{metric}"], .36,
                   yerr=summary[f"{target}_{metric}_seed_SD"].fillna(0), capsize=3, label=target)
        uncertainty = "error bars: training-seed SD" if summary.n_seeds.min() > 1 else "one seed: uncertainty not estimated"
        ax.set(xticks=positions, xticklabels=names, ylabel=f"{metric} (mmHg)", title=f"Participant-independent test {metric}; {uncertainty}")
        ax.tick_params(axis="x", labelrotation=28)
        ax.legend()
        save(fig, directory, f"model_comparison_{metric.lower()}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    selected = "proposed" if "proposed" in histories else next(iter(histories))
    history = histories[selected]
    for phase in ["train", "validation"]:
        axes[0].plot(history.epoch, history[phase + "_loss"], label=phase)
    axes[0].set(xlabel="Epoch", ylabel="Total objective (standardized MSE + auxiliary loss)", title=f"{selected}: first predeclared seed")
    axes[0].legend()
    for name, frame in histories.items():
        axes[1].plot(frame.epoch, frame.validation_mae_mmHg, label=name)
    axes[1].set(xlabel="Epoch", ylabel="Mean SBP/DBP validation MAE (mmHg)", title="Validation curves: first predeclared seed")
    axes[1].legend(fontsize=7)
    save(fig, directory, "training_validation_history")
    name = "proposed" if "proposed" in predictions else next(iter(predictions))
    frame = predictions[name][0]
    for c, target in enumerate(["SBP", "DBP"]):
        actual = frame[TARGETS].to_numpy()
        predicted = frame[["sbp_predicted", "dbp_predicted"]].to_numpy()
        y, p = actual[:, c], predicted[:, c]
        metrics = regression_metrics(actual, predicted)[c]
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.scatter(y, p, s=10, alpha=.4)
        low, high = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(xlabel=f"Cuff reference {target} (mmHg)", ylabel=f"Predicted {target} (mmHg)",
               title=f"{name}: {target}, MAE={metrics['MAE']:.2f} mmHg")
        save(fig, directory, f"actual_vs_predicted_{target.lower()}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter((p+y)/2, p-y, s=10, alpha=.4)
        for value, label, color in [(metrics["ME"], "Mean error", "black"), (metrics["LoA_lower"], "-1.96 SD", "red"), (metrics["LoA_upper"], "+1.96 SD", "red")]:
            ax.axhline(value, color=color, linestyle="--", label=f"{label}: {value:.2f}")
        ax.set(xlabel=f"Mean of prediction and reference {target} (mmHg)", ylabel="Prediction - reference (mmHg)", title=f"Descriptive Bland–Altman: {target}")
        ax.legend(fontsize=8)
        save(fig, directory, f"bland_altman_{target.lower()}")
    strata = pd.read_csv(output / "tables/quality_stratified.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for model in names:
            values = strata[(strata.model == model) & (strata.target == target) & (strata.modality == "worst_modality")].groupby("manual_poor_windows").MAE.mean()
            ax.plot(values.index, values.values, "o-", label=model)
        ax.set(xlabel="Worst-modality manual poor-window count", ylabel=f"{target} MAE (mmHg)", title=f"{target}: test quality strata", xticks=range(4))
    axes[1].legend(fontsize=7)
    save(fig, directory, "performance_by_signal_quality")
    ablation_names = ["ecg_only", "ppg_only", "standard_fusion", "quality_fusion", "proposed_no_gate", "proposed"]
    ablation = summary[summary.model.isin(ablation_names)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ablation.model, (ablation.SBP_MAE + ablation.DBP_MAE)/2)
    ax.set(ylabel="Mean SBP/DBP MAE (mmHg)", title="Ablation: modality, SQI gating, and supervised quality gating")
    ax.tick_params(axis="x", labelrotation=25)
    save(fig, directory, "ablation_comparison")
    error = (summary.SBP_MAE + summary.DBP_MAE)/2
    markers = ["o", "s", "^", "D", "P", "X", "v", "h"]
    for column, xlabel, filename in [("parameters", "Trainable parameters", "parameters_vs_error"), ("latency_ms", "Batch-1 model latency (ms)", "latency_vs_error")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for x, y, label, color, marker in zip(summary[column], error, names, colors, markers, strict=True):
            ax.scatter(x, y, color=color, marker=marker, s=65, label=label)
        ax.set(xlabel=xlabel, ylabel="Mean SBP/DBP MAE (mmHg)", title="Accuracy–efficiency trade-off")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        save(fig, directory, filename)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, marker) in enumerate(zip(names, markers, strict=True)):
        points = ax.scatter(summary.parameters.iloc[i], error.iloc[i], c=[summary.latency_ms.iloc[i]],
                            s=summary.model_size_KiB.iloc[i] * 1.5 + 40, marker=marker, cmap="viridis", alpha=.8,
                            vmin=summary.latency_ms.min(), vmax=summary.latency_ms.max(), label=label)
    ax.set(xlabel="Parameters; marker area ∝ serialized weight size", ylabel="Mean SBP/DBP MAE (mmHg)", title="Accuracy, parameter count, size, and latency")
    ax.legend(loc="upper left", bbox_to_anchor=(0., -.22), fontsize=8, ncol=3, markerscale=.6)
    fig.colorbar(points, ax=ax, label="Batch-1 latency (ms)")
    save(fig, directory, "accuracy_efficiency_tradeoff")
    robustness = pd.read_csv(output / "tables/robustness.csv")
    pivot = robustness.groupby(["model", "condition"]).MAE.mean().unstack().reindex(names)
    fig, ax = plt.subplots(figsize=(10, 5))
    heat = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(yticks=range(len(pivot)), yticklabels=pivot.index, xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           title="Controlled test-only degradation: mean SBP/DBP MAE (mmHg)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.colorbar(heat, ax=ax, label="MAE (mmHg)")
    save(fig, directory, "controlled_degradation")
    diagnostics = pd.read_csv(output / "tables/quality_estimator.csv")
    if (diagnostics.model == "proposed").any():
        selected = diagnostics[diagnostics.model == "proposed"].groupby(["condition", "modality"])[["mean_estimated_goodness", "mean_gate"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, field, title in zip(axes, ["mean_estimated_goodness", "mean_gate"], ["Estimated goodness", "Fusion weights"], strict=True):
            for modality in ["ECG", "PPG"]:
                values = selected[selected.modality == modality]
                ax.plot(values.condition, values[field], "o-", label=modality)
            ax.set(ylabel="Mean score (0–1)", title=f"Proposed: {title}", ylim=(0, 1.05))
            ax.tick_params(axis="x", labelrotation=30)
            ax.legend()
        save(fig, directory, "quality_gate_diagnostics")


def research_report(output, summary, audit, config, split_sizes):
    output = Path(output)
    lookup = summary.set_index("model")
    mode = "QUICK TEST — NOT RESEARCH RESULTS" if config["mode"] == "quick" else "FULL COHORT EXPERIMENT — local QC cohort"
    lines = [f"# Research results: {mode}", "", "All values below come from executed experiments.", "",
             f"Metadata: {audit['metadata_records']} records / {audit['metadata_subjects']} participants. "
             f"After pairing and QC: {audit['usable_records']} records / {audit['usable_subjects']} participants. "
             f"Excluded: {audit['excluded_records']}; unreferenced waveform pairs ignored: {audit['unreferenced_waveform_pairs']}.", "",
             f"Participant-independent split: {split_sizes}. Training seeds: {config['seeds']}; split seed: {config['split_seed']}.", "",
             "## Measured comparison", "", "Mean ± sample SD over training seeds, all on the same untouched participant test split. With one seed, SD is unavailable (n/a). Latency is model-only batch 1.", "",
             "| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | Parameters | Latency (ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        sbp_sd = f"{row.SBP_MAE_seed_SD:.3f}" if row.n_seeds > 1 else "n/a"
        dbp_sd = f"{row.DBP_MAE_seed_SD:.3f}" if row.n_seeds > 1 else "n/a"
        lines.append(f"| {row.model} | {row.SBP_MAE:.3f} ± {sbp_sd} | {row.DBP_MAE:.3f} ± {dbp_sd} | {row.SBP_RMSE:.3f} | {row.DBP_RMSE:.3f} | {int(row.parameters)} | {row.latency_ms:.3f} |")
    lines += ["", "## Research questions", ""]
    if {"standard_fusion", "ecg_only", "ppg_only", "proposed"}.issubset(lookup.index):
        proposed = lookup.loc["proposed"]
        for target in ["SBP", "DBP"]:
            best_uni = min(lookup.loc["ecg_only", target+"_MAE"], lookup.loc["ppg_only", target+"_MAE"])
            delta = lookup.loc["standard_fusion", target+"_MAE"] - best_uni
            lines.append(f"- RQ1 ({target}): standard fusion minus best unimodal MAE = {delta:+.3f} mmHg. Negative favors multimodal fusion.")
        robust = pd.read_csv(output / "tables/robustness.csv")
        for condition in robust.condition.unique():
            selection = robust[robust.condition == condition].groupby("model").MAE.mean()
            delta = selection["proposed"] - selection["standard_fusion"]
            lines.append(f"- RQ2 ({condition}): proposed minus standard fusion mean SBP/DBP MAE = {delta:+.3f} mmHg. See manual-quality strata and learned-gate diagnostics; robustness benefit is not assumed.")
        if "proposed_dense" in lookup.index:
            dense = lookup.loc["proposed_dense"]
            reduction = 100 * (1 - proposed.parameters/dense.parameters)
            lines.append(f"- RQ3: proposed has {int(proposed.parameters)} parameters versus {int(dense.parameters)} for its matched dense-convolution ablation ({reduction:.2f}% reduction). Measured latency: {proposed.latency_ms:.3f} versus {dense.latency_ms:.3f} ms; fewer MACs need not mean faster kernels.")
        lines.append(f"- RQ4: held-out participant proposed SBP MAE={proposed.SBP_MAE:.3f}, DBP MAE={proposed.DBP_MAE:.3f} mmHg; R²={proposed.SBP_R2:.3f}/{proposed.DBP_R2:.3f}. The three seeds repeat initialization on one subject split, not three independent cohorts.")
        errors = (summary.SBP_MAE + summary.DBP_MAE)/2
        pareto = []
        for i, row in summary.iterrows():
            dominates = (errors <= errors.iloc[i]) & (summary.latency_ms <= row.latency_ms) & ((errors < errors.iloc[i]) | (summary.latency_ms < row.latency_ms))
            if not dominates.any():
                pareto.append(row.model)
        lines.append(f"- RQ5: observed latency/mean-MAE Pareto set: {', '.join(pareto)}. Inspect parameters, MACs, RSS, and end-to-end latency in efficiency_comparison.csv.")
        paired = pd.read_csv(output / "tables/paired_MAE_differences.csv")
        for row in paired[paired.comparison.isin(["proposed minus standard_fusion", "proposed minus proposed_no_gate"])].itertuples():
            conclusion = "interval crosses zero: advantage not established" if row.lower_95 <= 0 <= row.upper_95 else "exploratory interval excludes zero"
            lines.append(f"- Quality contribution ({row.comparison}, {row.target}): delta MAE {row.delta_MAE:+.3f} mmHg, participant-bootstrap 95% CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}]; {conclusion}.")
    lines += ["", "## Interpretation limits", "",
              "- This is a technical, retrospective benchmark, not clinical device validation. Cuff reference was obtained after the waveform, not simultaneously.",
              "- This local package is not the complete documented cohort: missing record paths and ambiguous duplicate signals are excluded transparently; no replacement data or inferred labels are used.",
              "- A small single-center cohort, hardware dependence, repeated recordings, quality imbalance, and absent external validation restrict generalization.",
              "- Digital values and per-record standardized amplitudes are not calibrated physiological units. Zero-phase offline filtering requires a complete 24-second record; streaming latency is not established.",
              "- SQIs and peak regularity are heuristics, not validated physiological measurements. Manual windows partly extend outside the released crop. Manual labels are never inference inputs.",
              "- White noise/missing modalities are controlled stress tests, not a complete model of motion/contact artefacts. Targets remain unchanged.",
              "- Cluster-bootstrap intervals describe test-participant sampling for fixed trained models. Paired seed-averaged-loss intervals do not model training-population uncertainty; comparisons are exploratory and unadjusted for multiplicity.",
              "- Bland–Altman limits are descriptive across repeated records, not repeated-measures clinical agreement limits. SD uses ddof=1; error is prediction minus reference.",
              "- One prespecified optimizer/configuration and validation early stopping are used for every model; no architecture-specific hyperparameter search or test-set selection occurs.",
              "- Latency includes its device, threads, float32, warm-up, repetitions, and batch size. Isolated peak RSS includes the Python/PyTorch runtime, not only model allocations. MAC counts cover matrix operations only.",
              "- A negative R² or a training-mean baseline outperforming a model is reported as observed and is not evidence of a successful estimator.",
              "- Quick-test results are only pipeline checks and must not support research claims.", ""]
    (output / "RESEARCH_REPORT.md").write_text("\n".join(lines))
