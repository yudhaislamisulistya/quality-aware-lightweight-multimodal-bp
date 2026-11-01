"""Tables, figures and candid conclusions from the frozen factorial experiment."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset_loader import TARGETS
from .factorial_model import CELLS, STUDY_MODELS
from .visualization import save

LABELS = {"A_uniform_clean": "A: uniform, clean", "B_gated_clean": "B: gated, clean",
          "C_uniform_augmented": "C: uniform, augmented", "D_gated_augmented": "D: gated, augmented",
          "ecg_only_clean": "ECG only", "ppg_only_clean": "PPG only", "dense_uniform_clean": "Dense, uniform, clean",
          "training_mean": "Training mean", "training_median": "Training median"}


def render_report(output, summary, efficiency, config, predictions):
    output = Path(output)
    figures = output/"figures"
    clean = summary[summary.condition == "clean"]
    rows = []
    names = STUDY_MODELS + ["training_mean", "training_median"]
    hardware = efficiency.groupby("model").mean(numeric_only=True)
    for name in names:
        row = {"model": name}
        for target in ["SBP", "DBP"]:
            data = clean[(clean.model == name) & (clean.target == target)].iloc[0]
            for field in ["MAE", "MAE_seed_SD", "RMSE", "R2", "ME", "SD_error", "participant_macro_MAE", "predicted_SD", "reference_SD"]:
                row[f"{target}_{field}"] = data[field]
        for field in ["parameters", "MACs", "approx_FLOPs", "latency_ms", "serialized_weights_KiB", "throughput_batch32_per_s", "end_to_end_ms", "peak_RSS_MiB"]:
            row[field] = hardware.loc[name, field] if name in hardware.index else (0 if field in ["parameters", "MACs", "approx_FLOPs"] else None)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output/"tables/comparison.csv", index=False)
    contrasts = pd.read_csv(output/"tables/factorial_contrasts.csv")
    for metric in ["MAE", "RMSE"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
            yerr = comparison[f"{target}_MAE_seed_SD"].fillna(0) if metric == "MAE" else None
            ax.bar(range(len(names)), comparison[f"{target}_{metric}"], yerr=yerr, capsize=3)
            ax.set(xticks=range(len(names)), xticklabels=[LABELS[n] for n in names], ylabel=f"{target} {metric} (mmHg)", title=f"Development OOF {target}; all seeds")
            ax.tick_params(axis="x", labelrotation=45)
        save(fig, figures, f"clean_{metric.lower()}")
    primary = contrasts[contrasts.primary]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in enumerate(primary.itertuples()):
        low, high = row.primary_family_Bonferroni_lower, row.primary_family_Bonferroni_upper
        ax.plot([low, high], [i, i], color="steelblue")
        ax.scatter(row.delta_MAE, i, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(yticks=range(len(primary)), yticklabels=[r.contrast+": "+r.target for r in primary.itertuples()],
           xlabel="Delta MAE (mmHg); negative favors gating", title="Primary gating contrasts; Bonferroni conditional bootstrap intervals")
    save(fig, figures, "primary_gating_contrasts")
    pivot = summary.groupby(["model", "condition"]).MAE.mean().unstack().reindex(names)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "abstain", ha="center", va="center", fontsize=8)
    ax.set(yticks=range(len(pivot)), yticklabels=[LABELS[n] for n in names], xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           title="Absolute mean SBP/DBP MAE (mmHg); synthetic stress is exploratory")
    ax.tick_params(axis="x", labelrotation=40)
    fig.colorbar(im, ax=ax, label="MAE (mmHg)")
    save(fig, figures, "absolute_robustness")
    selective = pd.read_csv(output/"tables/selective_risk_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for name in CELLS:
            values = selective[(selective.condition == "clean") & (selective.model == name) & (selective.target == target)].groupby("validation_target_coverage").mean(numeric_only=True)
            ax.plot(values.actual_record_coverage*100, values.selective_MAE, "o-", label=LABELS[name])
            ax.plot(values.actual_record_coverage*100, values.random_rejection_MAE_mean, "--", alpha=.4)
        ax.set(xlabel="Actual retained records (%)", ylabel=f"{target} MAE (mmHg)", title="Validation-fixed quality thresholds; dashed=random rejection")
        ax.legend(fontsize=7)
    save(fig, figures, "selective_risk_coverage")
    subgroups = pd.read_csv(output/"tables/selective_subgroups.csv")
    values = subgroups[(subgroups.model == "D_gated_augmented") & (subgroups.validation_target_coverage == .7) & (subgroups.target == "SBP")].groupby("subgroup").subgroup_coverage.mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(values)), values*100)
    ax.set(xticks=range(len(values)), xticklabels=values.index, ylabel="Retained records (%)", title="D: subgroup coverage at validation target 70%; descriptive BP/age bins")
    ax.tick_params(axis="x", labelrotation=45)
    save(fig, figures, "selective_subgroup_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field in zip(axes, ["parameters", "latency_ms"], strict=True):
        for row in comparison[comparison.model.isin(STUDY_MODELS)].itertuples():
            ax.scatter(getattr(row, field), (row.SBP_MAE+row.DBP_MAE)/2, label=LABELS[row.model])
        ax.set(xlabel="Parameters" if field == "parameters" else "Model latency (ms; batch 1)", ylabel="Mean SBP/DBP MAE (mmHg)", title="Accuracy–efficiency on the recorded device")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, figures, "accuracy_efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in CELLS:
        history = pd.read_csv(output/"fits"/"fold_0"/name/f"seed_{config['seeds'][0]}"/"history.csv")
        axes[0].plot(history.epoch, history.train_regression, label=LABELS[name])
        axes[1].plot(history.epoch, history.validation_clean_MAE, label=LABELS[name])
    axes[0].set(xlabel="Epoch", ylabel="Smooth L1 regression objective (mmHg scale)", title="First declared fold/seed; training")
    axes[1].set(xlabel="Epoch", ylabel="Mean SBP/DBP MAE (mmHg)", title="Inner validation; checkpoint criterion")
    axes[1].legend(fontsize=7)
    save(fig, figures, "training_validation")
    sample = predictions["clean"]["D_gated_augmented"][0]
    for c, target in enumerate(["SBP", "DBP"]):
        y, p = sample[TARGETS[c]].to_numpy(), sample[["sbp_predicted", "dbp_predicted"][c]].to_numpy()
        error = p-y
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y, p, s=8, alpha=.25)
        axes[0].plot([min(y.min(), p.min()), max(y.max(), p.max())], [min(y.min(), p.min()), max(y.max(), p.max())], "k--")
        axes[0].set(xlabel=f"Cuff reference {target} (mmHg)", ylabel=f"Estimated {target} (mmHg)", title="D: first declared seed, OOF")
        axes[1].scatter((p+y)/2, error, s=8, alpha=.25)
        for value in [error.mean(), error.mean()-1.96*error.std(ddof=1), error.mean()+1.96*error.std(ddof=1)]:
            axes[1].axhline(value, linestyle="--", linewidth=1)
        axes[1].set(xlabel=f"Mean of reference and prediction (mmHg)", ylabel="Prediction minus reference (mmHg)", title=f"{target} Bland–Altman; descriptive repeated records")
        save(fig, figures, f"agreement_{target.lower()}")
    strata = pd.read_csv(output/"tables/quality_stratified.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for name in CELLS:
            data = strata[(strata.model == name) & (strata.target == target) & (strata.modality == "worst_modality")].groupby("manual_poor_windows").MAE.mean()
            ax.plot(data.index, data, "o-", label=LABELS[name])
        ax.set(xticks=range(4), xlabel="Worst manual poor-window count (clean records)", ylabel=f"{target} MAE (mmHg)")
    axes[1].legend(fontsize=7)
    save(fig, figures, "natural_quality_strata")

    lines = ["# Matched factorial development study", "",
             "**SMOKE TEST ONLY — NOT ARTICLE RESULTS**" if config["quick"] else "**Executed development experiment. Reused, previously analyzed cohort; not a fresh confirmatory test.**", "",
             "## Protocol and interpretation", "",
             "Three participant-disjoint outer folds; inner training/validation roles match the previous development study. Historical 143 test participants remain excluded. No outer record fits a scaler or selects a checkpoint/quality threshold for its own fold.",
             "All neural fits and validation-derived thresholds are sealed before any outer evaluation. All seven models use the same seeds, optimizer, Smooth L1 BP loss, clean validation criterion, maximum epoch budget, and early-stopping rule. No hyperparameter search was performed in this study.",
             "The four cells share identical parameter counts, encoder, regression head, availability masks, manual-quality auxiliary loss, and initial state per seed. A/B use clean BP training; C/D average clean and perturbed BP losses. A/C use uniform fusion over available sensors; B/D use estimated goodness gates. The corruption-ranking loss is disabled in every cell, so it cannot confound the augmentation contrast.",
             "Manual annotation is a training target only. Model inference and rejection accept waveform-derived inputs only. The quality head uses detached encoder features; BP gradients do not redefine its quality scores. No task-specific residual-risk predictor or oracle manual-quality inference was added.",
             "ECG-only and PPG-only use the same branch encoder, auxiliary supervision, loss and clean training as A; their regressors use one embedding. The dense control changes only depthwise convolution groups to ordinary convolution. Mean/median baselines fit only inner-training BP references and retain the same predictions under waveform perturbations.", "",
             "## Clean OOF results", "", "MAE/RMSE in mmHg, mean over all declared seeds, never best seed or ensemble. Constants are deterministic; their latency was not benchmarked.", "",
             "| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | Parameters | Model latency ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        latency = f"{row.latency_ms:.3f}" if np.isfinite(row.latency_ms) else "not measured"
        lines.append(f"| {LABELS[row.model]} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} | {row.SBP_RMSE:.3f} | {row.DBP_RMSE:.3f} | {int(row.parameters)} | {latency} |")
    lines += ["", "## Primary quality-gating contrasts", "",
              "Negative differences favor gating. Four predeclared clean contrasts (two gating effects × SBP/DBP) have Bonferroni-adjusted conditional participant-bootstrap intervals (98.75% individually for family coverage approximately 95%). These percentile intervals remain development estimates; correction does not remove adaptive reuse of the cohort or model-training uncertainty.", ""]
    for row in primary.itertuples():
        lower, upper = row.primary_family_Bonferroni_lower, row.primary_family_Bonferroni_upper
        finding = "gating benefit supported conditionally in this development analysis" if upper < 0 else ("gating worsens MAE conditionally in this development analysis" if lower > 0 else "gating benefit not established; this is not evidence of equivalence")
        lines.append(f"- {row.contrast}, {row.target}: {row.delta_MAE:+.3f} mmHg; adjusted interval [{lower:+.3f}, {upper:+.3f}]; {finding}.")
    lines += ["", "Secondary augmentation, interaction, multimodal, dense-control and constant-baseline contrasts, and all stress-condition contrasts, are in `tables/factorial_contrasts.csv`. Their ordinary 95% intervals are exploratory and not multiplicity-adjusted.", "",
              "## Absolute robustness", "", "Mean of SBP and DBP MAE, averaged over seeds; synthetic perturbations at unchanged observed BP labels. These are not new patients or validated real-world artefact recordings.", "",
              "| Condition | A | B | C | D | Training median |", "|---|---:|---:|---:|---:|---:|"]
    for condition in pivot.columns:
        lines.append("| "+condition+" | "+" | ".join(f"{pivot.loc[name,condition]:.3f}" for name in list(CELLS)+["training_median"])+" |")
    lines += ["", "A constant predictor cannot deteriorate under waveform noise, so small degradation alone is not proof of useful robustness. Use absolute error, reference baselines, predicted/reference SD, natural quality strata, and matched contrasts together. Single-sensor models abstain when their required sensor is absent; these conditions have no invented BP predictions.", "",
              "## Selective prediction", "",
              "Minimum estimated ECG/PPG goodness defines rejection. Thresholds are clean inner-validation quantiles at target coverages 100%, 90%, 70%, and 50%, fixed before outer evaluation. Ties are accepted; actual coverage can differ from target and can shift sharply under perturbation. All operating points are reported; none was chosen from outer errors.",
              "Random rejection uses the same number of accepted records within each outer fold, with repeated random draws. Its Monte Carlo SD is not a participant-level confidence interval. This is a simple quality-based rejection policy inspired by selective prediction, not an implementation or validation of SelectiveNet.", "",
              "| Model | Validation target coverage | Actual clean coverage | SBP selective MAE | DBP selective MAE |", "|---|---:|---:|---:|---:|"]
    for name in CELLS:
        for coverage in config["selective_coverages"]:
            data = selective[(selective.model == name) & (selective.condition == "clean") & (selective.validation_target_coverage == coverage)]
            means = data.groupby("target").mean(numeric_only=True)
            lines.append(f"| {LABELS[name]} | {coverage:.0%} | {means.actual_record_coverage.mean():.1%} | {means.loc['SBP','selective_MAE']:.3f} | {means.loc['DBP','selective_MAE']:.3f} |")
    lines += ["", "`selective_subgroups.csv` reports retention, error before/after rejection, and counts by observed BP bands, age, recorded sex code and clean manual quality. BP/age bins are descriptive, not a clinical classification. Missing demographics are explicit; small subgroups do not establish fairness. Rejecting difficult/high-pressure cases can reduce coverage or worsen error and must not be hidden by a lower accepted-only average.", "",
              "## Research-question evidence", ""]
    for question, contrast in [("RQ1 multimodal", "multimodal_minus_ecg"), ("RQ1 multimodal", "multimodal_minus_ppg"), ("RQ3 separable versus dense", "separable_minus_dense")]:
        for row in contrasts[(contrasts.condition == "clean") & (contrasts.contrast == contrast)].itertuples():
            lines.append(f"- {question}, {row.contrast}, {row.target}: delta MAE {row.delta_MAE:+.3f}, exploratory CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}] mmHg.")
    a, dense = comparison.set_index("model").loc[["A_uniform_clean", "dense_uniform_clean"]].to_dict("index").values()
    lines += [f"- RQ3 efficiency: separable A has {int(a['parameters'])} parameters / {int(a['MACs'])} MACs versus dense {int(dense['parameters'])} / {int(dense['MACs'])}; measured latency {a['latency_ms']:.3f} versus {dense['latency_ms']:.3f} ms. Fewer parameters/MACs alone do not establish faster execution or equal accuracy.",
              "- RQ2 quality awareness: assess both predeclared gating contrasts above, quality-estimator agreement, natural quality strata, synthetic stress, and selective coverage. Implementing a gate by itself does not establish a performance advantage.",
              "- RQ4 generalization: predictions are participant-independent within this development cohort; external populations, future devices, longitudinal drift and clinical performance remain untested.",
              "- RQ5 trade-off: the comparison table/plots report measured costs and all errors. No post-hoc noninferiority margin or clinical accuracy threshold was introduced.", "",
              "## Reproducibility and limits", "",
              "- Source/config/environment/data fingerprints, participant lists, training histories, checkpoint hashes, and thresholds are saved. `selection_locked.json` predates every outer prediction. Historical result directories and original dataset files are preserved.",
              "- Bootstrap resamples participants with all their records and averages per-record absolute losses over seeds. It conditions on fitted OOF predictions, omits parts of training/selection uncertainty, and does not resolve dependence between overlapping fold training sets. Seeds are initialization repeats, not independent clinical studies.",
              "- The QC cohort omits missing records and ambiguous duplicated signals; extra waveforms have no inferred labels. Manual quality partly describes windows outside the released crop. Digital waveform amplitudes are not calibrated physiological amplitudes.",
              "- Reference BP is cuff-based at record level; it is not continuous beat-to-beat ground truth. Dataset size, BP distribution, signal quality and device/cohort specificity limit generalization. Full-record zero-phase preprocessing is offline, not a streaming or wearable deployment validation.",
              "- CPU/device benchmarks report float32 batch 1/32, threads, warm-up and repetitions. MAC scope excludes normalization/activation/pooling/preprocessing; approximate FLOPs are twice counted matrix MACs. Sampled RSS includes the runtime, not just model memory.",
              "- This study isolates method components more clearly; it does not guarantee novelty, publication acceptance, superiority, external validity, or compliance with medical-device standards. Negative and inconclusive findings remain reportable.", "",
              "## Method references", "",
              "- BP-algorithm evaluation recommendations: https://pmc.ncbi.nlm.nih.gov/articles/PMC11245506/",
              "- Selective prediction background: https://proceedings.mlr.press/v97/geifman19a.html",
              "- Subgroup concerns in selective regression: https://proceedings.mlr.press/v162/shah22a.html", ""]
    (output/"RESEARCH_REPORT.md").write_text("\n".join(lines))
    LOGGER_MESSAGE = comparison[["model", "SBP_MAE", "DBP_MAE", "parameters", "latency_ms"]].to_string(index=False)
    print(LOGGER_MESSAGE)
