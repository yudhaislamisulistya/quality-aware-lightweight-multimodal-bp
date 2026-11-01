"""Publication figures and honest interpretation of the task-reliability development study."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset_loader import TARGETS
from .risk_model import POLICIES
from .visualization import save

LABELS = {"uniform_fusion": "Uniform", "quality_fusion": "Estimated quality", "static_risk_fusion": "Static OOF risk",
          "bp_only_risk_fusion": "BP-only OOF risk",
          "adaptive_risk_fusion": "Adaptive OOF risk", "manual_quality_oracle": "Manual oracle (nondeployable)"}


def render_report(output, summary, efficiency, config, predictions):
    output = Path(output)
    figures = output/"figures"
    clean = summary[summary.condition == "clean"].set_index(["model", "target"])
    costs = efficiency.groupby("model").mean(numeric_only=True)
    rows = []
    names = POLICIES+["manual_quality_oracle"]
    for name in names:
        row = {"model": name, "deployable_without_manual_quality": name in POLICIES}
        for target in ["SBP", "DBP"]:
            for metric in ["MAE", "MAE_seed_SD", "RMSE", "R2", "ME", "SD_error", "participant_macro_MAE"]:
                row[f"{target}_{metric}"] = float(clean.loc[(name, target), metric])
        for field in ["parameters", "MACs", "latency_ms", "serialized_weights_KiB", "throughput_batch32_per_s", "end_to_end_ms", "peak_RSS_MiB"]:
            row[field] = float(costs.loc[name, field]) if name in costs.index else None
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output/"tables/comparison.csv", index=False)
    contrasts = pd.read_csv(output/"tables/fusion_contrasts.csv")
    primary = contrasts[contrasts.primary]
    feature = pd.read_csv(output/"tables/feature_gate_metrics.csv")
    feature_contrasts = pd.read_csv(output/"tables/feature_gate_contrasts.csv")
    selective = pd.read_csv(output/"tables/selective_risk_coverage.csv")
    for metric in ["MAE", "RMSE"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
            bars = ax.bar(range(len(names)), comparison[f"{target}_{metric}"],
                          yerr=comparison[f"{target}_MAE_seed_SD"].fillna(0) if metric == "MAE" else None, capsize=3)
            bars[-1].set_hatch("//")
            ax.set(xticks=range(len(names)), xticklabels=[LABELS[n] for n in names], ylabel=f"{target} {metric} (mmHg)", title=f"Participant OOF {target}; all declared seeds")
            ax.tick_params(axis="x", labelrotation=30)
        save(fig, figures, f"clean_{metric.lower()}")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, row in enumerate(primary.itertuples()):
        ax.plot([row.adjusted_lower, row.adjusted_upper], [i, i], color="steelblue")
        ax.scatter(row.delta_MAE, i, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(yticks=range(len(primary)), yticklabels=[r.condition+": "+r.target for r in primary.itertuples()],
           xlabel="Full risk minus BP-only risk MAE (mmHg); negative favors quality inputs", title="Primary conditional participant-bootstrap intervals, Bonferroni adjusted")
    save(fig, figures, "primary_risk_contrasts")
    pivot = summary[summary.model.isin(POLICIES)].groupby(["model", "condition"]).MAE.mean().unstack().reindex(POLICIES)
    fig, ax = plt.subplots(figsize=(13, 4))
    heat = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(yticks=range(len(pivot)), yticklabels=[LABELS[n] for n in POLICIES], xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           title="Absolute mean SBP/DBP MAE (mmHg); synthetic measurement corruption")
    ax.tick_params(axis="x", labelrotation=35)
    fig.colorbar(heat, ax=ax, label="MAE (mmHg)")
    save(fig, figures, "robustness")
    feature = feature.copy()
    feature.loc[feature.model.str.startswith("shuffled_gate_"), "model"] = "shuffled_gate_mean_loss"
    feature_mean = feature.groupby(["model", "target"]).MAE.mean()
    feature_names = ["learned_gate", "uniform_intervention", "manual_quality_oracle", "shuffled_gate_mean_loss"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        ax.bar(range(len(feature_names)), [feature_mean.loc[(n, target)] for n in feature_names])
        ax.set(xticks=range(len(feature_names)), xticklabels=["Learned", "Uniform override", "Manual oracle", f"Shuffled loss mean ({config['shuffle_repetitions']})"],
               ylabel=f"{target} MAE (mmHg)", title="Frozen feature model D: sensitivity diagnostics")
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, figures, "feature_gate_interventions")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for selector in ["risk", "quality"]:
            data = selective[(selective.condition == "clean") & (selective.target == target) & (selective.selector == selector)].groupby("validation_target_coverage").mean(numeric_only=True)
            line, = ax.plot(data.actual_record_coverage*100, data.selective_MAE, "o-", label=selector)
            ax.plot(data.actual_record_coverage*100, data.random_rejection_MAE_mean, "--", color=line.get_color(), alpha=.5)
        ax.set(xlabel="Actual retained records (%)", ylabel=f"{target} MAE (mmHg)", title="Same adaptive predictor; solid=selector, dashed=random")
        ax.legend()
    save(fig, figures, "risk_quality_coverage")
    subgroup = pd.read_csv(output/"tables/selective_subgroups.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    available_groups = sorted(subgroup.subgroup.unique())
    for i, selector in enumerate(["risk", "quality"]):
        values = subgroup[(subgroup.selector == selector) & (subgroup.validation_target_coverage == .7) & (subgroup.target == "SBP")].groupby("subgroup").subgroup_coverage.mean().reindex(available_groups)
        ax.bar(np.arange(len(values))+(i-.5)*.38, values*100, .38, label=selector)
    ax.set(xticks=range(len(available_groups)), xticklabels=available_groups, ylabel="Retained records (%)", title="Subgroup coverage at validation target 70%; descriptive bins")
    ax.tick_params(axis="x", labelrotation=40)
    ax.legend()
    save(fig, figures, "subgroup_coverage")
    calibration = pd.read_csv(output/"tables/risk_calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for modality in ["ecg", "ppg"]:
            data = calibration[(calibration.condition == "clean") & (calibration.target == target) & (calibration.modality == modality)].copy()
            data["prediction_sum"], data["observed_sum"] = data.predicted_risk_mean*data.n, data.observed_error_mean*data.n
            grouped = data.groupby("fixed_error_band")[["prediction_sum", "observed_sum", "n"]].sum()
            ax.plot(grouped.prediction_sum/grouped.n, grouped.observed_sum/grouped.n, "o-", label=modality.upper())
        low, high = ax.get_xlim()
        ax.plot([max(0, low), high], [max(0, low), high], "k--", linewidth=1)
        ax.set(xlabel=f"Predicted {target} absolute error (mmHg)", ylabel="Observed mean absolute error (mmHg)", title="Fixed risk bins; descriptive calibration, not a guarantee")
        ax.legend()
    save(fig, figures, "risk_calibration")
    strata = pd.read_csv(output/"tables/quality_stratified.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for name in POLICIES:
            data = strata[(strata.model == name) & (strata.target == target) & (strata.modality == "worst_modality")].groupby("manual_poor_windows").MAE.mean()
            ax.plot(data.index, data, "o-", label=LABELS[name])
        ax.set(xticks=range(4), xlabel="Worst manual poor-window count", ylabel=f"{target} MAE (mmHg)")
    axes[1].legend(fontsize=8)
    save(fig, figures, "natural_quality_strata")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, label in zip(axes, ["latency_ms", "throughput_batch32_per_s"], ["Batch-1 latency (ms)", "Batch-32 throughput (records/s)"], strict=True):
        ax.bar(range(len(POLICIES)), costs.reindex(POLICIES)[field])
        ax.set(xticks=range(len(POLICIES)), xticklabels=[LABELS[n] for n in POLICIES], ylabel=label, title="Recorded device; matched checkpoint contents")
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, figures, "efficiency")
    sample = predictions["clean"]["adaptive_risk_fusion"][0]
    for c, target in enumerate(["SBP", "DBP"]):
        y, p = sample[TARGETS[c]].to_numpy(), sample[["sbp_predicted", "dbp_predicted"][c]].to_numpy()
        error = p-y
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y, p, s=8, alpha=.25)
        extent = [min(y.min(), p.min()), max(y.max(), p.max())]
        axes[0].plot(extent, extent, "k--", linewidth=1)
        axes[0].set(xlabel=f"Cuff reference {target} (mmHg)", ylabel=f"Estimated {target} (mmHg)", title="Adaptive risk fusion, first declared seed")
        axes[1].scatter((p+y)/2, error, s=8, alpha=.25)
        for value in [error.mean(), error.mean()-1.96*error.std(ddof=1), error.mean()+1.96*error.std(ddof=1)]:
            axes[1].axhline(value, linestyle="--", linewidth=1)
        axes[1].set(xlabel="Mean of estimate and reference (mmHg)", ylabel="Estimate minus reference (mmHg)", title="Bland–Altman; repeated records, descriptive")
        save(fig, figures, f"agreement_{target.lower()}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, label in [("ecg_only_clean", "ECG"), ("ppg_only_clean", "PPG")]:
        history = pd.read_csv(output/"teachers"/"fold_0"/f"seed_{config['seeds'][0]}"/"crossfit_0"/name/"history.csv")
        axes[0].plot(history.epoch, history.train_regression, label=label)
        axes[1].plot(history.epoch, history.validation_clean_MAE, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Training Smooth L1 (mmHg scale)", title="First declared cross-fit teacher pair")
    axes[1].set(xlabel="Epoch", ylabel="Mean validation MAE (mmHg)", title="Teacher-only internal validation")
    axes[1].legend()
    save(fig, figures, "teacher_training")

    lines = ["# Task-specific reliability and quality-gating diagnostics", "",
             "**QUICK PIPELINE CHECK ONLY; not research findings.**" if config["quick"] else "**Executed post-hoc development study on a previously analyzed cohort. Not an independent confirmatory test.**", "",
             "## Design", "",
             "Historical test participants remain excluded. The same three participant-disjoint outer folds and train/validation roles as the reference factorial study are retained. Three initialization seeds are used in the full study.",
             "The deployable branch fuses frozen ECG-only and PPG-only BP predictions. This is a new late-fusion architecture; it is not a correction to the historical feature-fusion results or a fully jointly trained new neural architecture. All five policies use the exact same expert checkpoints, preprocessing, risk-head size, and availability masks. BP-only risk omits explicit quality inputs; the other policies share the full risk head.",
             "- Uniform weights over available sensors.", "- Estimated manual-signal-quality weights, shared between SBP/DBP.",
             "- Static task risk: training-only OOF mean expert absolute error for each modality/target.",
             "- BP-only adaptive task risk: same residual targets and Ridge procedure, using only expert BP predictions and sensor availability. The 18 SQI/automatic/manual-quality-estimator feature coefficients are exactly zero. This controls whether any benefit comes from explicit quality inputs rather than BP predictions alone.",
             "- Adaptive task risk: inverse predicted absolute error for each modality/target. Missing sensors always have zero weight; both absent causes abstention.", "",
             "## Strict training-only risk targets", "",
             "For each outer fold and seed, only the original inner-training participants enter a separate three-fold participant cross-fit. Each teacher reserves its own internal validation participants from the remaining training pool. Every risk-target record is absent from both the teacher fitting and teacher checkpoint selection that generated its BP prediction.",
             "Teachers are clean-trained with the same residual encoders/loss/quality supervision as the reference unimodal models. Their held-out training predictions are evaluated on clean waveforms and five training corruption families with independent draws from outer evaluation. BP references are unchanged. The absent expert has no observed risk target and is excluded from that head's fitting.",
             f"Risk features: bounded SQIs, waveform automatic quality, estimated manual goodness, expert SBP/DBP predictions in mmHg, and availability indicators ({24} features). No human quality label, true BP, residual, participant ID, age or sex is an inference input.",
             f"StandardScaler and a multi-output Ridge head (fixed alpha={config['ridge_alpha']}) fit only these cross-fitted training features/errors. Sample mass is half clean, half spread over five known corruptions; unavailable targets are excluded. Linear risk predictions are floored at {config['risk_floor_mmHg']} mmHg for numerical stability, not as a clinical accuracy claim. There is no hyperparameter search or fit on original validation/outer residuals.",
             "Final reference experts are fit on the original inner-training participants and selected with the original inner validation. They have more training participants than the cross-fit teachers; this creates a possible risk-calibration transfer mismatch. The risk head is deliberately small and fixed, not claimed to be optimal.",
             "All heads, checkpoints and validation-only rejection thresholds are sealed before every new outer evaluation and oracle/shuffle diagnostic. The prior cohort has already informed method design, so this freeze does not make it a new blind test.", "",
             "## Clean OOF accuracy", "", "Mean across declared initialization seeds; all outer folds pooled per seed. Errors in mmHg. Oracle uses manual test annotations and is nondeployable.", "",
             "| Policy | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | Parameters | Latency ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        parameters = str(int(row.parameters)) if np.isfinite(row.parameters) else "diagnostic"
        latency = f"{row.latency_ms:.3f}" if np.isfinite(row.latency_ms) else "not benchmarked"
        lines.append(f"| {LABELS[row.model]} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} | {row.SBP_RMSE:.3f} | {row.DBP_RMSE:.3f} | {parameters} | {latency} |")
    lines += ["", "## Primary evidence for adaptive reliability", "",
              "The four primary comparisons are full adaptive risk minus BP-only adaptive risk, for clean and mean five known corruption conditions, separately SBP/DBP. Negative favors adding explicit quality information. Participant-bootstrap percentile intervals are conditional on fitted OOF predictions and Bonferroni-adjusted over these four comparisons (98.75% each, approximately 95% family coverage). They do not capture all fitting/model-selection uncertainty or dependence from overlapping fold training sets.", ""]
    for row in primary.itertuples():
        status = "supports a conditional development improvement" if row.adjusted_upper < 0 else ("supports a conditional development deterioration" if row.adjusted_lower > 0 else "adaptive benefit not established; not evidence of equivalence")
        lines.append(f"- {row.condition}, {row.target}: {row.delta_MAE:+.3f} mmHg, adjusted interval [{row.adjusted_lower:+.3f}, {row.adjusted_upper:+.3f}]; {status}.")
    lines += ["", "Other fusion contrasts, natural-quality strata, individual synthetic stress conditions, oracle results, and selective prediction are exploratory. Their ordinary 95% intervals are not corrected for all secondary comparisons.", "",
              "## Gate diagnostics", "",
              "The original feature model D is frozen. Its learned gate is reproduced against saved OOF predictions within 0.001 mmHg before substituting uniform, manual-quality, or shuffled gates. Five fixed random permutations shuffle whole goodness vectors within availability strata; losses are averaged over shuffles and seeds, never ensembled predictions. These are post-training sensitivity tests, not matched retraining. Changing gates can create a representation-distribution shift.", ""]
    for row in feature_contrasts.itertuples():
        lines.append(f"- {row.comparison}, {row.target}: delta MAE {row.delta_MAE:+.3f}, exploratory CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}] mmHg.")
    lines += ["", "The separate late-fusion manual oracle applies manual goodness to the same frozen experts without a learned fusion regressor. It is a diagnostic, not deployable inference and not a guaranteed upper bound: manual visual quality need not match BP information utility. Both oracle analyses use original clean records only. Clean manual annotations are never reused as ground truth after synthetic noise.", ""]
    for row in contrasts[contrasts.comparison.str.startswith("late_manual_oracle")].itertuples():
        lines.append(f"- {row.comparison}, {row.target}: {row.delta_MAE:+.3f}, exploratory CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}] mmHg.")
    lines += ["", "## Risk estimates and abstention", "",
              "Individual expert risk estimates are absolute-error predictions in mmHg, not visual-quality scores or probabilities. Their weighted average is a fused-error proxy; it is not the true error of the fused prediction and not a calibrated uncertainty bound. Both signals can have high estimated risk even if normalized fusion weights are 50/50.",
              "On the same adaptive BP predictor, compare low-risk retention versus high-visual-quality retention. Thresholds come only from clean inner-validation score quantiles at 100/90/70/50% target coverage; outer BP errors do not select thresholds. Inclusive ties and distribution shift can change actual coverage. Random rejection matches accepted counts within each outer fold; its Monte Carlo SD is not a participant confidence interval.", "",
              "| Selector | Validation target | Actual clean coverage | Accepted SBP MAE | Accepted DBP MAE |", "|---|---:|---:|---:|---:|"]
    for selector in ["risk", "quality"]:
        for coverage in config["selective_coverages"]:
            data = selective[(selective.condition == "clean") & (selective.selector == selector) & (selective.validation_target_coverage == coverage)].groupby("target").mean(numeric_only=True)
            lines.append(f"| {selector} | {coverage:.0%} | {data.actual_record_coverage.mean():.1%} | {data.loc['SBP','selective_MAE']:.3f} | {data.loc['DBP','selective_MAE']:.3f} |")
    lines += ["", "All operating points, noisy-condition coverage, error-estimation calibration/ranking, and subgroup retention/errors are saved. BP/age bins are descriptive, not clinical classifications. Lower accepted-only error can reflect rejection of high-pressure/difficult participants; full-coverage performance must remain visible. Missing or small demographic subgroups limit any fairness interpretation.", "",
              "## Computational scope and limits", "",
              "- All five policy checkpoints retain and compute a risk head of the same size for matched diagnostics, including controls that do not use it to choose weights. BP-only risk uses six active feature coefficients per output; the other 18 are zero. Reported cost is this implementation, not an optimized minimal uniform-fusion deployment. Final expert weights are frozen during head fitting; full/BP-only heads are fitted by Ridge and exported to PyTorch linear layers. Count newly trained teachers and the two heads separately, not as independent training of five expert pairs.",
              "- MACs include Conv/Linear/GRU matrix operations only; approximate FLOPs are twice counted MACs. Latency is float32 on the recorded device, batch 1/32, with recorded warm-up/repetitions/threads. RSS includes loaded runtime and sampled benchmark allocations, not training or model-only memory. Do not compare runtimes from different devices or claim streaming validation from offline 24-second inputs.",
              "- Convex late fusion stays within the two experts' prediction ranges for each target; it cannot recover BP information missing from both experts. Risk estimates can be miscalibrated under population or sensor shift and can also be wrong when both experts fail.",
              "- The local QC cohort has missing metadata-listed waveforms and excluded ambiguous duplicate signals. Manual annotations partly describe windows outside the released crop. Digital amplitudes are not calibrated physiological amplitudes; cuff references are per-record, not continuous beat-to-beat targets.",
              "- Historical original test subjects are retired. Existing development subjects were already analyzed; no reshuffle of them constitutes new confirmatory evidence. New subjects/external device cohorts and prospective/clinical validation remain untested. No external dataset was downloaded or fabricated.",
              "- This work tests the proposed correction and mechanisms. It does not guarantee lower MAE, usefulness of quality gating, novelty, journal acceptance or clinical suitability. Existing negative results are preserved.", "",
              "## Reproduction", "", "`.venv/bin/python -m src.reliability --quick` is a smoke test. `.venv/bin/python -m src.reliability` runs the full fixed protocol. Use a new `--output` for changed source/config. `--phase fit` stops at the selection lock. Checkpoints, cross-fit participant roles, residual targets, scaler/head coefficients, validation thresholds, predictions and source snapshots are saved.", "",
              "Cross-fitting background: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html",
              "Ridge implementation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html", ""]
    (output/"RESEARCH_REPORT.md").write_text("\n".join(lines))
    print(comparison[["model", "SBP_MAE", "DBP_MAE", "parameters", "latency_ms"]].to_string(index=False))
