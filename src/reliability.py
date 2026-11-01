"""Post-hoc development study: gate interventions and training-only cross-fitted BP risk."""
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch

from .dataset_loader import QUALITY, TARGETS, prepare_dataset
from .evaluate import cluster_bootstrap, quality_strata, regression_metrics
from .factorial import CONDITIONS, demographic_strata, metric_rows
from .optimization import KNOWN_CONDITIONS, build_banks, fit_new_model, nested_subject_splits, subset
from .preprocessing import DEFAULT_PREPROCESSING, scale_sqi
from .risk_model import FEATURE_NAMES, POLICIES, RiskFusion, fusion_weights, predict_components
from .train import load_model
from .utils import code_hash, environment, save_json

LOGGER = logging.getLogger(__name__)
EXPERT_NAMES = ["ecg_only_clean", "ppg_only_clean"]
TRAIN_CONDITIONS = ["clean"] + KNOWN_CONDITIONS
COMPARISONS = {"adaptive_minus_static": ("adaptive_risk_fusion", "static_risk_fusion"),
               "adaptive_minus_bp_only": ("adaptive_risk_fusion", "bp_only_risk_fusion"),
               "adaptive_minus_uniform": ("adaptive_risk_fusion", "uniform_fusion"),
               "adaptive_minus_quality": ("adaptive_risk_fusion", "quality_fusion")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_path(reference, fold, name, seed):
    return Path(reference)/"fits"/f"fold_{fold}"/name/f"seed_{seed}"/"best.pt"


def pair_from_checkpoints(paths, config):
    loaded = [load_model(p, config["device"]) for p in paths]
    model = RiskFusion([c["scalers"] for _, c in loaded], risk_floor=config["risk_floor_mmHg"]).to(config["device"])
    for c, (expert, _) in enumerate(loaded):
        model.experts[c].load_state_dict(expert.state_dict())
    model.eval()
    return model, [c["scalers"] for _, c in loaded]


def crossfit_partitions(local_cohort, config, outer_fold):
    return nested_subject_splits(local_cohort, {"outer_folds": config["crossfit_folds"],
        "protocol_seed": config["crossfit_seed"]+outer_fold, "inner_validation_fraction": config["inner_validation_fraction"]})


def generate_oof_risk_data(cohort, clean, banks, split, outer_fold, seed, config, output, run_hash):
    directory = output/"risk_training"/f"fold_{outer_fold}"/f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"complete.json").exists():
        with np.load(directory/"oof_risk_data.npz", allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    global_indices = split["indices"]["train"]
    local = cohort.iloc[global_indices].reset_index(drop=True)
    local_clean = subset(clean, global_indices)
    folds = crossfit_partitions(local, config, outer_fold)
    features = np.full((len(TRAIN_CONDITIONS), len(local), len(FEATURE_NAMES)), np.nan, np.float32)
    expert_predictions = np.full((len(TRAIN_CONDITIONS), len(local), 2, 2), np.nan, np.float32)
    availability = np.zeros((len(TRAIN_CONDITIONS), len(local), 2), bool)
    source_folds = np.full(len(local), -1)
    teacher_manifest = []
    for crossfold, fold in enumerate(folds):
        paths = []
        for name in EXPERT_NAMES:
            teacher_dir = output/"teachers"/f"fold_{outer_fold}"/f"seed_{seed}"/f"crossfit_{crossfold}"/name
            path = teacher_dir/"best.pt"
            provenance = {"run_sha256": run_hash, "phase": "training_only_crossfit_teacher", "outer_fold": outer_fold,
                          "crossfit_fold": crossfold, "participant_ids": {k: v for k, v in fold["participant_ids"].items() if k != "outer"},
                          "heldout_subject_ids": fold["participant_ids"]["outer"]}
            if not (teacher_dir/"training_complete.json").exists():
                teacher_config = {**config, "architecture_version": "factorial_v1", "training_augmentation": False,
                                  "validation_corrupted_weight": 0., "ranking_weight": 0.}
                # Clean-only fitting does not access perturbations; these SQIs come from local training records.
                clean_sqi_bank = {stage: {"sqi": local_clean["sqi"][None]} for stage in ["training", "evaluation"]}
                fit_new_model(name, seed, local, local_clean, clean_sqi_bank, fold["indices"], teacher_config, teacher_dir, provenance)
            paths.append(path)
            teacher_manifest.append({"checkpoint": str(path), "sha256": sha(path), **provenance})
        model, _ = pair_from_checkpoints(paths, config)
        local_indices = fold["indices"]["outer"]
        global_heldout = global_indices[local_indices]
        source_folds[local_indices] = crossfold
        for variant, condition in enumerate(TRAIN_CONDITIONS):
            inputs = subset(clean, global_heldout) if variant == 0 else subset(banks["training"], global_heldout, variant-1)
            result = predict_components(model, inputs, {"sqi_mode": "bounded_v2"}, config["device"], config["batch_size"])
            features[variant, local_indices] = result["features"]
            expert_predictions[variant, local_indices] = result["expert_prediction"]
            availability[variant, local_indices] = result["available"]
        LOGGER.info("Training-only OOF residuals outer=%d seed=%d crossfit=%d (%d held-out participants)", outer_fold, seed, crossfold, len(fold["participant_ids"]["outer"]))
    if not np.isfinite(features).all() or not np.isfinite(expert_predictions).all() or (source_folds < 0).any():
        raise AssertionError("Incomplete OOF risk targets")
    actual = local[TARGETS].to_numpy(np.float32)
    observed_risk = np.abs(expert_predictions-actual[None, :, None, :])
    observed_risk = np.where(availability[..., None], observed_risk, np.nan)
    data = {"features": features, "expert_prediction": expert_predictions, "available": availability,
            "observed_absolute_error": observed_risk, "actual": actual, "crossfit_fold": source_folds}
    np.savez_compressed(directory/"oof_risk_data.npz", **data)
    local[["record_id", "subject_id"]+TARGETS].to_csv(directory/"risk_training_records.csv", index=False)
    save_json(directory/"teacher_provenance.json", teacher_manifest)
    save_json(directory/"complete.json", {"run_sha256": run_hash, "oof_sha256": sha(directory/"oof_risk_data.npz"),
              "risk_subject_ids": sorted(local.subject_id.unique()), "conditions": TRAIN_CONDITIONS,
              "validation_or_outer_targets_used_to_fit_risk": False, "missing_expert_errors_are_undefined_and_excluded": True})
    return data


def fit_risk_head(model, data, config, feature_indices=None):
    features = data["features"].reshape(-1, len(FEATURE_NAMES)).astype(np.float64)
    available = data["available"].reshape(-1, 2)
    observed = data["observed_absolute_error"].reshape(-1, 2, 2)
    weights = np.repeat([config["risk_clean_sample_weight"]]+[(1-config["risk_clean_sample_weight"])/len(KNOWN_CONDITIONS)]*len(KNOWN_CONDITIONS), data["features"].shape[1])
    scaler = StandardScaler().fit(features, sample_weight=weights)
    transformed = scaler.transform(features)
    selected_features = list(range(len(FEATURE_NAMES))) if feature_indices is None else list(feature_indices)
    coefficients, intercepts, static = [], [], []
    for c in range(2):
        keep = available[:, c]
        if not np.isfinite(observed[keep, c]).all() or not keep.any():
            raise ValueError("Missing or nonfinite cross-fitted expert errors")
        ridge = Ridge(alpha=config["ridge_alpha"], solver="svd").fit(transformed[keep][:, selected_features], observed[keep, c], sample_weight=weights[keep])
        expanded = np.zeros((2, len(FEATURE_NAMES)))
        expanded[:, selected_features] = ridge.coef_
        coefficients.extend(expanded)
        intercepts.extend(ridge.intercept_)
        static.append(np.average(observed[keep, c], weights=weights[keep], axis=0))
    with torch.no_grad():
        for target, value in [(model.feature_mean, scaler.mean_), (model.feature_scale, scaler.scale_),
                              (model.risk_head.weight, np.asarray(coefficients)), (model.risk_head.bias, np.asarray(intercepts)),
                              (model.static_risks, np.maximum(static, config["risk_floor_mmHg"]))]:
            target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))
    return {"feature_mean": scaler.mean_.tolist(), "feature_scale": scaler.scale_.tolist(),
            "ridge_alpha": config["ridge_alpha"], "sample_weights_per_condition": dict(zip(TRAIN_CONDITIONS, weights.reshape(len(TRAIN_CONDITIONS), -1)[:, 0].tolist(), strict=True)),
            "feature_names": FEATURE_NAMES, "active_feature_indices": selected_features, "static_risks_mmHg": np.asarray(static).tolist(),
            "fit_target": "absolute mmHg error of participant-held-out training-only experts", "parameter_search": False}


def lock_all_heads(cohort, clean, banks, folds, config, output, reference, run_hash):
    path = output/"selection_locked.json"
    if path.exists():
        return json.loads(path.read_text())
    selections = []
    for fold, split in enumerate(folds):
        for seed in config["seeds"]:
            data = generate_oof_risk_data(cohort, clean, banks, split, fold, seed, config, output, run_hash)
            expert_paths = [reference_path(reference, fold, name, seed) for name in EXPERT_NAMES]
            model, scalers = pair_from_checkpoints(expert_paths, config)
            fit_info = fit_risk_head(model, data, config)
            full_state = copy.deepcopy(model.state_dict())
            bp_only_info = fit_risk_head(model, data, config, feature_indices=range(18, 24))
            bp_only_state = copy.deepcopy(model.state_dict())
            directory = output/"fits"/f"fold_{fold}"/f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            provenance = {"run_sha256": run_hash, "phase": "risk_head_training_only", "outer_fold": fold,
                          "risk_fit_subject_ids": split["participant_ids"]["train"],
                          "reference_experts": [{"path": str(p), "sha256": sha(p)} for p in expert_paths],
                          "risk_targets": "training-only participant cross-fitting; never in-sample residuals"}
            checkpoints = {}
            for policy in POLICIES:
                checkpoint = {"architecture_version": "risk_v1", "model_name": policy, "state_dict": bp_only_state if policy == "bp_only_risk_fusion" else full_state,
                              "expert_scalers": scalers, "scalers": {"sqi_mode": "bounded_v2", "y_mean": [0., 0.], "y_std": [1., 1.]},
                              "config": config, "seed": seed, "preprocessing": DEFAULT_PREPROCESSING, "provenance": provenance}
                target = directory/f"{policy}.pt"
                torch.save(checkpoint, target)
                checkpoints[policy] = {"path": str(target), "sha256": sha(target)}
            model.policy = "adaptive_risk_fusion"
            model.load_state_dict(full_state)
            indices = split["indices"]["validation"]
            validation = predict_components(model, subset(clean, indices), {"sqi_mode": "bounded_v2"}, config["device"], config["batch_size"])
            scores = {"risk": validation["risk_proxy"].mean(1), "quality": validation["goodness"].min(1)}
            thresholds = {selector: {str(c): None if c == 1 else float(np.quantile(values, c if selector == "risk" else 1-c))
                                    for c in config["selective_coverages"]} for selector, values in scores.items()}
            save_json(directory/"risk_fit.json", fit_info)
            save_json(directory/"bp_only_risk_fit.json", bp_only_info)
            np.savez_compressed(directory/"inner_validation_components.npz", **validation)
            save_json(directory/"thresholds.json", thresholds)
            selections.append({"fold": fold, "seed": seed, "checkpoints": checkpoints, "thresholds": thresholds,
                               "risk_data_sha256": sha(output/"risk_training"/f"fold_{fold}"/f"seed_{seed}"/"oof_risk_data.npz")})
            LOGGER.info("Risk head and validation-only thresholds sealed fold=%d seed=%d", fold, seed)
    lock = {"run_sha256": run_hash, "source_sha256": code_hash(), "selections": selections,
            "all_risk_targets_from_training_only_crossfit": True, "outer_predictions_read": False,
            "risk_head_hyperparameters_fixed_before_training": True, "validation_targets_fit_risk_head": False}
    save_json(path, lock)
    return lock


def risk_frame(records, result, policy, condition, fold):
    frame = records[["record_id", "subject_id"]+TARGETS+(QUALITY if condition == "clean" else [])].copy().reset_index(drop=True)
    frame["outer_fold"], frame["condition"], frame["policy"] = fold, condition, policy
    for c, target in enumerate(["sbp", "dbp"]):
        frame[target+"_predicted"] = result["prediction"][:, c]
        frame[target+"_error_proxy"] = result["risk_proxy"][:, c]
        for m, modality in enumerate(["ecg", "ppg"]):
            frame[f"{modality}_{target}_weight"] = result["weights"][:, m, c]
            frame[f"{modality}_{target}_estimated_risk"] = np.where(result["available"][:, m], result["risk"][:, m, c], np.nan)
            frame[f"{modality}_{target}_expert_prediction"] = np.where(result["available"][:, m], result["expert_prediction"][:, m, c], np.nan)
    for m, modality in enumerate(["ecg", "ppg"]):
        frame[modality+"_estimated_goodness"] = result["goodness"][:, m]
        frame[modality+"_available"] = result["available"][:, m]
    return frame


def evaluate_risk_heads(cohort, clean, banks, folds, config, output, root, lock):
    seal = sha(output/"selection_locked.json")
    for choice in lock["selections"]:
        fold, seed = choice["fold"], choice["seed"]
        indices = folds[fold]["indices"]["outer"]
        records = cohort.iloc[indices].reset_index(drop=True)
        for policy in POLICIES:
            directory = output/"outer"/f"fold_{fold}"/f"seed_{seed}"/policy
            if (directory/"complete.json").exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            path = choice["checkpoints"][policy]["path"]
            if sha(path) != choice["checkpoints"][policy]["sha256"]:
                raise ValueError("Risk checkpoint changed after lock")
            model, checkpoint = load_model(path, config["device"])
            for variant, condition in enumerate(CONDITIONS):
                inputs = subset(clean, indices) if variant == 0 else subset(banks["evaluation"], indices, variant-1)
                result = predict_components(model, inputs, checkpoint["scalers"], config["device"], config["batch_size"])
                risk_frame(records, result, policy, condition, fold).to_csv(directory/f"predictions_{condition}.csv", index=False)
                if condition == "clean" and policy == "quality_fusion":
                    oracle_weights = fusion_weights("quality_fusion", torch.tensor(1-records[QUALITY].to_numpy(np.float32)/3),
                        torch.tensor(result["risk"]), model.static_risks.cpu(), torch.tensor(result["available"])).numpy()
                    oracle = {**result, "weights": oracle_weights, "prediction": (oracle_weights*result["expert_prediction"]).sum(1),
                              "risk_proxy": (oracle_weights*result["risk"]).sum(1)}
                    risk_frame(records, oracle, "manual_quality_oracle", condition, fold).to_csv(directory/"predictions_manual_quality_oracle.csv", index=False)
            subprocess.run([sys.executable, "-m", "src.efficiency", str(path), str(root), records.iloc[0].wfdb_record_path, str(directory/"efficiency.json"),
                            "--threads", str(config["threads"]), "--warmup", str(config["latency_warmup"]),
                            "--repetitions", str(config["latency_repetitions"]), "--device", config["device"]], check=True)
            save_json(directory/"complete.json", {"selection_lock_sha256": seal, "checkpoint_sha256": sha(path), "outer_records": len(indices)})
            LOGGER.info("Outer risk fusion evaluated fold=%d seed=%d policy=%s", fold, seed, policy)


def shuffle_goodness(goodness, available, seed):
    result = np.asarray(goodness).copy()
    patterns = np.asarray(available, int) @ np.asarray([2, 1])
    rng = np.random.default_rng(seed)
    for pattern in np.unique(patterns):
        indices = np.flatnonzero(patterns == pattern)
        result[indices] = goodness[rng.permutation(indices)]
    return result


def feature_gate_interventions(cohort, clean, folds, config, output, reference):
    for fold, split in enumerate(folds):
        indices = split["indices"]["outer"]
        records = cohort.iloc[indices].reset_index(drop=True)
        inputs = subset(clean, indices)
        for seed in config["seeds"]:
            directory = output/"feature_diagnostics"/f"fold_{fold}"/f"seed_{seed}"
            if (directory/"complete.json").exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            model, checkpoint = load_model(reference_path(reference, fold, "D_gated_augmented", seed), config["device"])
            sqi = scale_sqi(inputs["sqi"], checkpoint["scalers"])
            embeddings, goodness, available = [], [], []
            with torch.inference_mode():
                for start in range(0, len(records), config["batch_size"]):
                    x = torch.tensor(inputs["x"][start:start+config["batch_size"]], device=config["device"])
                    s = torch.tensor(sqi[start:start+config["batch_size"]], device=config["device"])
                    z = [encoder(x[:, c:c+1]) for c, encoder in enumerate(model.encoders)]
                    a = x.abs().amax(-1) > 1e-6
                    q = torch.stack([module(z[c], s[:, c]) for c, module in enumerate(model.quality)], 1).masked_fill(~a, -20.).sigmoid()
                    embeddings.append(torch.stack(z, 1).cpu().numpy())
                    goodness.append(q.cpu().numpy())
                    available.append(a.cpu().numpy())
            embeddings, goodness, available = np.concatenate(embeddings), np.concatenate(goodness), np.concatenate(available)
            policies = {"learned_gate": goodness, "uniform_intervention": available.astype(np.float32),
                        "manual_quality_oracle": 1-records[QUALITY].to_numpy(np.float32)/3}
            policies.update({f"shuffled_gate_{j}": shuffle_goodness(goodness, available, config["shuffle_seed"]+fold*1000+seed*10+j)
                             for j in range(config["shuffle_repetitions"])})
            for name, quality in policies.items():
                weights = np.maximum(quality, .05)*available
                weights /= weights.sum(1, keepdims=True)
                ecg, ppg = [embeddings[:, c]*(2*weights[:, c:c+1]) for c in range(2)]
                fused = np.concatenate([ecg, ppg, ecg*ppg, np.abs(ecg-ppg)], axis=1)
                with torch.inference_mode():
                    p = model.regressor(torch.tensor(fused, device=config["device"])).cpu().numpy()
                p = p*np.asarray(checkpoint["scalers"]["y_std"]) + np.asarray(checkpoint["scalers"]["y_mean"])
                frame = records[["record_id", "subject_id"]+TARGETS+QUALITY].copy()
                frame["sbp_predicted"], frame["dbp_predicted"] = p[:, 0], p[:, 1]
                frame["ecg_weight"], frame["ppg_weight"] = weights[:, 0], weights[:, 1]
                frame["outer_fold"], frame["condition"] = fold, "clean"
                frame.to_csv(directory/f"{name}.csv", index=False)
                if name == "learned_gate":
                    original = pd.read_csv(Path(reference)/"outer"/f"fold_{fold}"/"D_gated_augmented"/f"seed_{seed}"/"predictions_clean.csv")
                    original = original.set_index("record_id").loc[frame.record_id, ["sbp_predicted", "dbp_predicted"]].to_numpy()
                    np.testing.assert_allclose(p, original, rtol=0, atol=.001)
            save_json(directory/"complete.json", {"selection_lock_sha256": sha(output/"selection_locked.json"), "clean_only": True,
                      "interpretation": "post-training sensitivity intervention, not a matched retraining; oracle is nondeployable and not a guaranteed upper bound"})


def conditional_contrasts(losses, subjects, comparisons, condition, config, primary=False):
    groups, inverse = np.unique(subjects, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(config["protocol_seed"]).integers(len(groups), size=(config["bootstrap_repetitions"], len(groups)))
    rows = []
    for name, (first, second) in comparisons.items():
        delta = losses[first]-losses[second]
        for c, target in enumerate(["SBP", "DBP"]):
            sums = np.bincount(inverse, weights=delta[:, c])
            values = sums[draws].sum(1)/counts[draws].sum(1)
            is_primary = primary and name == config["primary_comparison"] and condition in config["primary_conditions"]
            family = len(config["primary_conditions"])*len(config["primary_targets"])
            alpha = .05/family
            low, high = np.quantile(values, [.025, .975])
            adjusted = np.quantile(values, [alpha/2, 1-alpha/2]) if is_primary else [None, None]
            rows.append({"comparison": name, "condition": condition, "target": target, "delta_MAE": float(delta[:, c].mean()),
                         "lower_95": float(low), "upper_95": float(high), "primary": is_primary,
                         "adjusted_lower": adjusted[0], "adjusted_upper": adjusted[1], "adjusted_confidence": 1-alpha if is_primary else None,
                         "scope": "conditional participant bootstrap of seed-averaged record losses, not an ensemble; overlapping fold training uncertainty omitted"})
    return rows


def collect_results(cohort, folds, config, output):
    predictions, losses, metrics, strata, uncertainty, efficiency, risk_diagnostics, calibration = {}, {}, [], [], [], [], [], []
    for policy in POLICIES+["manual_quality_oracle"]:
        for seed in config["seeds"]:
            for condition in (CONDITIONS if policy in POLICIES else ["clean"]):
                frames = []
                for fold in range(len(folds)):
                    directory = output/"outer"/f"fold_{fold}"/f"seed_{seed}"/(policy if policy in POLICIES else "quality_fusion")
                    filename = f"predictions_{condition}.csv" if policy in POLICIES else "predictions_manual_quality_oracle.csv"
                    frames.append(pd.read_csv(directory/filename))
                    if condition == "clean" and policy in POLICIES:
                        e = json.loads((directory/"efficiency.json").read_text())
                        efficiency.append({"model": policy, "seed": seed, "outer_fold": fold, "parameters": e["parameters"], "MACs": e["MACs_per_record"],
                                           "latency_ms": e["model_only"]["batch_1"]["median_latency_ms"], "throughput_batch32_per_s": e["model_only"]["batch_32"]["records_per_second"],
                                           "end_to_end_ms": e["end_to_end_median_ms"], "peak_RSS_MiB": e["isolated_process_peak_RSS_MiB"],
                                           "serialized_weights_KiB": e["serialized_state_dict_bytes"]/1024, "device": e["device_name"]})
                frame = pd.concat(frames, ignore_index=True).sort_values("record_id").reset_index(drop=True)
                if frame.record_id.duplicated().any() or set(frame.record_id) != set(cohort.record_id):
                    raise AssertionError("Risk OOF predictions must cover each development record exactly once")
                frame.to_csv(output/"predictions"/f"{policy}_seed_{seed}_{condition}.csv", index=False)
                predictions.setdefault(condition, {}).setdefault(policy, []).append(frame)
                loss = np.abs(frame[["sbp_predicted", "dbp_predicted"]].to_numpy()-frame[TARGETS].to_numpy())
                losses.setdefault(condition, {}).setdefault(policy, []).append(loss)
                metrics.extend(metric_rows(frame, policy, seed, condition))
                if condition == "clean":
                    p = frame[["sbp_predicted", "dbp_predicted"]].to_numpy()
                    strata.extend({"model": policy, "seed": seed, **row} for row in quality_strata(frame, p))
                    uncertainty.extend({"model": policy, "seed": seed, **row} for row in cluster_bootstrap(frame[TARGETS].to_numpy(), p, frame.subject_id.to_numpy(), config["bootstrap_repetitions"], config["protocol_seed"]))
                if policy == "adaptive_risk_fusion":
                    from scipy.stats import spearmanr
                    for modality in ["ecg", "ppg"]:
                        available = frame[modality+"_available"].to_numpy(bool)
                        for target in ["sbp", "dbp"]:
                            if not available.any():
                                continue
                            estimated = frame.loc[available, f"{modality}_{target}_estimated_risk"].to_numpy()
                            observed = np.abs(frame.loc[available, f"{modality}_{target}_expert_prediction"].to_numpy()-frame.loc[available, target+"_mmHg"].to_numpy())
                            rho = float(spearmanr(estimated, observed).statistic) if np.std(estimated) > 0 and np.std(observed) > 0 else None
                            risk_diagnostics.append({"seed": seed, "condition": condition, "modality": modality, "target": target.upper(),
                                "n": len(estimated), "predicted_risk_mean": float(estimated.mean()), "observed_absolute_error_mean": float(observed.mean()),
                                "risk_estimation_MAE_mmHg": float(np.abs(estimated-observed).mean()), "Spearman_rho": rho})
                            bins = np.digitize(estimated, [5, 10, 15, 20, 30, 50, 100])
                            for band in np.unique(bins):
                                selected = bins == band
                                calibration.append({"seed": seed, "condition": condition, "modality": modality, "target": target.upper(), "fixed_error_band": int(band),
                                                    "n": int(selected.sum()), "predicted_risk_mean": float(estimated[selected].mean()),
                                                    "observed_error_mean": float(observed[selected].mean())})
    pd.DataFrame(metrics).to_csv(output/"tables/metrics_per_seed.csv", index=False)
    pd.DataFrame(strata).to_csv(output/"tables/quality_stratified.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output/"tables/participant_bootstrap_CI.csv", index=False)
    pd.DataFrame(efficiency).to_csv(output/"tables/efficiency_per_fold_seed.csv", index=False)
    pd.DataFrame(risk_diagnostics).to_csv(output/"tables/risk_estimator_diagnostics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output/"tables/risk_calibration.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["model", "condition", "target"]).agg(MAE=("MAE", "mean"), MAE_seed_SD=("MAE", "std"),
        RMSE=("RMSE", "mean"), R2=("R2", "mean"), ME=("ME", "mean"), SD_error=("SD_error", "mean"), participant_macro_MAE=("participant_macro_MAE", "mean")).reset_index()
    summary.to_csv(output/"tables/summary.csv", index=False)
    averaged = {c: {p: np.mean(values, axis=0) for p, values in models.items()} for c, models in losses.items()}
    averaged["known_corruption_mean"] = {p: np.mean([averaged[c][p] for c in KNOWN_CONDITIONS], axis=0) for p in POLICIES}
    subjects = predictions["clean"]["adaptive_risk_fusion"][0].subject_id.to_numpy()
    comparisons = []
    for condition, models in averaged.items():
        comparisons.extend(conditional_contrasts(models, subjects, COMPARISONS, condition, config, primary=True))
    comparisons.extend(conditional_contrasts(averaged["clean"], subjects,
        {"late_manual_oracle_minus_quality": ("manual_quality_oracle", "quality_fusion"), "late_manual_oracle_minus_uniform": ("manual_quality_oracle", "uniform_fusion")}, "clean", config))
    pd.DataFrame(comparisons).to_csv(output/"tables/fusion_contrasts.csv", index=False)
    return predictions, summary, pd.DataFrame(efficiency)


def collect_feature_diagnostics(cohort, folds, config, output):
    names = ["learned_gate", "uniform_intervention", "manual_quality_oracle"]+[f"shuffled_gate_{j}" for j in range(config["shuffle_repetitions"])]
    losses, metrics = {}, []
    subjects = None
    for name in names:
        for seed in config["seeds"]:
            frames = [pd.read_csv(output/"feature_diagnostics"/f"fold_{fold}"/f"seed_{seed}"/f"{name}.csv") for fold in range(len(folds))]
            frame = pd.concat(frames, ignore_index=True).sort_values("record_id").reset_index(drop=True)
            if frame.record_id.duplicated().any() or set(frame.record_id) != set(cohort.record_id):
                raise AssertionError("Incomplete feature-gate diagnostic coverage")
            subjects = frame.subject_id.to_numpy()
            loss = np.abs(frame[["sbp_predicted", "dbp_predicted"]].to_numpy()-frame[TARGETS].to_numpy())
            losses.setdefault(name, []).append(loss)
            metrics.extend(metric_rows(frame, name, seed, "clean"))
    averaged = {name: np.mean(values, axis=0) for name, values in losses.items()}
    averaged["shuffled_gate_mean_loss"] = np.mean([averaged[f"shuffled_gate_{j}"] for j in range(config["shuffle_repetitions"])], axis=0)
    comparisons = {"feature_manual_oracle_minus_learned": ("manual_quality_oracle", "learned_gate"),
                   "feature_uniform_intervention_minus_learned": ("uniform_intervention", "learned_gate"),
                   "feature_shuffled_minus_learned": ("shuffled_gate_mean_loss", "learned_gate")}
    pd.DataFrame(metrics).to_csv(output/"tables/feature_gate_metrics.csv", index=False)
    pd.DataFrame(conditional_contrasts(averaged, subjects, comparisons, "clean", config)).to_csv(output/"tables/feature_gate_contrasts.csv", index=False)


def selective_evaluation(predictions, config, lock, output, demographics):
    thresholds = {(r["fold"], r["seed"]): r["thresholds"] for r in lock["selections"]}
    metrics, subgroup_rows, decisions = [], [], []
    for condition in CONDITIONS:
        for seed, frame in zip(config["seeds"], predictions[condition]["adaptive_risk_fusion"], strict=True):
            frame = frame.merge(demographics[["subject_id", "age_years", "sex_code"]], on="subject_id", how="left", validate="many_to_one")
            errors = np.abs(frame[["sbp_predicted", "dbp_predicted"]].to_numpy()-frame[TARGETS].to_numpy())
            for selector in ["risk", "quality"]:
                score = frame[["sbp_error_proxy", "dbp_error_proxy"]].mean(axis=1).to_numpy() if selector == "risk" else frame[["ecg_estimated_goodness", "ppg_estimated_goodness"]].min(axis=1).to_numpy()
                for coverage in config["selective_coverages"]:
                    threshold = np.asarray([np.nan if coverage == 1 else thresholds[(int(f), seed)][selector][str(coverage)] for f in frame.outer_fold])
                    keep = np.ones(len(frame), bool) if coverage == 1 else (score <= threshold if selector == "risk" else score >= threshold)
                    rng = np.random.default_rng(config["protocol_seed"]+seed)
                    random_errors = []
                    for _ in range(config["random_rejection_repetitions"]):
                        selected = np.zeros(len(frame), bool)
                        for fold in range(config["outer_folds"]):
                            ix = np.flatnonzero(frame.outer_fold.to_numpy() == fold)
                            selected[rng.choice(ix, size=int(keep[ix].sum()), replace=False)] = True
                        random_errors.append(errors[selected].mean(0) if selected.any() else [np.nan, np.nan])
                    random_errors = np.asarray(random_errors)
                    common = {"selector": selector, "seed": seed, "condition": condition, "validation_target_coverage": coverage,
                              "actual_record_coverage": float(keep.mean()), "accepted_records": int(keep.sum()), "all_records": len(frame),
                              "accepted_participants": int(frame.loc[keep, "subject_id"].nunique())}
                    for c, target in enumerate(["SBP", "DBP"]):
                        metrics.append({**common, "target": target, "selective_MAE": float(errors[keep, c].mean()) if keep.any() else None,
                                        "all_records_MAE": float(errors[:, c].mean()), "random_rejection_MAE_mean": float(random_errors[:, c].mean()) if keep.any() else None,
                                        "random_rejection_MC_SD": float(random_errors[:, c].std(ddof=1)) if keep.any() else None})
                    if condition == "clean":
                        d = frame[["record_id", "subject_id", "outer_fold"]].copy()
                        d["seed"], d["selector"], d["validation_target_coverage"] = seed, selector, coverage
                        d["score"], d["threshold"], d["accepted"] = score, threshold, keep
                        decisions.append(d)
                        for group, mask in demographic_strata(frame).items():
                            mask = mask.to_numpy()
                            if not mask.any():
                                continue
                            accepted = keep & mask
                            for c, target in enumerate(["SBP", "DBP"]):
                                subgroup_rows.append({**common, "target": target, "subgroup": group, "subgroup_records": int(mask.sum()),
                                    "subgroup_accepted": int(accepted.sum()), "subgroup_coverage": float(keep[mask].mean()),
                                    "subgroup_all_MAE": float(errors[mask, c].mean()), "subgroup_selective_MAE": float(errors[accepted, c].mean()) if accepted.any() else None})
    pd.DataFrame(metrics).to_csv(output/"tables/selective_risk_coverage.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(output/"tables/selective_subgroups.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(output/"predictions/selective_clean_decisions.csv", index=False)


def run(args):
    extra = json.loads(Path(args.config).read_text())
    reference = Path(extra["reference_quick_study"] if args.quick else extra["reference_study"])
    reference_protocol = json.loads((reference/"protocol.json").read_text())
    config = {**reference_protocol["config"], **extra, "quick": args.quick, "reference": str(reference)}
    config["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    output = Path(args.output or ("results/reliability_smoke" if args.quick else "results/reliability"))
    for directory in [output, output/"logs", output/"tables", output/"predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output/"logs/reliability.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    torch.set_num_threads(config["threads"])
    for filename in ["factorial_model.py", "optimized_model.py", "preprocessing.py", "fusion.py"]:
        if sha(reference/"source_snapshot"/"src"/filename) != sha(Path("src")/filename):
            raise ValueError("Reference expert implementation changed")
    root = Path(args.data).resolve()
    cohort, clean, audit = prepare_dataset(root, args.cache, DEFAULT_PREPROCESSING)
    keep = np.flatnonzero(cohort.record_id.isin(reference_protocol["development_record_ids"]))
    cohort, clean = cohort.iloc[keep].reset_index(drop=True), subset(clean, keep)
    if cohort.record_id.tolist() != reference_protocol["development_record_ids"] or set(cohort.subject_id) & set(reference_protocol["retired_test_subject_ids"]):
        raise ValueError("Reference cohort or historical participant roles changed")
    folds = [{"participant_ids": fold["participant_ids"], "indices": {k: np.flatnonzero(cohort.subject_id.isin(v)) for k, v in fold["participant_ids"].items()}}
             for fold in reference_protocol["outer_folds"]]
    reference_lock = json.loads((reference/"selection_locked.json").read_text())
    expected = {(r["fold"], r["model"], r["seed"]): r["checkpoint_sha256"] for r in reference_lock["selections"]}
    sources = {}
    for fold in range(len(folds)):
        for name in EXPERT_NAMES+["D_gated_augmented"]:
            for seed in config["seeds"]:
                p = reference_path(reference, fold, name, seed)
                if sha(p) != expected[(fold, name, seed)]:
                    raise ValueError("Reference checkpoint changed")
                sources[str(p)] = sha(p)
    crossfits = [{"outer_fold": fold, "partitions": [{"crossfit_fold": j, "participant_ids": f["participant_ids"]}
                  for j, f in enumerate(crossfit_partitions(cohort.iloc[split["indices"]["train"]].reset_index(drop=True), config, fold))]}
                 for fold, split in enumerate(folds)]
    protocol = {"config": config, "source_sha256": code_hash(), "environment": environment(), "dataset_fingerprint": audit["fingerprint"],
                "development_subject_ids": sorted(cohort.subject_id.unique()), "development_record_ids": cohort.record_id.tolist(),
                "retired_test_subject_ids": reference_protocol["retired_test_subject_ids"], "reference_checkpoint_hashes": sources,
                "outer_folds": reference_protocol["outer_folds"], "training_only_crossfit_partitions": crossfits,
                "policies": POLICIES, "primary_family": "adaptive minus BP-only risk, clean and mean five known corruptions, SBP/DBP; four adjusted intervals",
                "quality_oracles": "clean original records only; manual annotations never supplied to deployable inference",
                "diagnostic_scope": "feature-gate oracle/uniform/shuffle are post-training interventions, not matched retraining or guaranteed performance ceilings",
                "risk_fitting": "fixed Ridge on participant-cross-fitted inner-training residuals, half clean/half known training corruptions; absent expert errors undefined",
                "risk_features": FEATURE_NAMES, "risk_head_hyperparameter_search": False,
                "absolute_risk_scope": "estimated error in mmHg; fused weighted error is a proxy, not a validated uncertainty bound",
                "prior_results_informed_design": True, "independent_confirmatory_test": False,
                "all_new_outer_evaluations_after_head_and_threshold_lock": True}
    run_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (output/"protocol.json").exists() and json.loads((output/"protocol.json").read_text())["run_sha256"] != run_hash:
        raise ValueError("Source/config/environment/data changed: use a new output directory")
    if not (output/"protocol.json").exists():
        save_json(output/"protocol.json", {"run_sha256": run_hash, **protocol})
        save_json(output/"resolved_config.json", config)
        cohort.to_csv(output/"tables/development_cohort.csv", index=False)
        snapshot = output/"source_snapshot"
        (snapshot/"src").mkdir(parents=True, exist_ok=True)
        (snapshot/"configs").mkdir(exist_ok=True)
        for p in Path("src").glob("*.py"):
            shutil.copy2(p, snapshot/"src"/p.name)
        for filename in ["main.py", "requirements.lock.txt", "pytest.ini"]:
            shutil.copy2(filename, snapshot/filename)
        shutil.copy2(args.config, snapshot/"configs"/Path(args.config).name)
    save_json(output/"status.json", {"status": "fitting_training_only_risk", "run_sha256": run_hash})
    banks = build_banks(root, cohort, config, Path(args.cache)/("factorial_smoke" if args.quick else "factorial"), audit["fingerprint"])
    lock = lock_all_heads(cohort, clean, banks, folds, config, output, reference, run_hash)
    if lock["source_sha256"] != code_hash():
        raise ValueError("Source changed during risk fitting")
    save_json(output/"status.json", {"status": "selection_locked" if args.phase == "fit" else "outer_evaluation", "run_sha256": run_hash})
    if args.phase == "fit":
        return
    feature_gate_interventions(cohort, clean, folds, config, output, reference)
    evaluate_risk_heads(cohort, clean, banks, folds, config, output, root, lock)
    predictions, summary, efficiency = collect_results(cohort, folds, config, output)
    collect_feature_diagnostics(cohort, folds, config, output)
    selective_evaluation(predictions, config, lock, output, pd.read_csv(root/"subject-info.csv"))
    from .reliability_report import render_report
    render_report(output, summary, efficiency, config, predictions)
    if code_hash() != protocol["source_sha256"]:
        raise ValueError("Source changed after protocol freeze")
    save_json(output/"status.json", {"status": "complete", "run_sha256": run_hash, "quick": args.quick,
        "development_records": len(cohort), "development_subjects": int(cohort.subject_id.nunique()),
        "retired_test_subjects": len(protocol["retired_test_subject_ids"]),
        "new_teacher_fits": len(folds)*len(config["seeds"])*config["crossfit_folds"]*2,
        "risk_fitting_pipelines": len(lock["selections"]), "risk_head_fits": 2*len(lock["selections"]),
        "outer_policy_evaluations": len(folds)*len(config["seeds"])*len(POLICIES),
        "independent_confirmatory_test": False, "clinical_validation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reliability.json")
    parser.add_argument("--data", default="physionet.org/files/cardiowave-portable/1.0.0")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--phase", choices=["all", "fit"], default="all")
    run(parser.parse_args())
