"""Frozen matched factorial development study; historical test participants stay retired."""
import argparse
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
from scipy.stats import spearmanr
import torch

from .dataset_loader import QUALITY, TARGETS, prepare_dataset
from .evaluate import cluster_bootstrap, prediction_frame, quality_strata, regression_metrics
from .factorial_model import CELLS, STUDY_MODELS, UNIMODAL
from .optimization import KNOWN_CONDITIONS, UNSEEN_CONDITIONS, build_banks, fit_new_model, nested_subject_splits, subset
from .preprocessing import DEFAULT_PREPROCESSING
from .train import load_model, predict
from .utils import code_hash, environment, save_json

LOGGER = logging.getLogger(__name__)
CONDITIONS = ["clean"] + KNOWN_CONDITIONS + UNSEEN_CONDITIONS
CONSTANTS = ["training_mean", "training_median"]
CONTRASTS = {
    "B_minus_A": {"B_gated_clean": 1, "A_uniform_clean": -1},
    "D_minus_C": {"D_gated_augmented": 1, "C_uniform_augmented": -1},
    "C_minus_A": {"C_uniform_augmented": 1, "A_uniform_clean": -1},
    "D_minus_B": {"D_gated_augmented": 1, "B_gated_clean": -1},
    "interaction_D_minus_C_minus_B_plus_A": {"D_gated_augmented": 1, "C_uniform_augmented": -1, "B_gated_clean": -1, "A_uniform_clean": 1},
    "multimodal_minus_ecg": {"A_uniform_clean": 1, "ecg_only_clean": -1},
    "multimodal_minus_ppg": {"A_uniform_clean": 1, "ppg_only_clean": -1},
    "separable_minus_dense": {"A_uniform_clean": 1, "dense_uniform_clean": -1},
    "D_minus_training_median": {"D_gated_augmented": 1, "training_median": -1},
    "A_minus_training_median": {"A_uniform_clean": 1, "training_median": -1},
}


def constant_prediction(training_targets, count, method):
    values = np.asarray(training_targets)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid training-only BP references")
    if method not in CONSTANTS:
        raise ValueError(method)
    center = values.mean(0) if method == "training_mean" else np.median(values, axis=0)
    return np.tile(center, (count, 1))


def quality_threshold(scores, coverage):
    scores = np.asarray(scores)
    if not 0 < coverage <= 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid quality threshold inputs")
    # Zero accepts every bounded goodness score at the explicitly reported 100% operating point.
    return 0. if coverage == 1 else float(np.quantile(scores, 1-coverage))


def demographic_strata(frame):
    # The dictionary top-codes age as '90+'; its lower bound suffices for the prespecified age bands.
    age = pd.to_numeric(frame.age_years.replace("90+", "90"), errors="raise")
    if not frame.sex_code.dropna().isin([0, 1, 2]).all():
        raise ValueError("Unknown questionnaire sex code")
    groups = {"SBP_<100": frame.sbp_mmHg < 100, "SBP_100_to_<140": frame.sbp_mmHg.between(100, 140, inclusive="left"),
              "SBP_>=140": frame.sbp_mmHg >= 140, "DBP_>=90": frame.dbp_mmHg >= 90,
              "age_<60": age < 60, "age_>=60": age >= 60, "age_missing": age.isna(),
              "sex_female": frame.sex_code == 0, "sex_male": frame.sex_code == 1,
              "sex_unavailable": frame.sex_code.isna() | (frame.sex_code == 2)}
    groups.update({f"worst_manual_poor_windows_{k}": frame[QUALITY].max(axis=1) == k for k in range(4)})
    return groups


def available_conditions(name):
    missing = "ecg_missing" if UNIMODAL.get(name) == 0 else "ppg_missing"
    return [c for c in CONDITIONS if name not in UNIMODAL or c != missing]


def fit_and_lock(cohort, clean, banks, folds, config, output, run_hash):
    path = output / "selection_locked.json"
    if path.exists():
        return json.loads(path.read_text())
    selections = []
    for fold, split in enumerate(folds):
        for name in STUDY_MODELS:
            for seed in config["seeds"]:
                directory = output / "fits" / f"fold_{fold}" / name / f"seed_{seed}"
                augmented = CELLS.get(name, (False, False))[1]
                training_config = {**config, "training_augmentation": augmented}
                provenance = {"run_sha256": run_hash, "phase": "inner_training_only", "fold": fold,
                              "participant_ids": {k: v for k, v in split["participant_ids"].items() if k != "outer"}}
                if not (directory / "training_complete.json").exists():
                    fit_new_model(name, seed, cohort, clean, banks, split["indices"], training_config, directory, provenance)
                model, checkpoint = load_model(directory / "best.pt", config["device"])
                indices = split["indices"]["validation"]
                p, q, gates = predict(model, subset(clean, indices), checkpoint["scalers"], config["device"], config["batch_size"])
                frame = prediction_frame(cohort.iloc[indices], p, q, gates)
                frame.to_csv(directory / "inner_validation_predictions.csv", index=False)
                thresholds = {str(c): quality_threshold(q.min(1), c) for c in config["selective_coverages"]} if name in CELLS else {}
                selections.append({"fold": fold, "model": name, "seed": seed, "best_epoch": checkpoint["epoch"],
                                   "validation_score": checkpoint["validation_score"], "quality_thresholds": thresholds,
                                   "checkpoint_sha256": hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()})
                LOGGER.info("Sealed inner fit fold=%d model=%s seed=%d epoch=%d", fold, name, seed, checkpoint["epoch"])
    lock = {"run_sha256": run_hash, "source_sha256": code_hash(), "selections": selections,
            "selection_objective": "clean inner-validation mean SBP/DBP MAE for every neural model",
            "quality_threshold_source": "clean inner validation goodness quantiles; BP errors and outer data not used",
            "outer_predictions_used": False, "hyperparameter_search": False}
    save_json(path, lock)
    return lock


def evaluate_locked(cohort, clean, banks, folds, config, output, lock, root, run_hash):
    seal = hashlib.sha256((output / "selection_locked.json").read_bytes()).hexdigest()
    for choice in lock["selections"]:
        fold, name, seed = choice["fold"], choice["model"], choice["seed"]
        directory = output / "outer" / f"fold_{fold}" / name / f"seed_{seed}"
        if (directory / "complete.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "fits" / f"fold_{fold}" / name / f"seed_{seed}" / "best.pt"
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != choice["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after selection lock")
        model, checkpoint = load_model(checkpoint_path, config["device"])
        indices = folds[fold]["indices"]["outer"]
        records = cohort.iloc[indices].reset_index(drop=True)
        for condition in available_conditions(name):
            inputs = subset(clean, indices) if condition == "clean" else subset(banks["evaluation"], indices, CONDITIONS.index(condition)-1)
            p, q, gates = predict(model, inputs, checkpoint["scalers"], config["device"], config["batch_size"])
            frame = prediction_frame(records, p, q, gates)
            if condition != "clean":
                frame = frame.drop(columns=QUALITY)
            frame["condition"], frame["outer_fold"] = condition, fold
            frame.to_csv(directory / f"predictions_{condition}.csv", index=False)
        save_json(directory / "abstention.json", {"unavailable_conditions": sorted(set(CONDITIONS)-set(available_conditions(name))),
                                                  "reason": "single-sensor model abstains when its only sensor is absent", "invented_predictions": False})
        subprocess.run([sys.executable, "-m", "src.efficiency", str(checkpoint_path), str(root), records.iloc[0].wfdb_record_path,
                        str(directory / "efficiency.json"), "--threads", str(config["threads"]), "--warmup", str(config["latency_warmup"]),
                        "--repetitions", str(config["latency_repetitions"]), "--device", config["device"]], check=True)
        save_json(directory / "complete.json", {"run_sha256": run_hash, "selection_lock_sha256": seal,
                                                "checkpoint_sha256": choice["checkpoint_sha256"], "outer_records": len(indices)})
        LOGGER.info("Outer evaluation completed fold=%d model=%s seed=%d", fold, name, seed)


def metric_rows(frame, model, seed, condition):
    actual, predicted = frame[TARGETS].to_numpy(), frame[["sbp_predicted", "dbp_predicted"]].to_numpy()
    rows = []
    for c, row in enumerate(regression_metrics(actual, predicted)):
        error = np.abs(predicted[:, c]-actual[:, c])
        rows.append({"model": model, "seed": seed, "condition": condition,
                     "participant_macro_MAE": float(pd.Series(error).groupby(frame.subject_id.to_numpy()).mean().mean()),
                     "within_5_mmHg_percent": float((error <= 5).mean()*100), "within_10_mmHg_percent": float((error <= 10).mean()*100),
                     "predicted_SD": float(predicted[:, c].std(ddof=1)), "reference_SD": float(actual[:, c].std(ddof=1)), **row})
    return rows


def collect(cohort, folds, config, output):
    predictions, metrics, strata, uncertainty, efficiency = {}, [], [], [], []
    for name in STUDY_MODELS + CONSTANTS:
        for seed in (config["seeds"] if name in STUDY_MODELS else [0]):
            for condition in (available_conditions(name) if name in STUDY_MODELS else CONDITIONS):
                frames = []
                for fold, split in enumerate(folds):
                    if name in STUDY_MODELS:
                        directory = output / "outer" / f"fold_{fold}" / name / f"seed_{seed}"
                        frame = pd.read_csv(directory / f"predictions_{condition}.csv")
                        if condition == "clean":
                            e = json.loads((directory / "efficiency.json").read_text())
                            efficiency.append({"model": name, "seed": seed, "outer_fold": fold, "parameters": e["parameters"], "MACs": e["MACs_per_record"],
                                               "approx_FLOPs": e["approx_FLOPs_per_record"], "serialized_weights_KiB": e["serialized_state_dict_bytes"]/1024,
                                               "latency_ms": e["model_only"]["batch_1"]["median_latency_ms"], "throughput_batch32_per_s": e["model_only"]["batch_32"]["records_per_second"],
                                               "end_to_end_ms": e["end_to_end_median_ms"], "peak_RSS_MiB": e["isolated_process_peak_RSS_MiB"], "device": e["device_name"]})
                    else:
                        frame = cohort.iloc[split["indices"]["outer"]][["record_id", "subject_id"]+TARGETS+QUALITY].copy()
                        p = constant_prediction(cohort.iloc[split["indices"]["train"]][TARGETS].to_numpy(), len(frame), name)
                        frame["sbp_predicted"], frame["dbp_predicted"] = p[:, 0], p[:, 1]
                        frame["outer_fold"], frame["condition"] = fold, condition
                        if condition != "clean":
                            frame = frame.drop(columns=QUALITY)
                    frames.append(frame)
                frame = pd.concat(frames, ignore_index=True).sort_values("record_id").reset_index(drop=True)
                if frame.record_id.duplicated().any() or set(frame.record_id) != set(cohort.record_id):
                    raise AssertionError("Incomplete or duplicated OOF predictions")
                frame.to_csv(output / "predictions" / f"{name}_seed_{seed}_{condition}.csv", index=False)
                predictions.setdefault(condition, {}).setdefault(name, []).append(frame)
                metrics.extend(metric_rows(frame, name, seed, condition))
                if condition == "clean":
                    p = frame[["sbp_predicted", "dbp_predicted"]].to_numpy()
                    strata.extend({"model": name, "seed": seed, **r} for r in quality_strata(frame, p))
                    uncertainty.extend({"model": name, "seed": seed, **r} for r in cluster_bootstrap(frame[TARGETS].to_numpy(), p, frame.subject_id.to_numpy(), config["bootstrap_repetitions"], config["protocol_seed"]))
    pd.DataFrame(metrics).to_csv(output / "tables/metrics_per_seed.csv", index=False)
    pd.DataFrame(strata).to_csv(output / "tables/quality_stratified.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "tables/participant_bootstrap_CI.csv", index=False)
    pd.DataFrame(efficiency).to_csv(output / "tables/efficiency_per_fold_seed.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["model", "condition", "target"]).agg(MAE=("MAE", "mean"), MAE_seed_SD=("MAE", "std"),
              RMSE=("RMSE", "mean"), R2=("R2", "mean"), ME=("ME", "mean"), SD_error=("SD_error", "mean"),
              participant_macro_MAE=("participant_macro_MAE", "mean"), predicted_SD=("predicted_SD", "mean"), reference_SD=("reference_SD", "mean")).reset_index()
    summary.to_csv(output / "tables/summary.csv", index=False)
    return predictions, summary, pd.DataFrame(efficiency)


def contrast_table(predictions, config):
    frame = predictions["clean"]["A_uniform_clean"][0]
    subjects, inverse = np.unique(frame.subject_id, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(config["protocol_seed"]).integers(len(subjects), size=(config["bootstrap_repetitions"], len(subjects)))
    losses = {}
    for condition, models in predictions.items():
        losses[condition] = {}
        for name, runs in models.items():
            for run in runs:
                if run.record_id.tolist() != frame.record_id.tolist():
                    raise AssertionError("Unaligned paired comparisons")
            losses[condition][name] = np.mean([np.abs(r[["sbp_predicted", "dbp_predicted"]].to_numpy()-r[TARGETS].to_numpy()) for r in runs], axis=0)
    losses["known_corruption_mean"] = {name: np.mean([losses[c][name] for c in KNOWN_CONDITIONS], axis=0) for name in CELLS}
    rows = []
    for condition, models in losses.items():
        for name, terms in CONTRASTS.items():
            if not set(terms).issubset(models):
                continue
            delta = sum(weight*models[key] for key, weight in terms.items())
            for c, target in enumerate(["SBP", "DBP"]):
                sums = np.bincount(inverse, weights=delta[:, c])
                distribution = sums[draws].sum(1)/counts[draws].sum(1)
                low, high = np.quantile(distribution, [.025, .975])
                primary = condition == config["primary_condition"] and name in config["primary_contrasts"] and target in config["primary_targets"]
                family_size = len(config["primary_contrasts"])*len(config["primary_targets"])
                alpha = .05/family_size
                adjusted = np.quantile(distribution, [alpha/2, 1-alpha/2]) if primary else [None, None]
                rows.append({"contrast": name, "condition": condition, "target": target, "delta_MAE": float(delta[:, c].mean()),
                             "lower_95": float(low), "upper_95": float(high), "primary": primary,
                             "primary_family_Bonferroni_lower": adjusted[0], "primary_family_Bonferroni_upper": adjusted[1],
                             "family_interval_confidence": 1-alpha if primary else None,
                             "bootstrap_unit": "participant", "interpretation": "negative favors first term; seed-averaged losses, not an ensemble; conditional on fitted OOF predictions"})
    return pd.DataFrame(rows)


def selective_tables(predictions, lock, config, output, demographics):
    thresholds = {(x["fold"], x["model"], x["seed"]): x["quality_thresholds"] for x in lock["selections"]}
    metrics, subgroups, decisions, quality = [], [], [], []
    for condition in CONDITIONS:
        for name in CELLS:
            for seed, frame in zip(config["seeds"], predictions[condition][name], strict=True):
                frame = frame.merge(demographics[["subject_id", "age_years", "sex_code"]], on="subject_id", how="left", validate="many_to_one")
                scores = frame[["ecg_estimated_goodness", "ppg_estimated_goodness"]].min(axis=1).to_numpy()
                errors = np.abs(frame[["sbp_predicted", "dbp_predicted"]].to_numpy()-frame[TARGETS].to_numpy())
                if condition == "clean":
                    for modality in ["ecg", "ppg"]:
                        estimated = frame[modality+"_estimated_goodness"].to_numpy()
                        labels = frame[modality+"_quality_label"].to_numpy()
                        err = np.abs(estimated-(1-labels/3))
                        rho = float(spearmanr(estimated, 1-labels/3).statistic)
                        quality.append({"model": name, "seed": seed, "modality": modality, "goodness_MAE": float(err.mean()),
                                        "class_macro_goodness_MAE": float(np.mean([err[labels == k].mean() for k in np.unique(labels)])),
                                        "Spearman_rho": rho if np.isfinite(rho) else None})
                for coverage in config["selective_coverages"]:
                    threshold = np.asarray([thresholds[(int(f), name, seed)][str(coverage)] for f in frame.outer_fold])
                    keep = scores >= threshold
                    random_errors = []
                    rng = np.random.default_rng(config["protocol_seed"] + seed)
                    for _ in range(config["random_rejection_repetitions"]):
                        random_keep = np.zeros(len(frame), dtype=bool)
                        for fold in range(config["outer_folds"]):
                            ix = np.flatnonzero(frame.outer_fold.to_numpy() == fold)
                            random_keep[rng.choice(ix, size=int(keep[ix].sum()), replace=False)] = True
                        random_errors.append(errors[random_keep].mean(0) if random_keep.any() else [np.nan, np.nan])
                    random_errors = np.asarray(random_errors)
                    metadata = {"model": name, "seed": seed, "condition": condition, "validation_target_coverage": coverage,
                                "actual_record_coverage": float(keep.mean()), "n_accepted": int(keep.sum()), "n_records": len(frame),
                                "participants_with_accepted_record": int(frame.loc[keep, "subject_id"].nunique())}
                    for c, target in enumerate(["SBP", "DBP"]):
                        metrics.append({**metadata, "target": target, "selective_MAE": float(errors[keep, c].mean()) if keep.any() else None,
                                        "all_records_MAE": float(errors[:, c].mean()),
                                        "random_rejection_MAE_mean": float(random_errors[:, c].mean()) if keep.any() else None,
                                        "random_rejection_MC_SD": float(random_errors[:, c].std(ddof=1)) if keep.any() else None,
                                        "random_scope": "same accepted record count within each outer fold; MC SD is not a participant confidence interval"})
                    if condition == "clean":
                        d = frame[["record_id", "subject_id", "outer_fold"]].copy()
                        d["model"], d["seed"], d["validation_target_coverage"] = name, seed, coverage
                        d["estimated_goodness_min"], d["validation_threshold"], d["accepted"] = scores, threshold, keep
                        decisions.append(d)
                        groups = demographic_strata(frame)
                        for group, mask in groups.items():
                            mask = mask.to_numpy()
                            if not mask.any():
                                continue
                            selected = keep & mask
                            for c, target in enumerate(["SBP", "DBP"]):
                                subgroups.append({**metadata, "subgroup": group, "target": target, "subgroup_n": int(mask.sum()),
                                                  "subgroup_participants": int(frame.loc[mask, "subject_id"].nunique()),
                                                  "subgroup_accepted": int(selected.sum()), "subgroup_coverage": float(keep[mask].mean()),
                                                  "subgroup_all_MAE": float(errors[mask, c].mean()),
                                                  "subgroup_selective_MAE": float(errors[selected, c].mean()) if selected.any() else None})
    pd.DataFrame(metrics).to_csv(output / "tables/selective_risk_coverage.csv", index=False)
    pd.DataFrame(subgroups).to_csv(output / "tables/selective_subgroups.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(output / "predictions/selective_clean_decisions.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "tables/quality_estimation.csv", index=False)


def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config.update({"seeds": [17], "epochs": 2, "patience": 2, "bootstrap_repetitions": 100,
                       "latency_warmup": 2, "latency_repetitions": 10, "random_rejection_repetitions": 10})
    config["quick"] = args.quick
    config["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    output = Path(args.output or ("results/factorial_smoke" if args.quick else "results/factorial"))
    for directory in [output, output/"logs", output/"tables", output/"predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output/"logs/factorial.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    torch.set_num_threads(config["threads"])
    root = Path(args.data).resolve()
    cohort, clean, audit = prepare_dataset(root, args.cache, DEFAULT_PREPROCESSING)
    historical = json.loads(Path(config["retired_test_split"]).read_text())["participant_ids"]
    retired = set(historical["test"])
    allowed = set(historical["train"]) | set(historical["validation"])
    if allowed & retired or set(cohort.subject_id) != allowed | retired:
        raise ValueError("Historical participant roles changed")
    if args.quick:
        pool = np.asarray(sorted(allowed))
        np.random.default_rng(config["protocol_seed"]).shuffle(pool)
        allowed = set(pool[:90])
    indices = np.flatnonzero(cohort.subject_id.isin(allowed))
    cohort, clean = cohort.iloc[indices].reset_index(drop=True), subset(clean, indices)
    folds = nested_subject_splits(cohort, config)
    protocol = {"config": config, "source_sha256": code_hash(), "environment": environment(), "dataset_fingerprint": audit["fingerprint"],
                "development_record_ids": cohort.record_id.tolist(), "development_subject_ids": sorted(allowed), "retired_test_subject_ids": sorted(retired),
                "outer_folds": [{"fold": i, "participant_ids": f["participant_ids"]} for i, f in enumerate(folds)],
                "models": STUDY_MODELS, "constants": CONSTANTS, "factorial_cells": CELLS, "contrasts": CONTRASTS,
                "primary_family": "two clean gating contrasts x SBP/DBP; four Bonferroni-adjusted conditional bootstrap intervals",
                "secondary_analyses": "augmentation/interaction/multimodal/dense contrasts, synthetic perturbations, selective prediction and subgroups are exploratory",
                "quality_supervision": "identical class-balanced manual soft BCE in every cell; no corruption-ranking loss",
                "selective_policy": "minimum automatic goodness, validation-only thresholds; inclusive ties; all coverage levels shown; random rejection matches per-fold accepted counts",
                "known_missing_sensor_rule": "identical waveform availability masks; single-sensor controls abstain on own-sensor absence",
                "prior_results_informed_design": True, "independent_confirmatory_test": False,
                "status_of_evidence": "post-hoc matched development study on previously analyzed cohort; no new blind validation"}
    run_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text())["run_sha256"] != run_hash:
        raise ValueError("Code/config/environment/data changed: use a new output directory")
    if not path.exists():
        save_json(path, {"run_sha256": run_hash, **protocol})
        save_json(output/"resolved_config.json", config)
        cohort.to_csv(output/"tables/development_cohort.csv", index=False)
        save_json(output/"tables/dataset_summary.json", audit)
        snapshot = output/"source_snapshot"
        (snapshot/"src").mkdir(parents=True, exist_ok=True)
        (snapshot/"configs").mkdir(exist_ok=True)
        for source in Path("src").glob("*.py"):
            shutil.copy2(source, snapshot/"src"/source.name)
        for filename in ["main.py", "requirements.lock.txt"]:
            shutil.copy2(filename, snapshot/filename)
        shutil.copy2(args.config, snapshot/"configs"/Path(args.config).name)
    save_json(output/"status.json", {"status": "building_caches", "run_sha256": run_hash})
    bank_dir = Path(args.cache) / ("factorial_smoke" if args.quick else "factorial")
    banks = build_banks(root, cohort, config, bank_dir, audit["fingerprint"])
    save_json(output/"status.json", {"status": "fitting_inner_only", "run_sha256": run_hash})
    lock = fit_and_lock(cohort, clean, banks, folds, config, output, run_hash)
    if lock["source_sha256"] != code_hash():
        raise ValueError("Source changed during fitting")
    save_json(output/"status.json", {"status": "selection_locked" if args.phase == "fit" else "outer_evaluation", "run_sha256": run_hash})
    if args.phase == "fit":
        return
    evaluate_locked(cohort, clean, banks, folds, config, output, lock, root, run_hash)
    predictions, summary, efficiency = collect(cohort, folds, config, output)
    contrast_table(predictions, config).to_csv(output/"tables/factorial_contrasts.csv", index=False)
    demographics = pd.read_csv(root/"subject-info.csv")
    selective_tables(predictions, lock, config, output, demographics)
    from .factorial_report import render_report
    render_report(output, summary, efficiency, config, predictions)
    if code_hash() != protocol["source_sha256"]:
        raise ValueError("Source changed after protocol freeze")
    save_json(output/"status.json", {"status": "complete", "run_sha256": run_hash, "quick": args.quick,
                                     "development_records": len(cohort), "development_subjects": int(cohort.subject_id.nunique()),
                                     "retired_test_subjects": len(retired), "neural_fits": len(lock["selections"]),
                                     "independent_confirmatory_test": False, "clinical_validation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/factorial.json")
    parser.add_argument("--data", default="physionet.org/files/cardiowave-portable/1.0.0")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--phase", choices=["all", "fit"], default="all")
    run(parser.parse_args())
