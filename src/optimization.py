"""Post-hoc development optimization: retired historical test, nested subject CV."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .ablation import CONDITIONS, degrade
from .dataset_loader import QUALITY, TARGETS, prepare_dataset, read_waveform
from .evaluate import cluster_bootstrap, paired_comparisons, prediction_frame, quality_strata, regression_metrics
from .optimized_model import RobustBPModel
from .preprocessing import DEFAULT_PREPROCESSING, fit_scalers, preprocess, scale_sqi
from .train import load_model, predict, train_model
from .utils import code_hash, environment, save_json, seed_everything

LOGGER = logging.getLogger(__name__)
KNOWN_CONDITIONS = list(CONDITIONS)
UNSEEN_CONDITIONS = ["ecg_baseline_drift", "ppg_motion", "ppg_clipping"]


def seed_for(record_id, stage, seed):
    return int(hashlib.sha256(f"{seed}:{stage}:{record_id}".encode()).hexdigest()[:16], 16)


def nested_subject_splits(cohort, config):
    subjects = np.asarray(sorted(cohort.subject_id.unique()))
    folds = []
    for fold, (development, outer) in enumerate(KFold(config["outer_folds"], shuffle=True, random_state=config["protocol_seed"]).split(subjects)):
        pool = subjects[development].copy()
        np.random.default_rng(config["protocol_seed"] + fold).shuffle(pool)
        count = max(1, int(np.ceil(len(pool)*config["inner_validation_fraction"])))
        groups = {"train": pool[count:].tolist(), "validation": pool[:count].tolist(), "outer": subjects[outer].tolist()}
        for left, right in [("train", "validation"), ("train", "outer"), ("validation", "outer")]:
            if set(groups[left]) & set(groups[right]):
                raise AssertionError("Nested participant leakage")
        indices = {k: np.flatnonzero(cohort.subject_id.isin(v)) for k, v in groups.items()}
        folds.append({"participant_ids": groups, "indices": indices})
    counts = np.bincount(np.concatenate([f["indices"]["outer"] for f in folds]), minlength=len(cohort))
    if not np.all(counts == 1):
        raise AssertionError("Each development record must be outer-held-out exactly once")
    return folds


def corrupt(raw, condition, seed, snr=None):
    rng = np.random.default_rng(seed)
    raw = np.asarray(raw, dtype=np.float64)
    if condition in CONDITIONS:
        altered = degrade(raw, condition, seed)
        base_snr = CONDITIONS[condition][1]
        if snr is not None and base_snr is not None:
            altered = raw + (altered - raw) * 10 ** ((base_snr - snr)/20)
        return altered
    altered = raw.copy()
    if condition == "ecg_baseline_drift":
        altered[:, 0] += .7*raw[:, 0].std()*np.sin(2*np.pi*.2*np.arange(len(raw))/500 + rng.uniform(0, 2*np.pi))
    elif condition == "ppg_motion":
        noise = signal.sosfiltfilt(signal.butter(3, [.4, 4.], fs=500, btype="bandpass", output="sos"), rng.normal(size=len(raw)))
        altered[:, 1] += noise/max(noise.std(), 1e-8)*raw[:, 1].std()
    elif condition == "ppg_clipping":
        median, std = np.median(raw[:, 1]), raw[:, 1].std()
        altered[:, 1] = median + np.clip(raw[:, 1]-median, -.5*std, .5*std)
    else:
        raise ValueError(condition)
    return altered


def build_banks(root, cohort, config, cache, dataset_fingerprint):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    descriptor = {"records": cohort.record_id.tolist(), "dataset_fingerprint": dataset_fingerprint,
                  "augmentation_seed": config["training_augmentation_seed"], "evaluation_seed": config["evaluation_corruption_seed"],
                  "snr": config["training_noise_snr_dB"], "both_snr": config["training_both_noise_snr_dB"],
                  "preprocessing": DEFAULT_PREPROCESSING, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    manifest = cache / "manifest.json"
    shapes = {"x": (2, 3000), "sqi": (2, 7), "automatic": (2,)}
    banks = {}
    existing = manifest.exists() and json.loads(manifest.read_text())["sha256"] == fingerprint
    for stage, conditions in [("training", KNOWN_CONDITIONS), ("evaluation", KNOWN_CONDITIONS + UNSEEN_CONDITIONS)]:
        banks[stage] = {}
        for key, shape in shapes.items():
            path = cache / f"{stage}_{key}.npy"
            banks[stage][key] = np.load(path, mmap_mode="r") if existing else np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(conditions), len(cohort), *shape))
    if existing:
        LOGGER.info("Using cached independent training/evaluation perturbation banks")
        return banks

    def prepare(row):
        index, record = row
        raw = read_waveform(root, record.wfdb_record_path)
        outputs, metadata = {}, []
        for stage, conditions in [("training", KNOWN_CONDITIONS), ("evaluation", KNOWN_CONDITIONS + UNSEEN_CONDITIONS)]:
            values = []
            for condition in conditions:
                seed = seed_for(record.record_id, stage + ":" + condition, config["training_augmentation_seed"] if stage == "training" else config["evaluation_corruption_seed"])
                snr = None
                if stage == "training" and CONDITIONS[condition][1] is not None:
                    low, high = config["training_both_noise_snr_dB"] if condition.startswith("both") else config["training_noise_snr_dB"]
                    snr = float(np.random.default_rng(seed).uniform(low, high))
                values.append(preprocess(corrupt(raw, condition, seed, snr), DEFAULT_PREPROCESSING))
                metadata.append({"record_id": record.record_id, "stage": stage, "condition": condition, "seed": seed, "training_snr_dB": snr})
            outputs[stage] = values
        return index, outputs, metadata

    metadata = []
    with ThreadPoolExecutor(max_workers=config["threads"]) as executor:
        for done, (index, outputs, details) in enumerate(executor.map(prepare, enumerate(cohort.itertuples(index=False))), 1):
            for stage, values in outputs.items():
                for variant, triple in enumerate(values):
                    for key, value in zip(shapes, triple, strict=True):
                        banks[stage][key][variant, index] = value
            metadata.extend(details)
            if done % 250 == 0:
                LOGGER.info("Prepared waveform-only perturbations %d/%d", done, len(cohort))
    for arrays in banks.values():
        for array in arrays.values():
            array.flush()
    pd.DataFrame(metadata).to_csv(cache / "perturbation_parameters.csv", index=False)
    save_json(manifest, {"sha256": fingerprint, **descriptor,
                         "BP_labels_synthesized": False, "manual_quality_labels_synthesized": False,
                         "interpretation": "Added noise and sensor absence simulate measurement corruption at fixed underlying BP; ranking is a proxy, not a new manual quality annotation."})
    return banks


def subset(arrays, indices, variant=None):
    return {key: value[indices] if variant is None else value[variant, indices] for key, value in arrays.items()}


def fit_new_model(name, seed, cohort, clean, bank, splits, config, directory, provenance, fixed_epochs=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, config["threads"])
    device = config["device"]
    y = cohort[TARGETS].to_numpy(np.float32)
    manual = 1 - cohort[QUALITY].to_numpy(np.float32)/3
    scalers = fit_scalers(clean["sqi"], y, splits["train"])
    scalers["sqi_mode"] = "bounded_v2"
    standardized = ((y - np.asarray(scalers["y_mean"])) / np.asarray(scalers["y_std"])).astype(np.float32)
    sqi = scale_sqi(clean["sqi"], scalers)
    train_sqi = scale_sqi(bank["training"]["sqi"], scalers)
    evaluation_sqi = scale_sqi(bank["evaluation"]["sqi"], scalers)
    quality_weights = np.ones_like(manual)
    for c, col in enumerate(QUALITY):
        counts = cohort.iloc[splits["train"]][col].value_counts()
        quality_weights[:, c] = [np.clip(len(splits["train"])/(4*max(int(counts.get(label, 0)), 1)), .5, 3.) for label in cohort[col]]
    if config.get("architecture_version") == "factorial_v1":
        from .factorial_model import FactorialBPModel
        model = FactorialBPModel(name).to(device)
    else:
        model = RobustBPModel(name).to(device)
    augmentation = config.get("training_augmentation", True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    y_std = torch.tensor(scalers["y_std"], dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    augmentation_rng = np.random.default_rng(seed + 19091)
    loader = DataLoader(TensorDataset(torch.as_tensor(splits["train"], dtype=torch.int64)), batch_size=config["batch_size"], shuffle=True, generator=generator)
    best, stalled, history = float("inf"), 0, []

    def tensors(values):
        return [torch.as_tensor(v, dtype=torch.float32, device=device) for v in values]

    def validation():
        indices = splits["validation"]
        clean_sum, bad_sum, count = 0., 0., 0
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), config["batch_size"]):
                ix = indices[start:start+config["batch_size"]]
                target = tensors([standardized[ix]])[0]
                values, _, _ = model(*tensors([clean["x"][ix], sqi[ix], clean["automatic"][ix]]))
                clean_sum += float(((values-target).abs()*y_std).mean().item())*len(ix)
                if config["validation_corrupted_weight"]:
                    variants = (ix*37 + config["protocol_seed"]) % len(KNOWN_CONDITIONS)
                    altered, _, _ = model(*tensors([bank["evaluation"]["x"][variants, ix], evaluation_sqi[variants, ix], bank["evaluation"]["automatic"][variants, ix]]))
                    bad_sum += float(((altered-target).abs()*y_std).mean().item())*len(ix)
                count += len(ix)
        weight = config["validation_corrupted_weight"]
        return clean_sum/count, bad_sum/count if weight else None, ((1-weight)*clean_sum + weight*bad_sum)/count

    epochs = fixed_epochs if fixed_epochs is not None else config["epochs"]
    for epoch in range(1, epochs+1):
        model.train()
        totals = np.zeros(4)
        for (batch_indices,) in loader:
            ix = batch_indices.numpy()
            variants = augmentation_rng.integers(len(KNOWN_CONDITIONS), size=len(ix))
            x, features, automatic, target, goodness, qw = tensors([clean["x"][ix], sqi[ix], clean["automatic"][ix], standardized[ix], manual[ix], quality_weights[ix]])
            predicted, logits, _ = model(x, features, automatic)
            clean_loss = F.smooth_l1_loss(predicted*y_std, target*y_std, beta=config["huber_beta_mmHg"])
            regression = clean_loss
            if augmentation:
                altered_x, altered_sqi, altered_auto = tensors([bank["training"]["x"][variants, ix], train_sqi[variants, ix], bank["training"]["automatic"][variants, ix]])
                corrupted, corrupted_logits, _ = model(altered_x, altered_sqi, altered_auto)
                altered_loss = F.smooth_l1_loss(corrupted*y_std, target*y_std, beta=config["huber_beta_mmHg"])
                mix = config["corrupted_bp_loss_weight"]
                regression = (1-mix)*clean_loss + mix*altered_loss
            auxiliary, ranking = regression.new_zeros(()), regression.new_zeros(())
            if model.supervised_quality:
                available = (x.abs().amax(-1) > 1e-6).to(x.dtype)
                if hasattr(model, "modalities"):
                    available = available * torch.as_tensor([c in model.modalities for c in range(2)], device=device)
                auxiliary = (F.binary_cross_entropy_with_logits(logits, goodness, reduction="none")*qw*available).sum() / available.sum().clamp(min=1)
                if augmentation:
                    affected = np.asarray([[True, False], [False, True], [True, True], [True, False], [False, True]])[variants]
                    mask = torch.as_tensor(affected, device=device) * available
                    ranking = (F.relu(corrupted_logits.sigmoid() - (1-config["ranking_fraction"])*logits.sigmoid().detach())*mask).sum()/mask.sum().clamp(min=1)
            rank_weight = 0. if name == "optimized_quality_no_rank" else config["ranking_weight"]
            loss = regression + config["quality_loss_weight"]*auxiliary + rank_weight*ranking
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite optimization loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals += np.asarray([loss.item(), regression.item(), auxiliary.item(), ranking.item()])*len(ix)
        row = {"epoch": epoch, "train_loss": totals[0]/len(splits["train"]), "train_regression": totals[1]/len(splits["train"]),
               "train_quality_BCE": totals[2]/len(splits["train"]), "train_ranking": totals[3]/len(splits["train"])}
        if fixed_epochs is None:
            val_clean, val_bad, score = validation()
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            row.update({"validation_clean_MAE": val_clean, "validation_corrupted_MAE": val_bad, "validation_score": score})
            scheduler.step(score)
        else:
            score = float(epochs-epoch)
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if score < best - .001 or fixed_epochs is not None:
            best, stalled = score, 0
            checkpoint = {"architecture_version": config.get("architecture_version", "quality_v2"), "model_name": name, "state_dict": model.state_dict(), "seed": seed,
                          "epoch": epoch, "scalers": scalers, "preprocessing": DEFAULT_PREPROCESSING, "config": config,
                          "validation_score": best if fixed_epochs is None else None, "fixed_epoch_development_refit": fixed_epochs is not None, "provenance": provenance}
            torch.save(checkpoint, directory / "best.tmp.pt")
            (directory / "best.tmp.pt").replace(directory / "best.pt")
        else:
            stalled += 1
        if epoch == 1 or epoch % 5 == 0:
            LOGGER.info("%s seed=%d epoch=%d/%d loss=%.3f selection_score=%s", name, seed, epoch, epochs, row["train_loss"], f"{score:.3f}" if fixed_epochs is None else "not evaluated (final refit)")
        if fixed_epochs is None and stalled >= config["patience"]:
            break
    save_json(directory / "training_complete.json", {"epochs_run": epoch, "best_epoch": checkpoint["epoch"], "provenance": provenance})
    return directory / "best.pt"


def inner_selection_score(checkpoint_path, cohort, clean, banks, validation_indices, config):
    model, checkpoint = load_model(checkpoint_path, config["device"])
    actual = cohort.iloc[validation_indices][TARGETS].to_numpy()
    rows = []
    for variant, condition in [(-1, "clean")] + list(enumerate(KNOWN_CONDITIONS)):
        inputs = subset(clean, validation_indices) if variant == -1 else subset(banks["evaluation"], validation_indices, variant)
        predicted, _, _ = predict(model, inputs, checkpoint["scalers"], config["device"], config["batch_size"])
        rows.extend({"condition": condition, **row} for row in regression_metrics(actual, predicted))
    table = pd.DataFrame(rows)
    clean_mae = float(table[table.condition == "clean"].MAE.mean())
    corrupted_mae = float(table[table.condition != "clean"].MAE.mean())
    weight = config["validation_corrupted_weight"]
    return (1-weight)*clean_mae + weight*corrupted_mae, table


def tune_all_folds(cohort, clean, banks, folds, config, output, run_hash):
    lock_path = output / "selection_locked.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())["selections"]
    selections, trials = [], []
    for fold, split in enumerate(folds):
        for candidate, quality_weight in enumerate(config["quality_loss_candidates"]):
            directory = output / "tuning" / f"fold_{fold}" / f"candidate_{candidate}"
            candidate_config = {**config, "quality_loss_weight": quality_weight}
            provenance = {"run_sha256": run_hash, "phase": "inner_tuning", "fold": fold,
                          "participant_ids": {k: v for k, v in split["participant_ids"].items() if k != "outer"}}
            checkpoint_path = directory / "best.pt"
            if not (directory / "training_complete.json").exists():
                checkpoint_path = fit_new_model("optimized_quality", config["tuning_seed"], cohort, clean, banks, split["indices"], candidate_config, directory, provenance)
            score, detail = inner_selection_score(checkpoint_path, cohort, clean, banks, split["indices"]["validation"], config)
            detail.to_csv(directory / "inner_validation_metrics.csv", index=False)
            trials.append({"fold": fold, "candidate": candidate, "quality_loss_weight": quality_weight, "selection_score": score,
                           "checkpoint": str(checkpoint_path), "best_epoch": torch.load(checkpoint_path, map_location="cpu", weights_only=True)["epoch"]})
        selected = min([row for row in trials if row["fold"] == fold], key=lambda row: (row["selection_score"], row["candidate"]))
        selections.append(selected)
        LOGGER.info("Fold %d candidate frozen using inner validation: %s", fold, selected)
        pd.DataFrame(trials).to_csv(output / "tables/inner_selection_trials.csv", index=False)
    save_json(lock_path, {"run_sha256": run_hash, "source_sha256": code_hash(), "selections": selections,
                          "outer_predictions_read_for_selection": False,
                          "selection_objective": "0.5 clean mean SBP/DBP MAE + 0.5 mean MAE over 5 independent inner-validation corruption conditions",
                          "unseen_stress_tests_used_for_selection": False})
    LOGGER.info("ALL outer-fold hyperparameter selections are locked before any outer evaluation")
    return selections


def outer_evaluate(checkpoint_path, cohort, clean, banks, indices, config, directory):
    model, checkpoint = load_model(checkpoint_path, config["device"])
    test = cohort.iloc[indices].reset_index(drop=True)
    actual = test[TARGETS].to_numpy()
    quality_rows, metrics_rows, strata = [], [], []
    for variant, condition in [(-1, "clean")] + list(enumerate(KNOWN_CONDITIONS + UNSEEN_CONDITIONS)):
        inputs = subset(clean, indices) if variant == -1 else subset(banks["evaluation"], indices, variant)
        p, q, gates = predict(model, inputs, checkpoint["scalers"], config["device"], config["batch_size"])
        frame = prediction_frame(test, p, q, gates)
        frame["condition"] = condition
        if condition != "clean":
            frame = frame.drop(columns=QUALITY)
        if not model.supervised_quality:
            frame = frame.drop(columns=["ecg_estimated_goodness", "ppg_estimated_goodness"])
        frame.to_csv(directory / f"predictions_{condition}.csv", index=False)
        for c, row in enumerate(regression_metrics(actual, p)):
            error = np.abs(p[:, c] - actual[:, c])
            metrics_rows.append({"condition": condition, "within_5_mmHg_percent": float((error <= 5).mean()*100),
                                 "within_10_mmHg_percent": float((error <= 10).mean()*100), **row})
        if condition == "clean":
            strata = quality_strata(test, p)
        if model.supervised_quality:
            for c, modality in enumerate(["ECG", "PPG"]):
                manual_goodness = 1-test[QUALITY[c]].to_numpy()/3
                quality_rows.append({"condition": condition, "modality": modality,
                                     "mean_estimated_goodness": float(q[:, c].mean()), "mean_gate": float(gates[:, c].mean()),
                                     "MAE_vs_manual_clean_goodness": float(np.abs(q[:, c]-manual_goodness).mean()) if condition == "clean" else None})
    pd.DataFrame(metrics_rows).to_csv(directory / "outer_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(directory / "quality_stratified.csv", index=False)
    pd.DataFrame(quality_rows, columns=["condition", "modality", "mean_estimated_goodness", "mean_gate", "MAE_vs_manual_clean_goodness"]).to_csv(directory / "quality_estimator.csv", index=False)


def evaluate_all_folds(cohort, clean, banks, folds, selections, config, output, root, run_hash):
    seal = hashlib.sha256((output / "selection_locked.json").read_bytes()).hexdigest()
    for fold, (split, choice) in enumerate(zip(folds, selections, strict=True)):
        selected_config = {**config, "quality_loss_weight": choice["quality_loss_weight"], "preprocessing": DEFAULT_PREPROCESSING}
        for name in config["models"]:
            for seed in config["seeds"]:
                directory = output / "outer" / f"fold_{fold}" / name / f"seed_{seed}"
                if (directory / "complete.json").exists():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                provenance = {"run_sha256": run_hash, "phase": "outer_evaluation", "fold": fold, "selection_lock_sha256": seal,
                              "participant_ids": {k: v for k, v in split["participant_ids"].items() if k != "outer"}}
                checkpoint_path = directory / "best.pt"
                if name == "optimized_quality" and seed == config["tuning_seed"]:
                    for filename in ["best.pt", "history.csv", "training_complete.json"]:
                        shutil.copy2(Path(choice["checkpoint"]).parent / filename, directory / filename)
                elif not (directory / "training_complete.json").exists():
                    if name == "legacy_proposed":
                        baseline_config = {**selected_config, "epochs": min(30, config["epochs"]), "patience": min(6, config["patience"]),
                                           "quality_loss_weight": .2, "weight_decay": .0001, "min_delta_mmHg": .001}
                        checkpoint_path = train_model("proposed", seed, cohort, clean, {k: v for k, v in split["indices"].items() if k != "outer"}, baseline_config, directory, provenance)
                        save_json(directory / "training_complete.json", {"provenance": provenance})
                    else:
                        checkpoint_path = fit_new_model(name, seed, cohort, clean, banks, split["indices"], selected_config, directory, provenance)
                outer_evaluate(checkpoint_path, cohort, clean, banks, split["indices"]["outer"], config, directory)
                subprocess.run([sys.executable, "-m", "src.efficiency", str(checkpoint_path), str(root),
                                cohort.iloc[split["indices"]["outer"][0]].wfdb_record_path, str(directory / "efficiency.json"),
                                "--threads", str(config["threads"]), "--warmup", str(config["latency_warmup"]),
                                "--repetitions", str(config["latency_repetitions"]), "--device", config["device"]], check=True)
                save_json(directory / "complete.json", {"run_sha256": run_hash, "selection_lock_sha256": seal,
                                                        "outer_records": len(split["indices"]["outer"]), "model": name, "seed": seed})
                LOGGER.info("Completed outer fold %d / %s / seed %d", fold, name, seed)


def collect_results(cohort, config, output):
    metrics, uncertainty, efficiency_rows, strata, diagnostic = [], [], [], [], []
    clean_predictions, condition_predictions = {}, {}
    for name in config["models"]:
        clean_predictions[name] = []
        for seed in config["seeds"]:
            for condition in ["clean"] + KNOWN_CONDITIONS + UNSEEN_CONDITIONS:
                frames = []
                for fold in range(config["outer_folds"]):
                    directory = output / "outer" / f"fold_{fold}" / name / f"seed_{seed}"
                    frame = pd.read_csv(directory / f"predictions_{condition}.csv")
                    frame["outer_fold"] = fold
                    frames.append(frame)
                    if condition == "clean":
                        efficiency = json.loads((directory / "efficiency.json").read_text())
                        efficiency_rows.append({"model": name, "seed": seed, "outer_fold": fold, "parameters": efficiency["parameters"],
                                                "MACs": efficiency["MACs_per_record"], "serialized_weights_KiB": efficiency["serialized_state_dict_bytes"]/1024,
                                                "latency_ms": efficiency["model_only"]["batch_1"]["median_latency_ms"], "throughput_batch32_per_s": efficiency["model_only"]["batch_32"]["records_per_second"],
                                                "end_to_end_ms": efficiency["end_to_end_median_ms"], "isolated_peak_RSS_MiB": efficiency["isolated_process_peak_RSS_MiB"], "device": efficiency["device_name"]})
                        q = pd.read_csv(directory / "quality_estimator.csv")
                        q["model"], q["seed"], q["outer_fold"] = name, seed, fold
                        diagnostic.append(q)
                combined = pd.concat(frames, ignore_index=True).sort_values("record_id").reset_index(drop=True)
                if combined.record_id.duplicated().any() or set(combined.record_id) != set(cohort.record_id):
                    raise AssertionError("OOF coverage is not exactly one prediction per development record")
                actual = combined[TARGETS].to_numpy()
                predicted = combined[["sbp_predicted", "dbp_predicted"]].to_numpy()
                for c, row in enumerate(regression_metrics(actual, predicted)):
                    error = np.abs(actual[:, c]-predicted[:, c])
                    metrics.append({"model": name, "seed": seed, "condition": condition,
                                    "participant_macro_MAE": float(pd.Series(error).groupby(combined.subject_id.to_numpy()).mean().mean()),
                                    "within_5_mmHg_percent": float((error<=5).mean()*100), "within_10_mmHg_percent": float((error<=10).mean()*100), **row})
                condition_predictions.setdefault(condition, {}).setdefault(name, []).append(combined)
                combined.to_csv(output / "predictions" / f"{name}_seed_{seed}_{condition}.csv", index=False)
                if condition == "clean":
                    clean_predictions[name].append(combined)
                    uncertainty.extend({"model": name, "seed": seed, **row} for row in cluster_bootstrap(actual, predicted, combined.subject_id.to_numpy(), config["bootstrap_repetitions"], config["protocol_seed"]))
                    strata.extend({"model": name, "seed": seed, **row} for row in quality_strata(combined, predicted))
    pd.DataFrame(metrics).to_csv(output / "tables/oof_metrics_per_seed.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "tables/participant_bootstrap_CI.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv(output / "tables/efficiency_per_fold_seed.csv", index=False)
    pd.DataFrame(strata).to_csv(output / "tables/quality_stratified.csv", index=False)
    pd.concat(diagnostic, ignore_index=True).to_csv(output / "tables/quality_estimator.csv", index=False)
    paired = []
    for condition, models in condition_predictions.items():
        mapped = {("proposed" if k == "optimized_quality" else k): v for k, v in models.items()}
        table = paired_comparisons(mapped, config["bootstrap_repetitions"], config["protocol_seed"])
        table["condition"] = condition
        table["comparison"] = table.comparison.str.replace("proposed minus", "optimized_quality minus", regex=False)
        paired.append(table)
    pd.concat(paired, ignore_index=True).to_csv(output / "tables/paired_MAE_differences.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["model", "condition", "target"])[["MAE", "RMSE", "R2", "ME", "SD_error", "within_5_mmHg_percent", "within_10_mmHg_percent"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary.to_csv(output / "tables/summary.csv", index=False)
    return summary, pd.DataFrame(efficiency_rows), clean_predictions


def render_report(summary, efficiency, config, selections, output, cohort):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .visualization import save

    names = config["models"]
    clean = summary[summary.condition == "clean"].set_index(["model", "target"])
    efficiency_mean = efficiency.groupby("model").mean(numeric_only=True)
    table_rows = []
    for name in names:
        row = {"model": name, "parameters": int(efficiency_mean.loc[name, "parameters"]),
               "MACs": int(efficiency_mean.loc[name, "MACs"]), "latency_ms": float(efficiency_mean.loc[name, "latency_ms"])}
        for target in ["SBP", "DBP"]:
            for metric in ["MAE", "RMSE", "R2", "ME", "SD_error"]:
                row[f"{target}_{metric}"] = float(clean.loc[(name, target), metric + "_mean"])
                row[f"{target}_{metric}_seed_SD"] = float(clean.loc[(name, target), metric + "_std"]) if len(config["seeds"]) > 1 else None
        table_rows.append(row)
    comparison = pd.DataFrame(table_rows)
    comparison.to_csv(output / "tables/comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        values = clean.xs(target, level="target").reindex(names)
        ax.bar(range(len(names)), values.MAE_mean, yerr=values.MAE_std.fillna(0), capsize=3)
        ax.set(xticks=range(len(names)), xticklabels=names, ylabel=f"{target} MAE (mmHg)", title=f"Nested participant OOF: {target}")
        ax.tick_params(axis="x", labelrotation=30)
    save(fig, output / "figures", "nested_clean_accuracy")
    pivot = summary.groupby(["model", "condition"]).MAE_mean.mean().unstack().reindex(names)
    fig, ax = plt.subplots(figsize=(13, 5))
    heat = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(yticks=range(len(pivot)), yticklabels=pivot.index, xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           title="OOF robustness: mean SBP/DBP MAE (mmHg); additional stress families were not used for tuning")
    ax.tick_params(axis="x", labelrotation=35)
    fig.colorbar(heat, ax=ax, label="MAE (mmHg)")
    save(fig, output / "figures", "nested_robustness")
    diagnostics = pd.read_csv(output / "tables/quality_estimator.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field in zip(axes, ["mean_estimated_goodness", "mean_gate"], strict=True):
        for name, style in [("legacy_proposed", "--"), ("optimized_quality", "-")]:
            for modality, marker in [("ECG", "o"), ("PPG", "s")]:
                data = diagnostics[(diagnostics.model == name) & (diagnostics.modality == modality)].groupby("condition")[field].mean()
                ax.plot(data.index, data.values, linestyle=style, marker=marker, label=name + " " + modality)
        ax.set(ylabel="Mean score (0–1)", ylim=(0, 1.05), title=field.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=40)
        ax.legend(fontsize=7)
    save(fig, output / "figures", "nested_quality_gates")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field in zip(axes, ["parameters", "latency_ms"], strict=True):
        for row in comparison.itertuples():
            ax.scatter(getattr(row, field), (row.SBP_MAE+row.DBP_MAE)/2, label=row.model)
        ax.set(xlabel=field, ylabel="Mean SBP/DBP MAE (mmHg)", title="Accuracy–efficiency trade-off")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, output / "figures", "nested_accuracy_efficiency")
    strata = pd.read_csv(output / "tables/quality_stratified.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for name in names:
            values = strata[(strata.model == name) & (strata.target == target) & (strata.modality == "worst_modality")].groupby("manual_poor_windows").MAE.mean()
            ax.plot(values.index, values.values, "o-", label=name)
        ax.set(xlabel="Worst manual poor-window count", ylabel=f"{target} MAE (mmHg)", xticks=range(4))
    axes[1].legend(fontsize=7)
    save(fig, output / "figures", "nested_quality_strata")
    lines = ["# Quality-aware optimization results", "",
             "**QUICK PIPELINE CHECK — NOT RESEARCH RESULTS**" if config.get("quick") else "**Post-hoc nested development cross-validation; not an independent confirmatory test.**", "",
             f"Cohort: {len(cohort)} records / {cohort.subject_id.nunique()} participants. Historical test participants are retired and excluded.",
             f"Outer folds: {config['outer_folds']}; seeds: {config['seeds']}. Each participant is evaluated out of fold exactly once per seed.",
             "The inner validation fraction is reserved for scheduling, early stopping, and choosing quality-loss weight. Outer observations never fit scalers or select candidates.", "",
             "All candidate selections were sealed before the first outer evaluation. Previous study results informed the method design, so these are development estimates; an external or genuinely untouched future cohort is still required.", "",
             "## Clean OOF accuracy", "", "Mean over training seeds, pooling the three outer folds per seed. MAE/RMSE in mmHg. Latency is model-only FP32 batch 1 on the recorded CPU/device.", "",
             "| Model | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | Parameters | Latency ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        lines.append(f"| {row.model} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} | {row.SBP_RMSE:.3f} | {row.DBP_RMSE:.3f} | {row.parameters} | {row.latency_ms:.3f} |")
    lines += ["", "## Quality-aware contribution", ""]
    paired = pd.read_csv(output / "tables/paired_MAE_differences.csv")
    for row in paired[(paired.condition == "clean") & paired.comparison.isin(["optimized_quality minus legacy_proposed", "optimized_quality minus optimized_no_gate", "optimized_quality minus optimized_quality_no_rank"])].itertuples():
        status = "interval includes zero; advantage not established" if row.lower_95 <= 0 <= row.upper_95 else "exploratory interval excludes zero"
        lines.append(f"- {row.comparison}, {row.target}: delta MAE {row.delta_MAE:+.3f}, participant-bootstrap 95% CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}] mmHg; {status}.")
    lines += ["", "## Robustness", ""]
    for condition in pivot.columns:
        delta = pivot.loc["optimized_quality", condition]-pivot.loc["legacy_proposed", condition]
        matched_delta = pivot.loc["optimized_quality", condition]-pivot.loc["optimized_no_gate", condition]
        lines.append(f"- {condition}: optimized minus legacy mean MAE {delta:+.3f} mmHg; optimized minus matched no-gate {matched_delta:+.3f} mmHg. Negative favors optimized quality gating.")
    lines += ["", "## Locked selections", "", json.dumps(selections, indent=2), "",
              "## Interpretation and validity limits", "",
              "- Historical MAE 13.00/8.57 mmHg used a different test cohort. Do not compare those numbers directly with this OOF cohort; use the legacy model retrained on these exact folds.",
              "- Legacy uses its original MSE/clean-training/30-epoch protocol. Improvements over legacy combine architecture, regression objective, and augmentation. Only matched optimized ablations isolate gating/ranking.",
              "- All optimized controls share architecture, availability masks, paired clean/corrupted BP losses, data, seeds, epoch budget, and validation objective. No-gate has the same trainable parameters and quality training but uniform gates over available sensors.",
              "- Bounded SQIs and exact missing-sensor masks remove numerical extrapolation and guarantee zero weight for an absent modality. Both absent causes abstention. No manual-quality annotation is an inference input.",
              "- BP gradients do not train the quality estimator; encoder features feeding quality estimation are detached. Quality is trained with class-balanced soft manual BCE and relative corruption-ranking loss.",
              "- Synthetic measurement corruption retains the observed BP reference. Added noise is not an additional patient or physiological observation; ranking is a degradation proxy, not a synthetic human quality label.",
              "- Noise/SNR and missing-sensor families are seen during training, with independent parameters/draws on held-out subjects. Baseline drift, motion-band noise, and clipping were not used for training or inner candidate selection; these remain synthetic stress tests.",
              "- Manual quality describes partially different raw windows and is imperfect. The cohort remains the local QC subset, not the complete documented dataset.",
              "- Bootstrap conditions on fitted OOF predictions; it does not capture all model-selection/training-population uncertainty or dependencies from overlapping fold training sets. No multiplicity correction is applied.",
              "- Three seeds describe repeated initialization on fixed folds. Report all folds and seeds, not a selected best run. There is no guarantee of improved accuracy or clinical validity.",
              "- Final refit uses all development participants and an epoch budget selected from inner validation only. It has no additional independent accuracy result; reported accuracy belongs to the nested procedure.",
              "- Benchmarks include environment, warm-up, repetitions, model size/MACs, throughput, and isolated process RSS. This is offline 24-second processing, not validated streaming hardware.", ""]
    (output / "OPTIMIZATION_REPORT.md").write_text("\n".join(lines))
    LOGGER.info("Nested OOF comparison\n%s", comparison.to_string(index=False))


def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config.update({"quick": True, "seeds": [17], "epochs": 2, "patience": 2,
                       "bootstrap_repetitions": 100, "latency_warmup": 2, "latency_repetitions": 10})
    else:
        config["quick"] = False
    config["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    output = Path(args.output or ("results/optimization_smoke" if args.quick else "results/optimization"))
    output.mkdir(parents=True, exist_ok=True)
    for name in ["logs", "tables", "predictions"]:
        (output / name).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "logs/optimization.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    torch.set_num_threads(config["threads"])
    root = Path(args.data).resolve()
    cohort, arrays, audit = prepare_dataset(root, args.cache, DEFAULT_PREPROCESSING)
    retired_split = json.loads(Path(config["retired_test_split"]).read_text())["participant_ids"]
    retired = set(retired_split["test"])
    allowed = set(retired_split["train"]) | set(retired_split["validation"])
    if allowed & retired or not set(cohort.subject_id).issubset(allowed | retired):
        raise ValueError("Ambiguous historical participant roles")
    if args.quick:
        pool = np.asarray(sorted(allowed))
        np.random.default_rng(config["protocol_seed"]).shuffle(pool)
        allowed = set(pool[:90])
    use = np.flatnonzero(cohort.subject_id.isin(allowed))
    cohort = cohort.iloc[use].reset_index(drop=True)
    arrays = subset(arrays, use)
    if set(cohort.subject_id) & retired:
        raise AssertionError("Historical test leakage")
    folds = nested_subject_splits(cohort, config)
    protocol = {"config": config, "source_sha256": code_hash(), "environment": environment(), "dataset_fingerprint": audit["fingerprint"],
                "development_record_ids": cohort.record_id.tolist(), "development_subject_ids": sorted(cohort.subject_id.unique()),
                "retired_test_subject_ids": sorted(retired), "outer_folds": [{"fold": i, "participant_ids": f["participant_ids"]} for i, f in enumerate(folds)],
                "status_of_evidence": "post-hoc nested development evaluation; not an untouched independent confirmatory test"}
    run_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (output / "protocol.json").exists() and json.loads((output / "protocol.json").read_text())["run_sha256"] != run_hash:
        raise ValueError("Code/config/environment/data changed: use a new output directory")
    save_json(output / "protocol.json", {"run_sha256": run_hash, **protocol})
    save_json(output / "resolved_config.json", config)
    save_json(output / "tables/dataset_summary.json", audit)
    cohort.to_csv(output / "tables/development_cohort.csv", index=False)
    snapshot = output / "source_snapshot"
    (snapshot / "src").mkdir(parents=True, exist_ok=True)
    for path in Path("src").glob("*.py"):
        shutil.copy2(path, snapshot / "src" / path.name)
    for filename in ["main.py", "requirements.lock.txt"]:
        shutil.copy2(filename, snapshot / filename)
    save_json(output / "status.json", {"status": "building_caches", "run_sha256": run_hash})
    banks = build_banks(root, cohort, config, Path(args.cache) / ("optimization_smoke" if args.quick else "optimization"), audit["fingerprint"])
    save_json(output / "status.json", {"status": "inner_tuning", "run_sha256": run_hash})
    selections = tune_all_folds(cohort, arrays, banks, folds, config, output, run_hash)
    if args.phase == "tune":
        save_json(output / "status.json", {"status": "selection_locked", "run_sha256": run_hash})
        return
    if json.loads((output / "selection_locked.json").read_text())["source_sha256"] != code_hash():
        raise ValueError("Source changed after hyperparameter selection")
    save_json(output / "status.json", {"status": "outer_evaluation", "run_sha256": run_hash})
    evaluate_all_folds(cohort, arrays, banks, folds, selections, config, output, root, run_hash)
    summary, efficiency, predictions = collect_results(cohort, config, output)
    trials = pd.read_csv(output / "tables/inner_selection_trials.csv")
    global_candidate = int(trials.groupby("candidate").selection_score.mean().idxmin())
    quality_weight = config["quality_loss_candidates"][global_candidate]
    epochs = max(1, int(np.median(trials[trials.candidate == global_candidate].best_epoch)))
    final_config = {**config, "quality_loss_weight": quality_weight}
    final_directory = output / "final" / "optimized_quality"
    if not (final_directory / "training_complete.json").exists():
        fit_new_model("optimized_quality", config["seeds"][0], cohort, arrays, banks, {"train": np.arange(len(cohort))}, final_config, final_directory,
                      {"run_sha256": run_hash, "phase": "final_development_refit", "subject_ids": sorted(cohort.subject_id.unique()),
                       "selection_source": "mean INNER validation score only", "quality_loss_weight": quality_weight}, fixed_epochs=epochs)
    save_json(output / "final/refit_selection.json", {"candidate": global_candidate, "quality_loss_weight": quality_weight, "epochs": epochs,
                                                     "selection_source": "inner validation only; outer metrics not used", "independent_accuracy_available": False})
    render_report(summary, efficiency, config, selections, output, cohort)
    save_json(output / "status.json", {"status": "complete", "run_sha256": run_hash, "outer_model_seed_runs": len(config["models"])*len(config["seeds"])*len(folds),
                                       "development_records": len(cohort), "development_subjects": int(cohort.subject_id.nunique()), "retired_test_subjects": len(retired),
                                       "quick": args.quick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.json")
    parser.add_argument("--data", default="physionet.org/files/cardiowave-portable/1.0.0")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--phase", choices=["all", "tune"], default="all")
    run(parser.parse_args())
