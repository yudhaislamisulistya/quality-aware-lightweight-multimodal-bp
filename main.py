"""Local-only, participant-independent research entrypoint."""
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from src.ablation import robustness_inputs
from src.dataset_loader import QUALITY, TARGETS, participant_split, prepare_dataset, read_waveform
from src.evaluate import cluster_bootstrap, paired_comparisons, prediction_frame, quality_strata, regression_metrics
from src.preprocessing import DEFAULT_PREPROCESSING, preprocess
from src.train import load_model, predict, train_model
from src.utils import code_hash, environment, save_json
from src.visualization import dataset_figures, research_report, result_figures

DEFAULT_DATA = "physionet.org/files/cardiowave-portable/1.0.0"


def write_audit_outputs(cache, output, root, cohort, arrays):
    (output / "tables").mkdir(exist_ok=True)
    for filename in ["dataset_summary.json", "exclusions.csv", "duplicate_waveforms.csv", "extra_waveforms.json", "target_outlier_flags_retained.csv", "sqi_by_manual_quality.csv", "cohort.csv"]:
        shutil.copy2(cache / filename, output / "tables" / filename)
    dataset_figures(root, cohort, arrays, output / "figures")


def inference(checkpoint_path, root, record):
    model, checkpoint = load_model(checkpoint_path)
    torch.set_num_threads(checkpoint["config"]["threads"])
    x, sqi, automatic = preprocess(read_waveform(root, record), checkpoint["preprocessing"])
    if checkpoint.get("architecture_version") == "risk_v1":
        from src.risk_model import predict_components
        details = predict_components(model, {"x": x[None], "sqi": sqi[None], "automatic": automatic[None]}, checkpoint["scalers"])
        return {"SBP_mmHg": float(details["prediction"][0, 0]), "DBP_mmHg": float(details["prediction"][0, 1]),
                "estimated_goodness_ECG_PPG": details["goodness"][0].tolist(),
                "fusion_weights_by_target": {t: details["weights"][0, :, c].tolist() for c, t in enumerate(["SBP", "DBP"])},
                "estimated_expert_absolute_error_mmHg": {m: details["risk"][0, c].tolist() for c, m in enumerate(["ECG", "PPG"])},
                "fused_error_proxy_mmHg": details["risk_proxy"][0].tolist(),
                "risk_interpretation": "cross-fitted error estimate, not a validated uncertainty bound",
                "manual_quality_used": False, "clinical_validation": False}
    values, quality, gates = predict(model, {"x": x[None], "sqi": sqi[None], "automatic": automatic[None]}, checkpoint["scalers"])
    return {"SBP_mmHg": float(values[0, 0]), "DBP_mmHg": float(values[0, 1]),
            "estimated_goodness_ECG_PPG": quality[0].tolist() if model.supervised_quality else None,
            "automatic_heuristic_goodness_ECG_PPG": automatic.tolist(), "fusion_weights_ECG_PPG": gates[0].tolist(),
            "manual_quality_used": False, "clinical_validation": False}


def evaluate_run(checkpoint_path, test, inputs, variants, config, directory, root):
    model, checkpoint = load_model(checkpoint_path, config["device"])
    scalers = checkpoint["scalers"]
    actual = test[TARGETS].to_numpy()
    predictions, qualities, gates = predict(model, inputs, scalers, config["device"], config["batch_size"])
    frame = prediction_frame(test, predictions, qualities, gates)
    if not model.supervised_quality:
        for c, prefix in enumerate(["ecg", "ppg"]):
            frame[prefix + "_estimated_goodness"] = inputs["automatic"][:, c] if model.name == "quality_fusion" else np.nan
    frame.to_csv(directory / "predictions.csv", index=False)
    metrics = regression_metrics(actual, predictions)
    for row in metrics:
        c = 0 if row["target"] == "SBP" else 1
        row["participant_macro_MAE"] = float(pd.Series(np.abs(predictions[:, c] - actual[:, c])).groupby(test.subject_id.to_numpy()).mean().mean())
        row["n_participants"] = int(test.subject_id.nunique())
    save_json(directory / "metrics.json", metrics)
    pd.DataFrame(cluster_bootstrap(actual, predictions, test.subject_id.to_numpy(), config["bootstrap_repetitions"], config["split_seed"])).to_csv(directory / "bootstrap_ci.csv", index=False)
    pd.DataFrame(quality_strata(test, predictions)).to_csv(directory / "quality_stratified.csv", index=False)
    diagnostics = []
    robust_rows = []
    for condition, arrays in {"clean": inputs, **variants}.items():
        if condition == "clean":
            p, q, weights = predictions, qualities, gates
        else:
            p, q, weights = predict(model, arrays, scalers, config["device"], config["batch_size"])
            corrupt_frame = prediction_frame(test, p, q, weights)
            # Original manual labels describe clean signals and are not noise-condition targets.
            corrupt_frame = corrupt_frame.drop(columns=QUALITY)
            if not model.supervised_quality:
                corrupt_frame = corrupt_frame.drop(columns=["ecg_estimated_goodness", "ppg_estimated_goodness"])
            corrupt_frame.to_csv(directory / f"predictions_{condition}.csv", index=False)
        for metrics_row in regression_metrics(actual, p):
            robust_rows.append({"condition": condition, "mean_ecg_gate": float(weights[:, 0].mean()),
                                "mean_ppg_gate": float(weights[:, 1].mean()), **metrics_row})
        if model.supervised_quality:
            for c, modality in enumerate(["ECG", "PPG"]):
                goodness = 1 - test[QUALITY[c]].to_numpy() / 3
                diagnostics.append({"condition": condition, "modality": modality, "mean_estimated_goodness": float(q[:, c].mean()),
                                    "mean_gate": float(weights[:, c].mean()),
                                    "MAE_vs_manual_clean_goodness": float(np.abs(q[:, c] - goodness).mean()) if condition == "clean" else None})
    pd.DataFrame(robust_rows).to_csv(directory / "robustness.csv", index=False)
    pd.DataFrame(diagnostics, columns=["condition", "modality", "mean_estimated_goodness", "mean_gate", "MAE_vs_manual_clean_goodness"]).to_csv(directory / "quality_estimator.csv", index=False)
    del model
    command = [sys.executable, "-m", "src.efficiency", str(checkpoint_path), str(root), test.iloc[0].wfdb_record_path,
               str(directory / "efficiency.json"), "--threads", str(config["threads"]), "--warmup", str(config["latency_warmup"]),
               "--repetitions", str(config["latency_repetitions"]), "--device", config["device"]]
    subprocess.run(command, check=True)
    save_json(directory / "complete.json", {"completed": True, "test_records": len(test)})


def aggregate_results(output, config, audit, split_sizes):
    rows, frames, histories = [], {}, {}
    collections = {name: [] for name in ["quality_stratified", "robustness", "bootstrap_ci", "quality_estimator"]}
    for name in config["models"]:
        frames[name] = []
        for seed in config["seeds"]:
            directory = output / "checkpoints" / name / f"seed_{seed}"
            metrics = json.loads((directory / "metrics.json").read_text())
            efficiency = json.loads((directory / "efficiency.json").read_text())
            history = pd.read_csv(directory / "history.csv")
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            row = {"model": name, "seed": seed, "parameters": efficiency["parameters"],
                   "model_size_KiB": efficiency["serialized_state_dict_bytes"] / 1024,
                   "weight_buffer_KiB": efficiency["weights_and_buffers_bytes"] / 1024,
                   "latency_ms": efficiency["model_only"]["batch_1"]["median_latency_ms"],
                   "latency_p95_ms": efficiency["model_only"]["batch_1"]["p95_latency_ms"],
                   "throughput_batch1_per_s": efficiency["model_only"]["batch_1"]["records_per_second"],
                   "throughput_batch32_per_s": efficiency["model_only"]["batch_32"]["records_per_second"],
                   "peak_RSS_MiB": efficiency["isolated_process_peak_RSS_MiB"],
                   "peak_CUDA_MiB": efficiency["peak_CUDA_allocated_MiB"],
                   "MACs": efficiency["MACs_per_record"], "approx_FLOPs": efficiency["approx_FLOPs_per_record"],
                   "end_to_end_ms": efficiency["end_to_end_median_ms"], "device": efficiency["device_name"],
                   "best_epoch": checkpoint["epoch"], "epochs_run": len(history), "n_seeds_planned": len(config["seeds"])}
            for metric in metrics:
                for key in ["MAE", "RMSE", "R2", "ME", "SD_error", "LoA_lower", "LoA_upper", "participant_macro_MAE"]:
                    row[f"{metric['target']}_{key}"] = metric[key]
            rows.append(row)
            for table in collections:
                table_frame = pd.read_csv(directory / f"{table}.csv")
                table_frame["model"], table_frame["seed"] = name, seed
                collections[table].append(table_frame)
            frames[name].append(pd.read_csv(directory / "predictions.csv"))
            if seed == config["seeds"][0]:
                histories[name] = history
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "tables/per_seed_results.csv", index=False)
    summaries = []
    for name in config["models"]:
        group = runs[runs.model == name]
        row = {"model": name, "n_seeds": len(group)}
        for col in runs.select_dtypes(include="number").columns:
            if col not in {"seed", "n_seeds_planned"}:
                row[col] = group[col].mean()
                row[col + "_seed_SD"] = group[col].std(ddof=1) if len(group) > 1 else np.nan
        row["device"] = group.device.iloc[0]
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "tables/baseline_comparison.csv", index=False)
    summary.to_csv(output / "tables/ablation_results.csv", index=False)
    summary[["model", "parameters", "model_size_KiB", "MACs", "approx_FLOPs", "latency_ms", "latency_p95_ms",
             "throughput_batch1_per_s", "throughput_batch32_per_s", "end_to_end_ms", "peak_RSS_MiB", "device"]].to_csv(output / "tables/efficiency_comparison.csv", index=False)
    for table, values in collections.items():
        pd.concat(values, ignore_index=True).to_csv(output / f"tables/{table}.csv", index=False)
    paired_comparisons(frames, config["bootstrap_repetitions"], config["split_seed"]).to_csv(output / "tables/paired_MAE_differences.csv", index=False)
    result_figures(output, summary, frames, histories)
    research_report(output, summary, audit, config, split_sizes)
    return summary


def run(args):
    root, cache = Path(args.data).resolve(), Path(args.cache).resolve()
    output = Path(args.output or f"results/{args.command}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "logs/pipeline.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    cohort, arrays, audit = prepare_dataset(root, cache, DEFAULT_PREPROCESSING)
    if args.command == "audit":
        write_audit_outputs(cache, output, root, cohort, arrays)
        logging.info("Audit complete: %d usable records / %d participants", len(cohort), cohort.subject_id.nunique())
        return
    config = json.loads(Path(args.config or f"configs/{args.command}.json").read_text())
    if config["mode"] != args.command:
        raise ValueError("Command and configuration mode disagree")
    config["preprocessing"] = DEFAULT_PREPROCESSING
    config["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    torch.set_num_threads(config["threads"])
    splits, ids = participant_split(cohort, config["split_seed"], config["quick_subjects"])
    split_sizes = {k: {"participants": len(ids[k]), "records": len(v)} for k, v in splits.items()}
    provenance = {"dataset_fingerprint": audit["fingerprint"], "source_sha256": code_hash(), "config": config, "participant_ids": ids, "environment": environment()}
    run_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = {"run_sha256": run_hash, **provenance}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text())["run_sha256"] != run_hash:
        raise ValueError("Output belongs to a different code/config/data/environment. Choose a new --output directory.")
    write_audit_outputs(cache, output, root, cohort, arrays)
    save_json(output / "manifest.json", manifest)
    save_json(output / "resolved_config.json", config)
    save_json(output / "participant_splits.json", {"participant_ids": ids, "sizes": split_sizes})
    save_json(output / "status.json", {"status": "running", "mode": args.command})
    snapshot = output / "source_snapshot"
    (snapshot / "src").mkdir(parents=True, exist_ok=True)
    for path in Path("src").glob("*.py"):
        shutil.copy2(path, snapshot / "src" / path.name)
    for filename in ["main.py", "requirements.lock.txt"]:
        shutil.copy2(filename, snapshot / filename)
    test = cohort.iloc[splits["test"]].reset_index(drop=True)
    inputs = {key: value[splits["test"]] for key, value in arrays.items()}
    variants = robustness_inputs(root, test, config["preprocessing"], config["split_seed"])
    train_mean = cohort.iloc[splits["train"]][TARGETS].mean().to_numpy()
    pd.DataFrame(regression_metrics(test[TARGETS].to_numpy(), np.tile(train_mean, (len(test), 1)))).to_csv(output / "tables/training_mean_baseline.csv", index=False)
    for name in config["models"]:
        for seed in config["seeds"]:
            directory = output / "checkpoints" / name / f"seed_{seed}"
            if (directory / "complete.json").exists():
                logging.info("Reusing completed identical run: %s seed %s", name, seed)
                continue
            checkpoint = train_model(name, seed, cohort, arrays, splits, config, directory, {"run_sha256": run_hash, "dataset_fingerprint": audit["fingerprint"], "participant_ids": ids})
            evaluate_run(checkpoint, test, inputs, variants, config, directory, root)
    summary = aggregate_results(output, config, audit, split_sizes)
    save_json(output / "status.json", {"status": "complete", "mode": args.command, "completed_runs": len(config["models"]) * len(config["seeds"]), "run_sha256": run_hash})
    logging.info("Completed %s\n%s", args.command, summary[["model", "SBP_MAE", "DBP_MAE", "parameters", "latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "quick", "full", "predict", "report"])
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--record", help="Dataset-relative WFDB record path without extension")
    args = parser.parse_args()
    if args.command == "report":
        if not args.output:
            parser.error("report requires --output pointing to a completed suite")
        output = Path(args.output)
        status = json.loads((output / "status.json").read_text())
        if status["status"] != "complete":
            parser.error("report requires a completed suite")
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        config = json.loads((output / "resolved_config.json").read_text())
        audit = json.loads((output / "tables/dataset_summary.json").read_text())
        sizes = json.loads((output / "participant_splits.json").read_text())["sizes"]
        aggregate_results(output, config, audit, sizes)
        # Preserve executed training source; record later figure/report revisions separately.
        snapshot = output / "postprocessing_snapshot"
        (snapshot / "src").mkdir(parents=True, exist_ok=True)
        for path in Path("src").glob("*.py"):
            shutil.copy2(path, snapshot / "src" / path.name)
        shutil.copy2("main.py", snapshot / "main.py")
        save_json(output / "postprocessing_manifest.json", {"training_run_sha256": status["run_sha256"], "analysis_source_sha256": code_hash(), "environment": environment()})
    elif args.command == "predict":
        if not args.checkpoint or not args.record:
            parser.error("predict requires --checkpoint and --record")
        print(json.dumps(inference(args.checkpoint, args.data, args.record), indent=2, allow_nan=False))
    else:
        run(args)
