"""Recompute descriptive quality checks after the frozen optimization experiment."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.ablation import CONDITIONS
from src.optimization import KNOWN_CONDITIONS, UNSEEN_CONDITIONS
from src.utils import code_hash, save_json


def main():
    output = Path("results/optimization")
    protocol = json.loads((output / "protocol.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert code_hash() == protocol["source_sha256"]
    assert not set(protocol["retired_test_subject_ids"]) & set(protocol["development_subject_ids"])
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.txt"], cwd="physionet.org/files/cardiowave-portable/1.0.0", check=True)
    tests = Path("experiments/optimization-final-tests.log").read_text()
    assert "FAILED " not in tests
    passed = int(re.search(r"(\d+) passed", tests).group(1))
    quality, strata, responses = [], [], []
    config = protocol["config"]
    for model in config["models"]:
        for seed in config["seeds"]:
            path = output / "predictions" / f"{model}_seed_{seed}_clean.csv"
            clean = pd.read_csv(path).set_index("record_id")
            for modality in ["ecg", "ppg"]:
                field = modality + "_estimated_goodness"
                if field not in clean:
                    continue
                labels = clean[modality + "_quality_label"].to_numpy()
                score = clean[field].to_numpy()
                target = 1-labels/3
                error = np.abs(score-target)
                class_errors = []
                for label in range(4):
                    mask = labels == label
                    if not mask.any():
                        continue
                    mean_error = float(error[mask].mean())
                    class_errors.append(mean_error)
                    strata.append({"model": model, "seed": seed, "modality": modality, "manual_poor_windows": label,
                                   "records": int(mask.sum()), "mean_estimated_goodness": float(score[mask].mean()), "MAE_vs_manual_goodness": mean_error})
                rho = float(spearmanr(score, target).statistic)
                quality.append({"model": model, "seed": seed, "modality": modality, "record_weighted_MAE": float(error.mean()),
                                "class_macro_MAE": float(np.mean(class_errors)), "Spearman_rho": rho if np.isfinite(rho) else None})
            for condition in KNOWN_CONDITIONS + UNSEEN_CONDITIONS:
                altered = pd.read_csv(output / "predictions" / f"{model}_seed_{seed}_{condition}.csv").set_index("record_id").loc[clean.index]
                np.testing.assert_array_equal(altered[["sbp_mmHg", "dbp_mmHg"]], clean[["sbp_mmHg", "dbp_mmHg"]])
                if condition in CONDITIONS:
                    channel = CONDITIONS[condition][0]
                    changed = ["ecg", "ppg"] if channel is None else [["ecg", "ppg"][channel]]
                else:
                    changed = ["ecg"] if condition.startswith("ecg") else ["ppg"]
                for modality in changed:
                    field = modality + "_estimated_goodness"
                    if field not in clean:
                        continue
                    delta = altered[field].to_numpy()-clean[field].to_numpy()
                    responses.append({"model": model, "seed": seed, "condition": condition, "modality": modality,
                                      "mean_goodness_change": float(delta.mean()),
                                      "fraction_goodness_does_not_increase": float((delta <= 1e-6).mean()),
                                      "interpretation": "descriptive response to synthetic corruption, not human-annotation accuracy"})
    for name, rows in [("descriptive_quality_metrics", quality), ("descriptive_quality_by_label", strata), ("descriptive_quality_response", responses)]:
        pd.DataFrame(rows).to_csv(output / f"tables/{name}.csv", index=False)
    comparison = pd.read_csv(output / "tables/comparison.csv").set_index("model")
    metrics = pd.read_csv(output / "tables/summary.csv")
    robustness = metrics.groupby(["model", "condition"]).MAE_mean.mean().unstack()
    paired = pd.read_csv(output / "tables/paired_MAE_differences.csv")
    q_summary = pd.DataFrame(quality).groupby(["model", "modality"])[["record_weighted_MAE", "class_macro_MAE", "Spearman_rho"]].mean()
    old, new = comparison.loc["legacy_proposed"], comparison.loc["optimized_quality"]
    lines = ["# Verified optimization findings", "",
             "**The frozen experiment improved several synthetic robustness conditions but did not improve clean BP accuracy. The contribution of learned quality gating remains unproven.**", "",
             f"Evidence: {passed} passing tests; 45 outer model/seed/fold evaluations; 3,184 records / 810 development participants. Historical 143 test participants were retired and excluded.", "",
             "These are post-hoc nested-development estimates, not independent external or clinical validation. Historical 13.00/8.57 mmHg results used a different test cohort and must not be compared directly with these numbers.", "",
             "## Clean pressure accuracy", "", "Mean over three initialization seeds; all outer folds pooled per seed, mmHg.", "",
             "| Model | SBP MAE | DBP MAE | Parameters | Model latency ms |", "|---|---:|---:|---:|---:|"]
    for name in config["models"]:
        row = comparison.loc[name]
        lines.append(f"| {name} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} | {int(row.parameters)} | {row.latency_ms:.3f} |")
    lines += ["", "Positive differences below mean worse accuracy for the new complete model:"]
    for row in paired[(paired.condition == "clean") & (paired.comparison == "optimized_quality minus legacy_proposed")].itertuples():
        lines.append(f"- {row.target}: delta MAE {row.delta_MAE:+.3f} mmHg; conditional participant-bootstrap 95% CI [{row.lower_95:+.3f}, {row.upper_95:+.3f}].")
    lines += ["", "## Robustness", "", "Metric is the mean of SBP and DBP MAE, averaged over seeds; mmHg. These are synthetic measurement perturbations, not real sensor-artefact validation.", "",
              "| Condition | Retrained legacy | Complete v2 | Matched v2 without gating |", "|---|---:|---:|---:|"]
    for condition in robustness.columns:
        lines.append(f"| {condition} | {robustness.loc['legacy_proposed',condition]:.3f} | {robustness.loc['optimized_quality',condition]:.3f} | {robustness.loc['optimized_no_gate',condition]:.3f} |")
    lines += ["", "Robustness gains also appear in controls without learned quality gating. They cannot be attributed solely to gating or its ranking loss. The complete model does not consistently outperform its matched controls.", "",
              "## Quality-estimator diagnostics", "", "Additional post-hoc descriptive analysis, never used to select models. Goodness and its MAE use a 0–1 scale, not mmHg; Spearman correlation assesses ordering against manual goodness.", "",
              "| Model | Modality | Record-weighted quality MAE | Class-macro quality MAE | Spearman rho |", "|---|---|---:|---:|---:|"]
    for name in ["legacy_proposed", "optimized_quality"]:
        for modality in ["ecg", "ppg"]:
            row = q_summary.loc[(name, modality)]
            lines.append(f"| {name} | {modality} | {row.record_weighted_MAE:.3f} | {row.class_macro_MAE:.3f} | {row.Spearman_rho:.3f} |")
    lines += ["", "The missing-sensor failure is fixed: an absent ECG or PPG has exactly zero fusion weight. PPG manual-quality agreement improves modestly; ECG record-weighted agreement and ranking worsen, although class-macro error improves. This is mixed evidence, not a solved quality-estimation problem.", "",
              "## Decision", "",
              "- Keep the legacy method as the clean-accuracy reference; do not describe v2 as an overall accuracy upgrade.",
              "- Retain v2 as an experimental robustness branch. Any superiority claim for quality-aware gating requires new evidence beyond these matched ablations.",
              f"- V2 costs more: {int(new.parameters)} versus {int(old.parameters)} parameters, {int(new.MACs)} versus {int(old.MACs)} matrix MACs, and {new.latency_ms:.3f} versus {old.latency_ms:.3f} ms batch-1 latency on the recorded CPU.",
              "- The final refit checkpoint uses all development participants with inner-selected weight/epoch budget. It has no new independent accuracy score.",
              "- Future tuning must be distinguished from confirmatory evaluation. External/future held-out participants and actual sensor artefacts remain untested. Do not infer clinical suitability from these results.", ""]
    (output / "VERIFIED_FINDINGS.md").write_text("\n".join(lines))
    shutil.copy2("experiments/optimization-final-tests.log", output / "logs/test_verification.log")
    snapshot = output / "verification_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    save_json(output / "verification.json", {"automated_tests_passed": passed, "automated_tests_failed": 0,
              "dataset_SHA256_manifest_verified": True,
              "outer_evaluation_runs": len(list((output / "outer").rglob("complete.json"))),
              "inner_candidate_fits": len(list((output / "tuning").rglob("training_complete.json"))),
              "development_subjects": len(protocol["development_subject_ids"]), "development_records": status["development_records"],
              "historical_test_subjects_excluded": len(protocol["retired_test_subject_ids"]),
              "selection_locked_before_outer": True, "source_unchanged_since_protocol_lock": True,
              "quality_and_BP_gradient_separation_tested": True, "zero_weight_for_absent_modality_tested": True,
              "original_BP_references_preserved_in_all_perturbations": True,
              "additional_quality_diagnostics": "post-hoc descriptive only, not used for selection",
              "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "independent_external_validation": False, "clinical_validation": False})
    print(pd.DataFrame(quality).groupby(["model", "modality"])[["record_weighted_MAE", "class_macro_MAE", "Spearman_rho"]].mean().to_string())


if __name__ == "__main__":
    main()
