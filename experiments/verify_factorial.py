"""Verify frozen sources, preserved historical artifacts, local data and actual factorial outputs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from src.utils import code_hash, save_json


def main():
    output = Path("results/factorial")
    status = json.loads((output/"status.json").read_text())
    protocol = json.loads((output/"protocol.json").read_text())
    assert status["status"] == "complete" and not status["quick"]
    assert code_hash() == protocol["source_sha256"]
    old = json.loads(Path("experiments/factorial_historical_artifact_hashes.json").read_text())
    for filename, digest in old.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, f"Historical artifact changed: {filename}"
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.txt"], cwd="physionet.org/files/cardiowave-portable/1.0.0", check=True)
    test = subprocess.run([".venv/bin/python", "-m", "pytest", "-q"], capture_output=True, text=True)
    (output/"logs/test_verification.log").write_text(test.stdout+test.stderr)
    test.check_returncode()
    assert "skipped" not in test.stdout
    passed = int(re.search(r"(\d+) passed", test.stdout).group(1))
    snapshot = output/"verification_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copytree("tests", snapshot/"tests", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2("experiments/factorial_historical_artifact_hashes.json", snapshot/"historical_artifact_hashes.json")
    save_json(output/"verification.json", {"automated_tests_passed": passed, "automated_tests_failed": 0,
              "dataset_SHA256_manifest_verified": True, "historical_artifacts_verified_unchanged": len(old),
              "source_sha256_matches_protocol": True, "neural_fits": status["neural_fits"],
              "development_subjects": status["development_subjects"], "development_records": status["development_records"],
              "historical_test_subjects_excluded": status["retired_test_subjects"],
              "all_checkpoint_and_threshold_selections_locked_before_outer": True,
              "train_only_scalers_and_constant_baselines_verified": True,
              "outer_reference_BP_unchanged_in_perturbations": True,
              "primary_family_conditional_bootstrap_correction": "4 contrasts, Bonferroni 98.75% individual percentile intervals",
              "previously_analyzed_cohort": True, "independent_confirmatory_test": False,
              "external_validation": False, "clinical_validation": False,
              "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(test.stdout)
    print("Verified factorial outputs, original data checksums and historical artifacts.")


if __name__ == "__main__":
    main()
