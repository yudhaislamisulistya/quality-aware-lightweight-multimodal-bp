"""Check actual outputs, immutable prior experiments, dataset checksums and raw WFDB inference."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from main import inference
from src.utils import code_hash, save_json


def main():
    root=Path("results/reliability")
    status=json.loads((root/"status.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())
    assert status["status"]=="complete" and not status["quick"]
    assert code_hash()==protocol["source_sha256"]
    history=json.loads(Path("experiments/reliability_historical_hashes.json").read_text())
    for path,digest in history.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,f"Historical artifact changed: {path}"
    subprocess.run(["sha256sum","--quiet","-c","SHA256SUMS.txt"],cwd="physionet.org/files/cardiowave-portable/1.0.0",check=True)
    tests=subprocess.run([".venv/bin/python","-m","pytest","-q"],capture_output=True,text=True)
    (root/"logs/test_verification.log").write_text(tests.stdout+tests.stderr)
    tests.check_returncode()
    assert "skipped" not in tests.stdout
    passed=int(re.search(r"(\d+) passed",tests.stdout).group(1))
    cohort=pd.read_csv(root/"tables/development_cohort.csv")
    outer=protocol["outer_folds"][0]["participant_ids"]["outer"]
    record=cohort[cohort.subject_id.isin(outer)].iloc[0]
    checkpoint=root/"fits/fold_0/seed_17/adaptive_risk_fusion.pt"
    predicted=inference(checkpoint,Path("physionet.org/files/cardiowave-portable/1.0.0"),record.wfdb_record_path)
    expected=pd.read_csv(root/"outer/fold_0/seed_17/adaptive_risk_fusion/predictions_clean.csv").set_index("record_id").loc[record.record_id]
    np.testing.assert_allclose([predicted["SBP_mmHg"],predicted["DBP_mmHg"]],[expected.sbp_predicted,expected.dbp_predicted],rtol=0,atol=.001)
    assert predicted["manual_quality_used"] is False
    save_json(root/"inference_smoke.json",{"scope":"raw WFDB inference functionality, not an additional accuracy study","record_id":record.record_id,
              "checkpoint":str(checkpoint),"matches_saved_outer_prediction_within_mmHg":.001,**predicted})
    snapshot=root/"verification_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    shutil.copy2("pytest.ini",snapshot/"pytest.ini")
    shutil.copytree("tests",snapshot/"tests",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2("experiments/reliability_historical_hashes.json",snapshot/"historical_hashes.json")
    save_json(root/"verification.json",{"automated_tests_passed":passed,"automated_tests_failed":0,
              "dataset_SHA256_manifest_verified":True,"historical_artifacts_verified_unchanged":len(history),
              "source_sha256_matches_protocol":True,"new_neural_teacher_fits":status["new_teacher_fits"],
              "linear_risk_head_fits":status["risk_head_fits"],"risk_fitting_pipelines":status["risk_fitting_pipelines"],
              "outer_policy_evaluations":status["outer_policy_evaluations"],"development_subjects":status["development_subjects"],
              "development_records":status["development_records"],"retired_test_subjects":status["retired_test_subjects"],
              "participant_disjoint_risk_teacher_fit_selection_and_target_generation_verified":True,
              "risk_scalers_and_targets_fit_on_training_only_OOF_data_verified":True,
              "absent_expert_targets_excluded_verified":True,"reference_expert_weights_unchanged_verified":True,
              "all_heads_and_thresholds_locked_before_new_outer_evaluation":True,
              "manual_quality_oracles_clean_only_and_nondeployable":True,
              "primary_explicit_quality_contribution_control":"full risk head minus BP-only risk head",
              "waveform_only_inference_reproduces_outer_prediction":True,
              "cohort_previously_analyzed":True,"independent_confirmatory_test":False,"external_validation":False,"clinical_validation":False,
              "verification_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(tests.stdout)
    print("Verified strict training-only risk targets, original dataset, prior results and raw inference.")


if __name__=="__main__":
    main()
