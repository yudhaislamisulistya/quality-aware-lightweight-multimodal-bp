"""Participant provenance, recomputed metrics and real inference artifacts for the risk study."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.reliability import TRAIN_CONDITIONS
from src.risk_model import POLICIES


@pytest.mark.parametrize("mode", ["reliability_smoke", "reliability"])
def test_executed_reliability_study(mode):
    root=Path("results")/mode
    if not (root/"status.json").exists() or json.loads((root/"status.json").read_text())["status"] != "complete":
        pytest.skip(f"{mode} has not completed")
    protocol=json.loads((root/"protocol.json").read_text())
    config=protocol["config"]
    lock_path=root/"selection_locked.json"
    lock=json.loads(lock_path.read_text())
    seal=hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert lock["source_sha256"] == protocol["source_sha256"]
    assert not protocol["independent_confirmatory_test"] and not lock["outer_predictions_read"]
    assert not lock["validation_targets_fit_risk_head"]
    cohort=pd.read_csv(root/"tables/development_cohort.csv")
    assert not set(cohort.subject_id)&set(protocol["retired_test_subject_ids"])
    reference=cohort.set_index("record_id")
    for selection in lock["selections"]:
        fold,seed=selection["fold"],selection["seed"]
        groups=protocol["outer_folds"][fold]["participant_ids"]
        train_ids=set(groups["train"])
        excluded=set(groups["validation"])|set(groups["outer"])
        training=root/"risk_training"/f"fold_{fold}"/f"seed_{seed}"
        records=pd.read_csv(training/"risk_training_records.csv")
        assert set(records.subject_id)==train_ids and not set(records.subject_id)&excluded
        teachers=json.loads((training/"teacher_provenance.json").read_text())
        for teacher in teachers:
            roles=teacher["participant_ids"]
            seen=set(roles["train"])|set(roles["validation"])
            held=set(teacher["heldout_subject_ids"])
            assert not seen&held and seen|held==train_ids
            assert not set(roles["train"])&set(roles["validation"])
            assert not (seen|held)&excluded
            path=Path(teacher["checkpoint"])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==teacher["sha256"]
            checkpoint=torch.load(path,map_location="cpu",weights_only=True)
            y=records[records.subject_id.isin(roles["train"])][["sbp_mmHg","dbp_mmHg"]].to_numpy()
            np.testing.assert_allclose(checkpoint["scalers"]["y_mean"],y.mean(0),atol=1e-4,rtol=0)
        with np.load(training/"oof_risk_data.npz",allow_pickle=False) as data:
            available=data["available"]
            actual=records[["sbp_mmHg","dbp_mmHg"]].to_numpy()
            expected=np.abs(data["expert_prediction"]-actual[None,:,None,:])
            expected=np.where(available[...,None],expected,np.nan)
            np.testing.assert_allclose(data["observed_absolute_error"],expected,atol=1e-4,rtol=0,equal_nan=True)
            assert np.isnan(data["observed_absolute_error"][~available]).all()
            np.testing.assert_allclose(data["features"][:,:,-2:],available)
            np.testing.assert_allclose(data["features"][:,:,18:22],data["expert_prediction"].reshape(6,len(records),4))
            for cross in protocol["training_only_crossfit_partitions"][fold]["partitions"]:
                held=data["crossfit_fold"]==cross["crossfit_fold"]
                assert set(records.loc[held,"subject_id"])==set(cross["participant_ids"]["outer"])
            x=data["features"].reshape(-1,24).astype(np.float64)
            weights=np.repeat([config["risk_clean_sample_weight"]]+[(1-config["risk_clean_sample_weight"])/5]*5,len(records))
            expected_mean=np.average(x,weights=weights,axis=0)
        for policy,path_info in selection["checkpoints"].items():
            path=Path(path_info["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==path_info["sha256"]
            checkpoint=torch.load(path,map_location="cpu",weights_only=True)
            np.testing.assert_allclose(checkpoint["state_dict"]["feature_mean"],expected_mean,atol=1e-4,rtol=0)
            assert set(checkpoint["provenance"]["risk_fit_subject_ids"])==train_ids
            for c,expert in enumerate(checkpoint["provenance"]["reference_experts"]):
                original=torch.load(expert["path"],map_location="cpu",weights_only=True)
                for key,value in original["state_dict"].items():
                    torch.testing.assert_close(checkpoint["state_dict"][f"experts.{c}."+key],value,rtol=0,atol=0)
            if policy=="bp_only_risk_fusion":
                np.testing.assert_array_equal(checkpoint["state_dict"]["risk_head.weight"][:,:18],0)
            outer=root/"outer"/f"fold_{fold}"/f"seed_{seed}"/policy
            assert json.loads((outer/"complete.json").read_text())["selection_lock_sha256"]==seal
            for path in outer.glob("predictions_*.csv"):
                assert path.stat().st_mtime_ns>=lock_path.stat().st_mtime_ns
                frame=pd.read_csv(path)
                assert set(frame.subject_id)==set(groups["outer"])
                np.testing.assert_array_equal(frame[["sbp_mmHg","dbp_mmHg"]],reference.loc[frame.record_id,["sbp_mmHg","dbp_mmHg"]])
                for target in ["sbp","dbp"]:
                    weight=frame[[f"ecg_{target}_weight",f"ppg_{target}_weight"]]
                    np.testing.assert_allclose(weight.sum(axis=1),1.,atol=1e-6)
                    assert (weight.to_numpy()>=0).all()
                    expert=frame[[f"ecg_{target}_expert_prediction",f"ppg_{target}_expert_prediction"]]
                    assert (frame[target+"_predicted"]>=expert.min(axis=1)-1e-4).all()
                    assert (frame[target+"_predicted"]<=expert.max(axis=1)+1e-4).all()
                    for modality in ["ecg","ppg"]:
                        missing=~frame[modality+"_available"].to_numpy(bool)
                        np.testing.assert_array_equal(frame.loc[missing,f"{modality}_{target}_weight"],0)
                        assert frame.loc[missing,f"{modality}_{target}_estimated_risk"].isna().all()
        validation=root/"fits"/f"fold_{fold}"/f"seed_{seed}"/"inner_validation_components.npz"
        with np.load(validation,allow_pickle=False) as data:
            assert "actual" not in data.files
            for selector in ["risk","quality"]:
                score=data["risk_proxy"].mean(1) if selector=="risk" else data["goodness"].min(1)
                for coverage in config["selective_coverages"]:
                    expected=None if coverage==1 else np.quantile(score,coverage if selector=="risk" else 1-coverage)
                    stored=selection["thresholds"][selector][str(coverage)]
                    assert stored is None if expected is None else stored==pytest.approx(float(expected))
    metrics=pd.read_csv(root/"tables/metrics_per_seed.csv")
    for row in metrics.itertuples():
        frame=pd.read_csv(root/"predictions"/f"{row.model}_seed_{row.seed}_{row.condition}.csv")
        assert set(frame.record_id)==set(cohort.record_id) and not frame.record_id.duplicated().any()
        target=row.target.lower()
        error=frame[target+"_predicted"]-frame[target+"_mmHg"]
        assert row.MAE==pytest.approx(error.abs().mean())
        assert row.RMSE==pytest.approx(np.sqrt((error**2).mean()))
        assert row.ME==pytest.approx(error.mean())
        assert row.SD_error==pytest.approx(error.std(ddof=1))
    primary=pd.read_csv(root/"tables/fusion_contrasts.csv")
    primary=primary[primary.primary]
    assert len(primary)==4 and set(primary.comparison)=={config["primary_comparison"]}
    np.testing.assert_allclose(primary.adjusted_confidence,.9875)
    means=metrics.groupby(["model","condition","target"]).MAE.mean()
    for row in primary.itertuples():
        conditions=TRAIN_CONDITIONS[1:] if row.condition=="known_corruption_mean" else [row.condition]
        expected=np.mean([means.loc[("adaptive_risk_fusion",c,row.target)]-means.loc[("bp_only_risk_fusion",c,row.target)] for c in conditions])
        assert row.delta_MAE==pytest.approx(expected,abs=1e-12)
    decisions=pd.read_csv(root/"predictions/selective_clean_decisions.csv")
    for selector in ["risk","quality"]:
        part=decisions[(decisions.selector==selector)&(decisions.validation_target_coverage!=1)]
        expected=part.score<=part.threshold if selector=="risk" else part.score>=part.threshold
        np.testing.assert_array_equal(part.accepted,expected)
    assert decisions[decisions.validation_target_coverage==1].accepted.all()
    selective=pd.read_csv(root/"tables/selective_risk_coverage.csv")
    full=selective[selective.validation_target_coverage==1]
    np.testing.assert_allclose(full.selective_MAE,full.all_records_MAE)
    np.testing.assert_allclose(full.random_rejection_MAE_mean,full.all_records_MAE)
    assert len(list((root/"figures").glob("*.png")))>=13
    assert len(list((root/"figures").glob("*.pdf")))>=13
