import numpy as np
import pandas as pd

from src.factorial_model import FactorialBPModel
from src.parameter_audit import audit_models, parameter_breakdown
from scripts.prepare_development_split import roles_from_cohort


def test_stored_count_is_independent_of_frozen_state():
    model = FactorialBPModel("A_uniform_clean")
    before = parameter_breakdown(model)
    model.requires_grad_(False)
    after = parameter_breakdown(model)
    assert before["total_stored_parameters"] == after["total_stored_parameters"] == 36164
    assert before["requires_grad_parameters"] == 36164 and after["requires_grad_parameters"] == 0


def test_parameter_components_and_buffers_are_disjoint():
    rows = {r["model"]: r for r in audit_models()}
    assert rows["unimodal"]["total_stored_parameters"] == 17107
    assert rows["feature_A_to_D"]["total_stored_parameters"] == 36164
    assert rows["dense_feature_control"]["total_stored_parameters"] == 50452
    late = rows["late_fusion_all_policies"]
    assert [late[k] for k in ["encoders", "quality_heads", "bp_heads", "risk_head"]] == [30656, 1314, 2244, 100]
    assert late["total_stored_parameters"] == 34314
    assert late["buffer_elements_excluded"] == 60 and late["requires_grad_parameters"] == 0


def test_training_free_historical_roles_keep_all_records_and_subjects_disjoint():
    cohort = pd.DataFrame({"subject_id": np.repeat([f"test-{i:03}" for i in range(100)], 3)})
    plan = roles_from_cohort(cohort, 2026)
    sets = [set(plan["participant_ids"][k]) for k in ["train", "validation", "test"]]
    assert [len(x) for x in sets] == [70, 15, 15]
    assert sum(v["records"] for v in plan["sizes"].values()) == len(cohort)
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(set.union(*sets)) == 100
