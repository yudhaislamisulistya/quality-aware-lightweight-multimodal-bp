"""Count stored parameter slots, including frozen heads; do not count buffers as parameters."""
import argparse
import csv
import json
from pathlib import Path

import torch

from .factorial_model import FactorialBPModel
from .risk_model import RiskFusion


def parameter_breakdown(model):
    groups = {"encoders": 0, "quality_heads": 0, "bp_heads": 0, "risk_head": 0}
    for name, value in model.named_parameters():
        if name.startswith("risk_head."):
            key = "risk_head"
        elif ".encoders." in name or name.startswith("encoders."):
            key = "encoders"
        elif ".quality." in name or name.startswith("quality."):
            key = "quality_heads"
        elif ".regressor." in name or name.startswith("regressor."):
            key = "bp_heads"
        else:
            raise ValueError("Unaccounted parameter: "+name)
        groups[key] += value.numel()
    total = sum(value.numel() for value in model.parameters())
    assert sum(groups.values()) == total
    return {**groups, "total_stored_parameters": total,
            "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "buffer_elements_excluded": sum(p.numel() for p in model.buffers())}


def audit_models():
    # These scaler placeholders instantiate shapes only; they are not BP labels or fitted data.
    scalers = [{"sqi_mode": "bounded_v2", "y_mean": [0., 0.], "y_std": [1., 1.]} for _ in range(2)]
    models = {"unimodal": FactorialBPModel("ecg_only_clean"),
              "feature_A_to_D": FactorialBPModel("A_uniform_clean"),
              "dense_feature_control": FactorialBPModel("dense_uniform_clean"),
              "late_fusion_all_policies": RiskFusion(scalers)}
    return [{"model": name, **parameter_breakdown(model)} for name, model in models.items()]


def verify_checkpoints(results_root):
    from .train import load_model
    results_root = Path(results_root)
    paths = sorted((results_root/"factorial/fits").glob("fold_*/*/seed_*/best.pt"))
    paths += sorted((results_root/"reliability/fits").glob("fold_*/seed_*/*.pt"))
    if not paths:
        raise ValueError("No completed factorial/reliability checkpoints found")
    totals = {}
    for path in paths:
        model, checkpoint = load_model(path)
        row = parameter_breakdown(model)
        totals.setdefault(checkpoint["model_name"], set()).add(row["total_stored_parameters"])
    assert all(len(values) == 1 for values in totals.values())
    return {"checkpoints_verified": len(paths), "stored_counts_by_model": {k: list(v)[0] for k, v in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/parameter_counts.json")
    parser.add_argument("--checkpoints-root", help="Optional trusted local results folder; checkpoints are never distributed")
    args = parser.parse_args()
    rows = audit_models()
    result = {"definition": "sum(p.numel()) over all Parameter objects, regardless of requires_grad or policy use",
              "excluded": "buffers, preprocessing, hyperparameters, parameter-free operations and cross-fit teacher replicas",
              "models": rows}
    if args.checkpoints_root:
        result["checkpoint_verification"] = verify_checkpoints(args.checkpoints_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
