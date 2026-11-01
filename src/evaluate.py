import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .dataset_loader import QUALITY, TARGETS


def regression_metrics(actual, predicted):
    rows = []
    for c, target in enumerate(["SBP", "DBP"]):
        y, p = np.asarray(actual)[:, c], np.asarray(predicted)[:, c]
        error = p - y
        sd = float(np.std(error, ddof=1)) if len(error) > 1 else None
        me = float(error.mean())
        rows.append({"target": target, "n_records": len(y), "MAE": float(np.abs(error).mean()),
                     "RMSE": float(np.sqrt(np.mean(error**2))), "R2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None,
                     "ME": me, "SD_error": sd, "LoA_lower": me - 1.96 * sd if sd is not None else None,
                     "LoA_upper": me + 1.96 * sd if sd is not None else None})
    return rows


def cluster_bootstrap(actual, predicted, subjects, repetitions, seed):
    """Resample participants with all their records, preserving repeated draws."""
    groups, inverse = np.unique(subjects, return_inverse=True)
    counts = np.bincount(inverse)
    error = predicted - actual
    abs_sums = np.stack([np.bincount(inverse, weights=np.abs(error[:, c])) for c in range(2)], axis=1)
    sq_sums = np.stack([np.bincount(inverse, weights=error[:, c]**2) for c in range(2)], axis=1)
    draws = np.random.default_rng(seed).integers(len(groups), size=(repetitions, len(groups)))
    denominator = counts[draws].sum(axis=1)[:, None]
    mae = abs_sums[draws].sum(axis=1) / denominator
    rmse = np.sqrt(sq_sums[draws].sum(axis=1) / denominator)
    rows = []
    for c, target in enumerate(["SBP", "DBP"]):
        for name, values in [("MAE", mae), ("RMSE", rmse)]:
            low, high = np.quantile(values[:, c], [0.025, 0.975])
            rows.append({"target": target, "metric": name, "lower_95": float(low), "upper_95": float(high),
                         "bootstrap_unit": "participant", "repetitions": repetitions})
    return rows


def quality_strata(cohort, predicted):
    actual = cohort[TARGETS].to_numpy()
    labels = {"ECG": cohort[QUALITY[0]].to_numpy(), "PPG": cohort[QUALITY[1]].to_numpy(),
              "worst_modality": cohort[QUALITY].max(axis=1).to_numpy()}
    rows = []
    for modality, values in labels.items():
        for label in range(4):
            mask = values == label
            if not mask.any():
                continue
            for metrics in regression_metrics(actual[mask], predicted[mask]):
                rows.append({"modality": modality, "manual_poor_windows": label,
                             "n_participants": int(cohort.loc[mask, "subject_id"].nunique()), **metrics})
    return rows


def prediction_frame(cohort, predicted, qualities, gates):
    frame = cohort[["record_id", "subject_id"] + TARGETS + QUALITY].copy().reset_index(drop=True)
    for c, prefix in enumerate(["sbp", "dbp"]):
        frame[prefix + "_predicted"] = predicted[:, c]
    for c, prefix in enumerate(["ecg", "ppg"]):
        frame[prefix + "_estimated_goodness"] = qualities[:, c]
        frame[prefix + "_fusion_weight"] = gates[:, c]
    return frame


def paired_comparisons(predictions, repetitions, seed):
    """Differences in per-record losses averaged over training seeds, not ensembled predictions."""
    losses = {}
    for name, runs in predictions.items():
        losses[name] = np.mean([np.abs(p[["sbp_predicted", "dbp_predicted"]].to_numpy() - p[TARGETS].to_numpy()) for p in runs], axis=0)
    if "proposed" not in losses:
        return pd.DataFrame()
    frame = predictions["proposed"][0]
    ids, inverse = np.unique(frame.subject_id, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(len(ids), size=(repetitions, len(ids)))
    rows = []
    for name, loss in losses.items():
        if name == "proposed":
            continue
        delta = losses["proposed"] - loss
        for c, target in enumerate(["SBP", "DBP"]):
            sums = np.bincount(inverse, weights=delta[:, c])
            values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({"comparison": "proposed minus " + name, "target": target,
                         "delta_MAE": float(delta[:, c].mean()), "lower_95": float(lower), "upper_95": float(upper),
                         "note": "paired participant bootstrap; seed-averaged losses; no multiplicity adjustment"})
    return pd.DataFrame(rows)
