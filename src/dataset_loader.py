import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .preprocessing import preprocess
from .signal_quality import SQI_NAMES
from .utils import save_json

LOGGER = logging.getLogger(__name__)
TARGETS = ["sbp_mmHg", "dbp_mmHg"]
QUALITY = ["ecg_quality_label", "ppg_quality_label"]


def read_waveform(root, relative_path):
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Waveform path escapes dataset root")
    record = wfdb.rdrecord(str(path), physical=False)
    if record.fs != 500 or record.sig_len != 12000 or record.sig_name != ["ECG", "PPG"]:
        raise ValueError("Unexpected WFDB rate, length, or channel order")
    raw = record.d_signal
    if raw.shape != (12000, 2) or not np.isfinite(raw).all() or (raw == -(2 ** 31)).any():
        raise ValueError("Invalid or missing WFDB digital samples")
    if not np.array_equal(raw.sum(axis=0, dtype=np.int64) % 65536, np.asarray(record.checksum) % 65536):
        raise ValueError("WFDB channel checksum mismatch")
    if np.all(raw.std(axis=0) == 0):
        raise ValueError("Both channels are flat")
    return raw


def dataset_fingerprint(root, config):
    root = Path(root)
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name in ["metadata/records.csv", "subject-info.csv", "RECORDS", "SHA256SUMS.txt"]:
        h.update((root / name).read_bytes())
    for path in sorted(root.rglob("*")):
        if path.suffix in {".hea", ".dat"}:
            stat = path.stat()
            h.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for name in ["dataset_loader.py", "preprocessing.py", "signal_quality.py"]:
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def prepare_dataset(root, cache, config):
    """Audit every metadata record. Extra files never acquire inferred labels."""
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(root, config)
    summary_path = cache / "dataset_summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["fingerprint"] == fingerprint:
        LOGGER.info("Using validated preprocessing cache")
        return load_cache(cache)
    records = pd.read_csv(root / "metadata/records.csv")
    subjects = pd.read_csv(root / "subject-info.csv")
    required = ["record_id", "subject_id", "wfdb_record_path"] + TARGETS + QUALITY
    if not set(required).issubset(records):
        raise ValueError("Required metadata fields are missing")
    if records["record_id"].duplicated().any() or records["wfdb_record_path"].duplicated().any() or subjects["subject_id"].duplicated().any():
        raise ValueError("Ambiguous metadata identifiers")
    if not set(records.subject_id).issubset(set(subjects.subject_id)):
        raise ValueError("Unlinked participant IDs")
    if set(records.wfdb_record_path) != set((root / "RECORDS").read_text().splitlines()):
        raise ValueError("RECORDS and metadata do not agree")
    expected = set(records.wfdb_record_path)
    extra = sorted(str(p.relative_to(root)) for p in root.rglob("*.hea") if str(p.relative_to(root).with_suffix("")) not in expected)
    save_json(cache / "extra_waveforms.json", extra)
    missing = records.isna().sum().astype(int).to_dict()
    exclusions, valid, signals, sqis, automatic, hashes = [], [], [], [], [], []
    for n, (_, row) in enumerate(records.iterrows()):
        try:
            if row[required].isna().any():
                raise ValueError("Missing required metadata")
            if not row[QUALITY].isin([0, 1, 2, 3]).all():
                raise ValueError("Invalid manual quality label")
            if not (np.isfinite(row[TARGETS].to_numpy(dtype=float)).all() and row.sbp_mmHg > row.dbp_mmHg > 0):
                raise ValueError("Invalid cuff reference BP")
            raw = read_waveform(root, row.wfdb_record_path)
            x, sqi, aq = preprocess(raw, config)
            valid.append(row)
            signals.append(x)
            sqis.append(sqi)
            automatic.append(aq)
            hashes.append(hashlib.sha256(raw.tobytes()).hexdigest())
        except (OSError, ValueError, IndexError) as exc:
            exclusions.append({"record_id": row.record_id, "subject_id": row.subject_id, "reason": str(exc)})
        if n % 500 == 0:
            LOGGER.info("Audited %d/%d records", n + 1, len(records))
    if not valid:
        raise ValueError("No valid paired waveforms")
    valid = pd.DataFrame(valid).reset_index(drop=True)
    valid["waveform_sha256"] = hashes
    duplicates = valid[valid.waveform_sha256.duplicated(keep=False)].copy()
    duplicates.to_csv(cache / "duplicate_waveforms.csv", index=False)
    # Identical signals with conflicting references have ambiguous pairing: exclude the entire cluster.
    keep = ~valid.waveform_sha256.duplicated(keep=False).to_numpy()
    for row in duplicates.itertuples():
        exclusions.append({"record_id": row.record_id, "subject_id": row.subject_id, "reason": "ambiguous_duplicate_waveform"})
    valid = valid.loc[keep].reset_index(drop=True)
    x, sqi, aq = np.asarray(signals)[keep], np.asarray(sqis)[keep], np.asarray(automatic)[keep]
    valid.to_csv(cache / "cohort.csv", index=False)
    pd.DataFrame(exclusions, columns=["record_id", "subject_id", "reason"]).to_csv(cache / "exclusions.csv", index=False)
    outliers = records[["record_id"] + TARGETS].copy()
    for target in TARGETS:
        q1, q3 = records[target].quantile([0.25, 0.75])
        outliers[target + "_iqr_flag"] = (records[target] < q1 - 1.5 * (q3 - q1)) | (records[target] > q3 + 1.5 * (q3 - q1))
    outliers.to_csv(cache / "target_outlier_flags_retained.csv", index=False)
    quality_rows = []
    for c, modality in enumerate(["ECG", "PPG"]):
        for j, feature in enumerate(SQI_NAMES):
            for label in range(4):
                values = sqi[valid[QUALITY[c]].to_numpy() == label, c, j]
                quality_rows.append({"modality": modality, "manual_poor_windows": label, "indicator": feature,
                                     "n": len(values), "mean": float(values.mean()) if len(values) else None})
    pd.DataFrame(quality_rows).to_csv(cache / "sqi_by_manual_quality.csv", index=False)
    summary = {"fingerprint": fingerprint, "metadata_records": len(records), "metadata_subjects": int(records.subject_id.nunique()),
               "documented_records": 3849, "documented_subjects": 977,
               "local_dat_files": len(list(root.rglob("*.dat"))), "local_hea_files": len(list(root.rglob("*.hea"))),
               "unreferenced_waveform_pairs": len(extra), "excluded_records": len(exclusions),
               "usable_records": len(valid), "usable_subjects": int(valid.subject_id.nunique()),
               "exclusion_reasons": pd.Series([r["reason"] for r in exclusions]).str.replace(str(root), "DATASET", regex=False).value_counts().to_dict(),
               "ambiguous_duplicate_records": len(duplicates), "metadata_missing": missing,
               "participant_metadata_missing": subjects.isna().sum().astype(int).to_dict(),
               "target_description": records[TARGETS].describe().to_dict(), "preprocessing": config,
               "quality_counts_metadata": {q: {str(k): int(v) for k, v in records[q].value_counts().sort_index().items()} for q in QUALITY},
               "notes": ["99 expected records are absent in this local package; no downloads attempted.",
                         "Extra waveforms are ignored, without inferred labels.", "BP IQR flags are descriptive, not exclusion criteria.",
                         "Manual quality windows partly precede the released 24-second crop."]}
    with (cache / "processed.tmp.npz").open("wb") as f:
        np.savez_compressed(f, x=x, sqi=sqi, automatic=aq)
    (cache / "processed.tmp.npz").replace(cache / "processed.npz")
    save_json(summary_path, summary)
    LOGGER.info("Usable cohort: %d records, %d participants", len(valid), valid.subject_id.nunique())
    return load_cache(cache)


def load_cache(cache):
    cache = Path(cache)
    with np.load(cache / "processed.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return pd.read_csv(cache / "cohort.csv"), arrays, json.loads((cache / "dataset_summary.json").read_text())


def participant_split(cohort, seed, quick_subjects=None):
    subjects = np.asarray(sorted(cohort.subject_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    groups = {"train": subjects[:int(0.7*n)], "validation": subjects[int(0.7*n):int(0.85*n)], "test": subjects[int(0.85*n):]}
    if quick_subjects:
        # Smoke tests use only the full protocol's training participants.
        pool = groups["train"]
        if sum(quick_subjects) > len(pool):
            raise ValueError("Quick-test participant budget exceeds the full training pool")
        a, b, c = quick_subjects
        groups = {"train": pool[:a], "validation": pool[a:a+b], "test": pool[a+b:a+b+c]}
    sets = [set(v) for v in groups.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("Participant leakage")
    indices = {key: np.flatnonzero(cohort.subject_id.isin(ids)) for key, ids in groups.items()}
    if any(len(v) == 0 for v in indices.values()):
        raise ValueError("Empty participant split")
    return indices, {key: value.tolist() for key, value in groups.items()}
