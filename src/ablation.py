import hashlib
import logging

import numpy as np

from .dataset_loader import read_waveform
from .preprocessing import preprocess

CONDITIONS = {"ecg_noise_10dB": (0, 10), "ppg_noise_10dB": (1, 10),
              "both_noise_5dB": (None, 5), "ecg_missing": (0, None), "ppg_missing": (1, None)}


def degrade(raw, condition, seed):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    channel, snr = CONDITIONS[condition]
    altered = raw.astype(np.float64).copy()
    rng = np.random.default_rng(seed)
    for c in ([0, 1] if channel is None else [channel]):
        if snr is None:
            altered[:, c] = 0
        else:
            noise = rng.normal(size=len(raw))
            noise *= np.std(altered[:, c]) / max(noise.std(), 1e-8) * 10 ** (-snr / 20)
            altered[:, c] += noise
    return altered


def robustness_inputs(root, test_cohort, preprocessing, seed):
    variants = {key: {k: [] for k in ["x", "sqi", "automatic"]} for key in CONDITIONS}
    for i, row in enumerate(test_cohort.itertuples()):
        raw = read_waveform(root, row.wfdb_record_path)
        record_seed = seed + int(hashlib.sha256(row.record_id.encode()).hexdigest()[:8], 16)
        for condition, arrays in variants.items():
            values = preprocess(degrade(raw, condition, record_seed), preprocessing)
            for key, value in zip(arrays, values, strict=True):
                arrays[key].append(value)
        if i % 200 == 0:
            logging.info("Prepared evaluation-only degradations: %d/%d", i + 1, len(test_cohort))
    return {condition: {key: np.asarray(value) for key, value in arrays.items()} for condition, arrays in variants.items()}
