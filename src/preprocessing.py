import numpy as np
from scipy import signal

from .signal_quality import signal_quality

DEFAULT_PREPROCESSING = {"input_fs": 500, "samples": 12000, "output_fs": 125,
                         "filter_order": 4, "bands": [[0.5, 40.0], [0.5, 8.0]],
                         "representation": "digital", "normalization": "per-record-channel-zscore"}


def preprocess(raw, config):
    raw = np.asarray(raw)
    if raw.shape != (config["samples"], 2) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid waveform shape or nonfinite samples: {raw.shape}")
    fs, out_fs = config["input_fs"], config["output_fs"]
    if fs % out_fs:
        raise ValueError("output_fs must divide input_fs")
    sqi, automatic = signal_quality(raw, fs)
    channels = []
    for c, band in enumerate(config["bands"]):
        x = raw[:, c].astype(np.float64)
        x -= np.median(x)
        sos = signal.butter(config["filter_order"], band, btype="bandpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
        x = signal.resample_poly(x, 1, fs // out_fs)
        x = (x - x.mean()) / max(float(x.std()), 1e-8)
        channels.append(x)
    waveform = np.asarray(channels, np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("Nonfinite preprocessed signal")
    return waveform, sqi, automatic


def fit_scalers(sqi, y, train_indices):
    return {"sqi_mean": sqi[train_indices].mean(0).tolist(),
            "sqi_std": np.maximum(sqi[train_indices].std(0), 1e-6).tolist(),
            "y_mean": y[train_indices].mean(0).tolist(),
            "y_std": np.maximum(y[train_indices].std(0), 1e-6).tolist()}


def scale_sqi(sqi, scalers):
    if scalers.get("sqi_mode") == "bounded_v2":
        # SQI fractions already have physical bounds; avoid extrapolation from near-zero training SD.
        return np.clip(np.asarray(sqi) / np.asarray([1, 1, 1, 1, np.log1p(10000), 1, 1]), 0, 1).astype(np.float32)
    return ((sqi - np.asarray(scalers["sqi_mean"])) / np.asarray(scalers["sqi_std"])).astype(np.float32)
