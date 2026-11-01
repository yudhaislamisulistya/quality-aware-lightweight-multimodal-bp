import numpy as np
from scipy import signal, stats

SQI_NAMES = ["band_fraction", "drift_fraction", "high_frequency_fraction",
             "flat_fraction", "log_kurtosis", "peak_regularity", "plausible_peak_rate"]


def signal_quality(raw, fs=500):
    """Waveform-only heuristic indicators; none is a calibrated clinical SQI."""
    features, automatic = [], []
    for channel, (low, high) in enumerate([(0.5, 40), (0.5, 8)]):
        x = raw[:, channel].astype(np.float64)
        x = x - np.median(x)
        if np.std(x) < 1e-8:
            features.append([0, 0, 0, 1, 0, 0, 0])
            automatic.append(0.0)
            continue
        x /= np.std(x)
        f, psd = signal.welch(x, fs, nperseg=min(len(x), 2048))
        total = psd[(f >= 0.05) & (f <= 100)].sum() + 1e-12
        band = psd[(f >= low) & (f <= high)].sum() / total
        drift = psd[(f >= 0.05) & (f < 0.5)].sum() / total
        noise = psd[(f > (40 if channel == 0 else 15)) & (f <= 100)].sum() / total
        flat = np.mean(np.diff(x) == 0)
        filtered = signal.sosfiltfilt(signal.butter(3, [5, 25] if channel == 0 else [0.5, 8], fs=fs, btype="bandpass", output="sos"), x)
        peaks, _ = signal.find_peaks(np.abs(filtered) if channel == 0 else filtered,
                                     distance=int(0.3 * fs), prominence=0.4 * np.std(filtered))
        intervals = np.diff(peaks) / fs
        plausible = float(len(intervals) >= 4 and np.mean((intervals >= 0.3) & (intervals <= 2.0)) >= 0.8)
        regularity = float(np.exp(-np.std(intervals) / (np.mean(intervals) + 1e-8))) if plausible else 0.0
        kurtosis = np.log1p(np.clip(stats.kurtosis(x, fisher=False), 0, 10000))
        features.append([band, drift, noise, flat, kurtosis, regularity, plausible])
        automatic.append(float(np.clip(band * (1 - drift) * (1 - flat) * (0.5 + 0.5 * regularity), 0, 1)))
    return np.asarray(features, np.float32), np.asarray(automatic, np.float32)
