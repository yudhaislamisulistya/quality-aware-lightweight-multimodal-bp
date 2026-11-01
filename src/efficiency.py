"""Run in a fresh process so peak RSS is not inherited from other experiments."""
import argparse
import io
import json
import platform
import resource
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from .dataset_loader import read_waveform
from .preprocessing import preprocess, scale_sqi
from .train import load_model
from .utils import environment, save_json


def count_macs(model, inputs):
    macs, handles = [], []

    def hook(module, args, output):
        if isinstance(module, nn.Conv1d):
            macs.append(output.numel() * module.kernel_size[0] * module.in_channels // module.groups)
        elif isinstance(module, nn.Linear):
            macs.append(output.numel() * module.in_features)
        elif isinstance(module, nn.GRU):
            batch, steps, features = args[0].shape
            directions = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                width = features if layer == 0 else module.hidden_size * directions
                macs.append(batch * steps * directions * 3 * module.hidden_size * (width + module.hidden_size))

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.GRU)):
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return int(sum(macs))


def benchmark(checkpoint_path, root, record_path, output, threads, warmup, repetitions, device):
    torch.set_num_threads(threads)
    model, checkpoint = load_model(checkpoint_path, device)
    raw = read_waveform(root, record_path)
    scalers = checkpoint["scalers"]

    def make_inputs():
        x, sqi, aq = preprocess(read_waveform(root, record_path), checkpoint["preprocessing"])
        sqi = scale_sqi(sqi[None], scalers)[0]
        return [torch.tensor(v[None], device=device) for v in (x, sqi, aq)]

    inputs = make_inputs()
    macs = count_macs(model, inputs)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    baseline_rss = psutil.Process().memory_info().rss / 2**20
    sampled_rss = [baseline_rss]
    memory_stop = threading.Event()

    def sample_memory():
        process = psutil.Process()
        while not memory_stop.wait(.001):
            sampled_rss[0] = max(sampled_rss[0], process.memory_info().rss / 2**20)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    timings = {}
    with torch.inference_mode():
        for batch_size in [1, 32]:
            batch = [v.repeat(batch_size, *([1] * (v.ndim - 1))) for v in inputs]
            for _ in range(warmup):
                model(*batch)
            synchronize()
            values = []
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                model(*batch)
                synchronize()
                values.append((time.perf_counter_ns() - start) / 1e6)
            timings[f"batch_{batch_size}"] = {"median_latency_ms": float(np.median(values)),
                                              "p95_latency_ms": float(np.quantile(values, .95)),
                                              "records_per_second": float(batch_size * 1000 / np.median(values))}
        e2e = []
        for _ in range(max(3, repetitions // 10)):
            synchronize()
            start = time.perf_counter_ns()
            model(*make_inputs())
            synchronize()
            e2e.append((time.perf_counter_ns() - start) / 1e6)
    memory_stop.set()
    monitor.join()
    sampled_rss[0] = max(sampled_rss[0], psutil.Process().memory_info().rss / 2**20)
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu_model)
    result = {"parameters": sum(p.numel() for p in model.parameters()),
              "weights_and_buffers_bytes": sum(v.numel() * v.element_size() for v in model.state_dict().values()),
              "serialized_state_dict_bytes": buffer.getbuffer().nbytes,
              "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
              "MACs_per_record": macs, "approx_FLOPs_per_record": 2 * macs,
              "mac_scope": "Conv1d, Linear, GRU matrix MACs; excludes bias, activations, normalization, pooling, preprocessing",
              "device": device, "device_name": torch.cuda.get_device_name() if device.startswith("cuda") else cpu_model,
              "threads": threads, "warmup_iterations": warmup, "measurement_iterations": repetitions,
              "input_shape_batch1": list(inputs[0].shape), "precision": "float32", "model_only": timings,
              "end_to_end_median_ms": float(np.median(e2e)), "end_to_end_repetitions": len(e2e),
              "end_to_end_scope": "warm-filesystem-cache WFDB read + SQI + filtering + resampling + normalization + neural inference",
              "isolated_process_peak_RSS_MiB": sampled_rss[0],
              "OS_high_water_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "RSS_after_model_load_MiB": baseline_rss,
              "memory_scope": "psutil RSS sampled every 1 ms during benchmark, includes loaded runtime/model and batches 1/32; approximate peak, not training/incremental memory. OS high-water is separate because fork/exec can inherit parent HWM",
              "peak_CUDA_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
              "environment": environment()}
    save_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("root")
    parser.add_argument("record")
    parser.add_argument("output")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    benchmark(args.checkpoint, args.root, args.record, args.output, args.threads, args.warmup, args.repetitions, args.device)
