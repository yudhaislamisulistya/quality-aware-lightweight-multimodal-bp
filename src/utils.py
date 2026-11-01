import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    packages = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "wfdb", "matplotlib", "psutil"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "cuda_available": torch.cuda.is_available(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def code_hash():
    h = hashlib.sha256()
    for p in sorted(Path("src").glob("*.py")) + [Path("main.py")]:
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()
