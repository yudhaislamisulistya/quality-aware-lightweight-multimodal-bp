import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset_loader import QUALITY, TARGETS
from .models import BPModel
from .preprocessing import fit_scalers, scale_sqi
from .utils import seed_everything

LOGGER = logging.getLogger(__name__)


def make_loader(cohort, arrays, indices, scalers, batch_size, shuffle, seed):
    y = cohort[TARGETS].to_numpy(np.float32)
    q = 1 - cohort[QUALITY].to_numpy(np.float32) / 3
    tensors = [torch.from_numpy(values[indices].astype(np.float32)) for values in
               [arrays["x"], scale_sqi(arrays["sqi"], scalers), arrays["automatic"],
                (y - np.asarray(scalers["y_mean"])) / np.asarray(scalers["y_std"]), q]]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_model(name, seed, cohort, arrays, splits, config, directory, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, config["threads"])
    device = torch.device(config["device"])
    scalers = fit_scalers(arrays["sqi"], cohort[TARGETS].to_numpy(np.float32), splits["train"])
    loaders = {key: make_loader(cohort, arrays, indices, scalers, config["batch_size"], key == "train", seed) for key, indices in splits.items() if key != "test"}
    model = BPModel(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    y_std = torch.tensor(scalers["y_std"], device=device, dtype=torch.float32)
    best, bad_epochs, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        for phase in ["train", "validation"]:
            model.train(phase == "train")
            loss_sum, reg_sum, aux_sum, mae_sum, n = 0., 0., 0., 0., 0
            with torch.set_grad_enabled(phase == "train"):
                for batch in loaders[phase]:
                    x, sqi, automatic, target, manual_goodness = [v.to(device) for v in batch]
                    predicted, logits, _ = model(x, sqi, automatic)
                    reg = mse(predicted, target)
                    aux = bce(logits, manual_goodness) if model.supervised_quality else reg.new_zeros(())
                    loss = reg + config["quality_loss_weight"] * aux
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss: {name}, epoch {epoch}")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                    size = len(x)
                    n += size
                    loss_sum += loss.item() * size
                    reg_sum += reg.item() * size
                    aux_sum += aux.item() * size
                    mae_sum += ((predicted - target).abs() * y_std).mean().item() * size
            row.update({phase + "_loss": loss_sum / n, phase + "_regression_mse": reg_sum / n,
                        phase + "_quality_bce": aux_sum / n, phase + "_mae_mmHg": mae_sum / n})
        validation = row["validation_mae_mmHg"]
        scheduler.step(validation)
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if validation < best - config["min_delta_mmHg"]:
            best, bad_epochs = validation, 0
            checkpoint = {"state_dict": model.state_dict(), "model_name": name, "seed": seed,
                          "epoch": epoch, "validation_mae_mmHg": best, "scalers": scalers,
                          "preprocessing": config["preprocessing"], "config": config, "provenance": provenance}
            torch.save(checkpoint, directory / "best.tmp.pt")
            (directory / "best.tmp.pt").replace(directory / "best.pt")
        else:
            bad_epochs += 1
        LOGGER.info("%s seed=%s epoch=%d train_MAE=%.3f val_MAE=%.3f elapsed=%.1fs", name, seed, epoch, row["train_mae_mmHg"], validation, time.perf_counter() - started)
        if bad_epochs >= config["patience"]:
            break
    return directory / "best.pt"


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture_version") == "risk_v1":
        from .risk_model import RiskFusion
        model = RiskFusion(checkpoint["expert_scalers"], checkpoint["model_name"], checkpoint["config"]["risk_floor_mmHg"]).to(device)
    elif checkpoint.get("architecture_version") == "factorial_v1":
        from .factorial_model import FactorialBPModel
        model = FactorialBPModel(checkpoint["model_name"]).to(device)
    elif checkpoint.get("architecture_version") == "quality_v2":
        from .optimized_model import RobustBPModel
        model = RobustBPModel(checkpoint["model_name"]).to(device)
    else:
        model = BPModel(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict(model, arrays, scalers, device="cpu", batch_size=64):
    predictions, qualities, gates = [], [], []
    sqi = scale_sqi(arrays["sqi"], scalers)
    with torch.inference_mode():
        for start in range(0, len(arrays["x"]), batch_size):
            values = [torch.as_tensor(v[start:start+batch_size], device=device) for v in [arrays["x"], sqi, arrays["automatic"]]]
            prediction, logits, weights = model(*values)
            predictions.append(prediction.cpu().numpy())
            qualities.append(logits.sigmoid().cpu().numpy())
            gates.append(weights.cpu().numpy())
    y = np.concatenate(predictions) * np.asarray(scalers["y_std"]) + np.asarray(scalers["y_mean"])
    return y, np.concatenate(qualities), np.concatenate(gates)
