"""Regenerate the historical participant roles locally, without training or publishing IDs."""
import argparse
import json
from pathlib import Path

from src.dataset_loader import participant_split, prepare_dataset
from src.preprocessing import DEFAULT_PREPROCESSING
from src.utils import save_json


def roles_from_cohort(cohort, seed):
    indices, ids = participant_split(cohort, seed)
    return {"participant_ids": ids,
            "sizes": {key: {"participants": len(ids[key]), "records": len(value)} for key, value in indices.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Authorized local CardioWave-Portable 1.0.0 directory")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--output", default="results/full/participant_splits.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    cohort, _, _ = prepare_dataset(Path(args.data).resolve(), args.cache, DEFAULT_PREPROCESSING)
    plan = roles_from_cohort(cohort, config["split_seed"])
    output = Path(args.output)
    if output.exists():
        if json.loads(output.read_text())["participant_ids"] != plan["participant_ids"]:
            raise ValueError("Existing historical roles differ; refusing to overwrite them")
    else:
        save_json(output, plan)
    print(json.dumps({"output": str(output), "sizes": plan["sizes"], "training_performed": False}))


if __name__ == "__main__":
    main()
