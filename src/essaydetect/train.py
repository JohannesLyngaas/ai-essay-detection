"""Run one experiment from a config file.

    uv run python -m essaydetect.train --config configs/lr_random.yaml

Steps:
    1. Read the YAML config.
    2. Load the split it names (random or topic).
    3. Train the model it names (classical or transformer).
    4. Score the test predictions and save everything to results/<name>/.
"""

from __future__ import annotations

import argparse
import time

import yaml

from essaydetect.data import LABEL_COL, load_split
from essaydetect.evaluate import compute_metrics, save_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment from a config file.")
    parser.add_argument("--config", required=True, help="path to a YAML file in configs/")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"Experiment: {config['name']}  ({config['type']}, {config['split']} split)")

    splits = load_split(config["split"], args.data_dir)

    # For quick test runs: train on a small sample. Val and test stay complete,
    # so the scores are still comparable (just worse).
    if config.get("subsample"):
        n = min(config["subsample"], len(splits["train"]))
        splits["train"] = splits["train"].sample(n=n, random_state=config["seed"])
        print(f"Subsample: training on {n:,} essays")

    start = time.time()
    if config["type"] == "classical":
        from essaydetect.classical import train_classical

        y_pred, y_score, details = train_classical(config, splits)
    elif config["type"] == "transformer":
        # Imported here, so the classical runs work without loading PyTorch.
        from essaydetect.transformer import train_transformer

        y_pred, y_score, details = train_transformer(config, splits)
    else:
        raise ValueError(f"Unknown type {config['type']!r}, expected classical or transformer.")
    train_seconds = time.time() - start

    test = splits["test"]
    metrics = compute_metrics(test[LABEL_COL], y_pred, y_score)
    out = save_run(
        config, test, y_pred, y_score, metrics, details,
        n_train=len(splits["train"]), train_seconds=train_seconds,
        results_dir=args.results_dir,
    )  # fmt: skip

    print(f"\nTest results ({len(test):,} essays, trained in {train_seconds:.0f} s)")
    for key in ("accuracy", "macro_f1", "ai_recall", "false_positive_rate", "roc_auc"):
        print(f"  {key:<20} {metrics[key]:.4f}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
