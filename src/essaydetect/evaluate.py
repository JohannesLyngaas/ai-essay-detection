"""Score every model in exactly the same way, and save the results.

Two jobs:

1. compute_metrics() and save_run() are called by train.py after every experiment.
   Keeping them in one shared file is what makes the comparison fair.

2. Run this file on its own to collect all finished experiments into one table:

       uv run python -m essaydetect.evaluate

   It prints a Markdown table (ready for the README) and writes results/summary.md.

A note on the metrics. The two test sets have very different class balance:
39 % AI in the random split, 18 % AI in the topic split. Accuracy and F1 depend on
that balance (on the topic test set, a model that always says "student" already gets
82 % accuracy). AI recall, false positive rate and ROC-AUC do not, so those are the
fairest way to compare the two splits.

ai_recall_at_1pct_fpr is there for the same reason: it answers "if we only accept that 1 %
of real student essays get flagged, how many AI essays do we still catch?", which is both
independent of class balance and independent of the threshold each model happens to use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from essaydetect.data import LABEL_COL, SOURCE_COL, TOPIC_COL


def ai_recall_at_fpr(y_true, y_score, max_fpr: float = 0.01) -> float:
    """The highest AI recall reachable while flagging at most max_fpr of student essays.

    y_pred uses one fixed threshold (score > 0). This metric asks a different question:
    if we were free to move the threshold, how many AI essays could we catch while
    keeping the false positive rate inside a budget we can defend to a teacher?

    roc_curve walks through every threshold the scores allow and reports, for each one,
    the false positive rate (share of student essays flagged) and the true positive rate
    (which is exactly AI recall). We keep the thresholds that stay inside the budget and
    take the best recall among them.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # roc_curve always includes the threshold that flags nothing (fpr = 0, tpr = 0),
    # so at least one point is always inside the budget and .max() is always defined.
    return float(tpr[fpr <= max_fpr].max())


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """All test metrics for one model.

    y_true:  true labels (1 = AI, 0 = student)
    y_pred:  predicted labels
    y_score: optional score per essay, higher = more likely AI. Needed for ROC-AUC.
    """
    # The confusion matrix counts the four possible outcomes.
    #   tn: student essay, predicted student   (correct)
    #   fp: student essay, predicted AI        (the worst error for a teacher)
    #   fn: AI essay, predicted student        (AI essay slipped through)
    #   tp: AI essay, predicted AI             (correct)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_ai": f1_score(y_true, y_pred, pos_label=1),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),  # mean of F1 for both classes
        "ai_recall": tp / (tp + fn),  # share of AI essays that were caught
        "false_positive_rate": fp / (fp + tn),  # share of student essays flagged as AI
    }
    if y_score is not None:
        # ROC-AUC: the chance that a random AI essay gets a higher score than a random
        # student essay. 0.5 = guessing, 1.0 = perfect ranking. Independent of threshold.
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        # Best AI recall we could get at a false positive rate of at most 1 %.
        metrics["ai_recall_at_1pct_fpr"] = ai_recall_at_fpr(y_true, y_score, max_fpr=0.01)

    metrics = {key: round(float(value), 4) for key, value in metrics.items()}
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def save_run(
    config: dict,
    test_df: pd.DataFrame,
    y_pred,
    y_score,
    metrics: dict,
    details: dict,
    n_train: int,
    train_seconds: float,
    results_dir: str | Path = "results",
) -> Path:
    """Write results/<name>/metrics.json and results/<name>/predictions.csv."""
    out = Path(results_dir) / config["name"]
    out.mkdir(parents=True, exist_ok=True)

    record = {
        "name": config["name"],
        "split": config["split"],
        "n_train": n_train,
        "n_test": len(test_df),
        "train_seconds": round(train_seconds, 1),
        "test": metrics,
        "details": details,  # model-specific extras, e.g. the chosen C or top n-grams
        "config": config,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2))

    # One row per test essay. The essay text is left out on purpose: it keeps the
    # repo small and avoids redistributing the dataset. Join on `id` to get the text.
    columns = ["id", TOPIC_COL, LABEL_COL] + ([SOURCE_COL] if SOURCE_COL in test_df else [])
    predictions = test_df[columns].copy()
    predictions["pred"] = y_pred
    predictions["score"] = y_score
    predictions.to_csv(out / "predictions.csv", index=False)
    return out


def collect_results(results_dir: str | Path = "results") -> pd.DataFrame:
    """One row per finished experiment, read from results/*/metrics.json."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        record = json.loads(path.read_text())
        test = record["test"]
        rows.append(
            {
                "experiment": record["name"],
                "split": record["split"],
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "ai_recall": test["ai_recall"],
                "false_positive_rate": test["false_positive_rate"],
                # .get: these two are missing if the run had no scores.
                "ai_recall_at_1pct_fpr": test.get("ai_recall_at_1pct_fpr"),
                "roc_auc": test.get("roc_auc"),
                "train_seconds": round(record["train_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    """Format a DataFrame as a Markdown table, numbers rounded to 3 decimals."""

    def fmt(value) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    table = collect_results()
    if table.empty:
        print("No results yet. Run an experiment with essaydetect.train first.")
        return
    markdown = to_markdown(table)
    print(markdown)
    Path("results/summary.md").write_text(markdown + "\n")
    print("\nWrote results/summary.md")


if __name__ == "__main__":
    main()
