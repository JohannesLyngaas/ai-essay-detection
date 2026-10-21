"""Load the DAIGT v2 dataset, clean it, and create two train/val/test splits.

Run once from the repository root:

    uv run python -m essaydetect.data --raw data/raw/train_v2_drcat_02.csv

What it writes:

    data/splits/random/{train,val,test}.csv   stratified random 80/10/10 split
    data/splits/topic/{train,val,test}.csv    whole essay topics held out for val and test
    results/splits/split_summary.json         sizes, class balance and topics per split
                                              (small, so it is committed for the report)

Why two splits? With a random split, essays on the same topic end up in both
train and test, so a model can score well by learning topic words instead of
writing style. The topic split puts whole topics in val and test, which tests
whether a model works on essay prompts it has never seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Column names in the DAIGT v2 CSV. If the file uses different names,
# change them here and nowhere else.
TEXT_COL = "text"
LABEL_COL = "label"  # 1 = AI-generated, 0 = written by a student
TOPIC_COL = "prompt_name"  # which essay prompt the essay answers
SOURCE_COL = "source"  # where the essay came from (which LLM, or the student corpus)

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------
# 1. Load and clean
# --------------------------------------------------------------------------


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and return a clean DataFrame with an `id` column.

    Cleaning steps:
    1. keep only the columns we use
    2. drop rows with a missing text, label or topic
    3. strip whitespace and drop empty essays
    4. drop texts that appear with BOTH labels (we cannot know which is right)
    5. drop remaining exact duplicates, keeping the first copy
    6. give every essay a stable `id`, so predictions can be traced back later
    """
    df = pd.read_csv(path)

    missing = [c for c in (TEXT_COL, LABEL_COL, TOPIC_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {path}. "
            f"Columns in the file: {list(df.columns)}. "
            "Update the *_COL constants at the top of data.py."
        )

    n_raw = len(df)
    keep = [TEXT_COL, LABEL_COL, TOPIC_COL]
    if SOURCE_COL in df.columns:
        keep.append(SOURCE_COL)
    df = df[keep].dropna(subset=[TEXT_COL, LABEL_COL, TOPIC_COL]).copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    if not set(df[LABEL_COL].unique()) <= {0, 1}:
        raise ValueError(f"Expected labels 0 and 1, found {sorted(df[LABEL_COL].unique())}")

    # The same text with two different labels: we cannot trust either, so drop every copy.
    labels_per_text = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    df = df[~df[TEXT_COL].isin(conflicting)]

    # Exact duplicates with the same label: keep one copy. Otherwise the same essay
    # could land in both train and test, which would inflate the test score.
    df = df.drop_duplicates(subset=TEXT_COL, keep="first")

    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    print(
        f"Loaded {n_raw:,} rows, kept {len(df):,} after cleaning "
        f"({len(conflicting)} texts had conflicting labels)."
    )
    return df


# --------------------------------------------------------------------------
# 2. Split
# --------------------------------------------------------------------------


def make_random_split(
    df: pd.DataFrame, seed: int = 42, val_size: float = 0.1, test_size: float = 0.1
) -> dict[str, pd.DataFrame]:
    """Stratified random split: every part keeps the same share of AI essays."""
    rest, test = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COL], random_state=seed
    )
    # val_size is a share of the WHOLE dataset, so rescale it to a share of `rest`.
    # Example: 0.1 / (1 - 0.1) = 0.111, and 11.1 % of 90 % is 10 % of the whole.
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=rest[LABEL_COL],
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def make_topic_split(
    df: pd.DataFrame, seed: int = 42, n_val_topics: int = 2, n_test_topics: int = 3
) -> dict[str, pd.DataFrame]:
    """Hold out whole topics: val and test only contain prompts never seen in training.

    The topics are shuffled with a fixed seed, so the split is identical on every
    machine (your laptop and Kaggle get the same topics in test).
    """
    topics = sorted(df[TOPIC_COL].unique())  # sort first, so the shuffle is reproducible
    if len(topics) < n_val_topics + n_test_topics + 1:
        raise ValueError(
            f"Only {len(topics)} topics, need at least {n_val_topics + n_test_topics + 1}."
        )

    rng = np.random.default_rng(seed)
    shuffled = [str(t) for t in rng.permutation(topics)]
    test_topics = shuffled[:n_test_topics]
    val_topics = shuffled[n_test_topics : n_test_topics + n_val_topics]

    splits = {
        "train": df[~df[TOPIC_COL].isin(test_topics + val_topics)],
        "val": df[df[TOPIC_COL].isin(val_topics)],
        "test": df[df[TOPIC_COL].isin(test_topics)],
    }

    # A test set with only one class would make F1 meaningless.
    for name, part in splits.items():
        if part[LABEL_COL].nunique() < 2:
            raise ValueError(f"The {name} split has only one class. Try another seed.")
    return splits


# --------------------------------------------------------------------------
# 3. Summarise, save, load
# --------------------------------------------------------------------------


def summarize(splits: dict[str, pd.DataFrame]) -> dict:
    """Size, class balance and topics for each part of a split."""
    summary = {}
    for name, part in splits.items():
        n_ai = int(part[LABEL_COL].sum())
        summary[name] = {
            "n_essays": len(part),
            "n_ai": n_ai,
            "n_human": len(part) - n_ai,
            "ai_share": round(n_ai / len(part), 3),
            "topics": sorted(str(t) for t in part[TOPIC_COL].unique()),
        }
    return summary


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)


def load_split(split: str, data_dir: str | Path = "data/splits") -> dict[str, pd.DataFrame]:
    """Read back train/val/test for one split ('random' or 'topic').

    This is what the training code and the notebooks use.
    """
    folder = Path(data_dir) / split
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} does not exist. Run `python -m essaydetect.data` first."
        )
    return {name: pd.read_csv(folder / f"{name}.csv") for name in SPLIT_NAMES}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw", default="data/raw/train_v2_drcat_02.csv", help="path to the DAIGT v2 CSV"
    )
    parser.add_argument("--out", default="data/splits", help="where to write the split CSVs")
    parser.add_argument(
        "--summary",
        default="results/splits/split_summary.json",
        help="where to write the split summary",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_clean(args.raw)

    all_splits = {
        "random": make_random_split(df, seed=args.seed),
        "topic": make_topic_split(df, seed=args.seed),
    }

    summary: dict = {"seed": args.seed, "n_essays_after_cleaning": len(df)}
    for scheme, splits in all_splits.items():
        save_splits(splits, Path(args.out) / scheme)
        summary[scheme] = summarize(splits)

        print(f"\n{scheme} split")
        for name in SPLIT_NAMES:
            s = summary[scheme][name]
            print(
                f"  {name:<5} {s['n_essays']:>7,} essays  {s['ai_share']:>6.1%} AI  "
                f"{len(s['topics']):>2} topics"
            )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nTest topics (topic split): {summary['topic']['test']['topics']}")
    print(f"Wrote splits to {args.out}/ and summary to {summary_path}")


if __name__ == "__main__":
    main()
