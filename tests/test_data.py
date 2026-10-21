"""Tests for data.py, run with `uv run pytest`.

They use a tiny fake dataset, so they run in about a second and need no download.
"""

import pandas as pd
import pytest

from essaydetect.data import load_and_clean, make_random_split, make_topic_split


def fake_essays(n_topics: int = 8, per_topic: int = 20) -> pd.DataFrame:
    """A small dataset shaped like DAIGT v2: half AI, half human, in every topic."""
    rows = [
        {"text": f"essay {t}-{i}", "label": i % 2, "prompt_name": f"topic {t}"}
        for t in range(n_topics)
        for i in range(per_topic)
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df


def test_topic_split_has_no_shared_topics():
    splits = make_topic_split(fake_essays(), seed=0)
    train, val, test = (set(splits[name]["prompt_name"]) for name in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


@pytest.mark.parametrize("make_split", [make_random_split, make_topic_split])
def test_every_essay_lands_in_exactly_one_part(make_split):
    df = fake_essays()
    splits = make_split(df, seed=0)
    ids = pd.concat(splits.values())["id"]
    assert sorted(ids) == sorted(df["id"])


def test_random_split_keeps_class_balance():
    splits = make_random_split(fake_essays(), seed=0)
    for part in splits.values():
        assert part["label"].mean() == pytest.approx(0.5, abs=0.05)


def test_splits_are_reproducible():
    first = make_topic_split(fake_essays(), seed=42)
    second = make_topic_split(fake_essays(), seed=42)
    assert set(first["test"]["id"]) == set(second["test"]["id"])


def test_cleaning_removes_duplicates_and_conflicts(tmp_path):
    raw = pd.DataFrame(
        {
            "text": ["same essay", "same essay", "unsure", "unsure", "  ", "fine"],
            "label": [1, 1, 0, 1, 0, 0],
            "prompt_name": ["a"] * 6,
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    df = load_and_clean(path)

    # "same essay" kept once, "unsure" dropped (two labels), blank dropped, "fine" kept
    assert sorted(df["text"]) == ["fine", "same essay"]
    assert list(df["id"]) == [0, 1]
