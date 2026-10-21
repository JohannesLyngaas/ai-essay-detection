"""Tests for evaluate.py: check the metrics on a case small enough to count by hand."""

import pytest

from essaydetect.evaluate import compute_metrics


def test_metrics_on_a_hand_counted_example():
    # 4 student essays (0) and 2 AI essays (1).
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0]  # one student essay flagged, one AI essay missed
    m = compute_metrics(y_true, y_pred)

    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)  # 1 of 4 students flagged
    assert m["ai_recall"] == pytest.approx(1 / 2)  # 1 of 2 AI essays caught
    assert "roc_auc" not in m  # no scores given


def test_roc_auc_is_one_for_a_perfect_ranking():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0
