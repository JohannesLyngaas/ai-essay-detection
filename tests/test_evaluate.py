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
    assert "ai_recall_at_1pct_fpr" not in m  # same: needs scores


def test_roc_auc_is_one_for_a_perfect_ranking():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0


# 100 student essays, so 1 % of them is exactly one essay. That makes the budget easy
# to reason about: at most one student essay may end up above the threshold.
STUDENTS = [0.0] * 99 + [0.9]  # 99 clearly-student scores + one that looks AI-ish
AI = [0.5, 0.5, 0.95, 0.95]  # 2 borderline AI essays + 2 obvious ones
Y_TRUE = [0] * 100 + [1] * 4


def test_ai_recall_at_1pct_fpr_allows_one_false_positive_in_a_hundred():
    # A threshold just above 0.5 flags all 4 AI essays and the single 0.9 student essay:
    # 1 of 100 students = 1 % FPR, which is inside the budget. So all 4 are allowed.
    m = compute_metrics(Y_TRUE, [0] * 104, y_score=STUDENTS + AI)
    assert m["ai_recall_at_1pct_fpr"] == 1.0


def test_ai_recall_at_1pct_fpr_with_two_false_positives():
    # Same data, but one more student essay scoring 0.6. The threshold above 0.5 now
    # flags 2 of 100 students = 2 % FPR, too high. We have to move above 0.9, where
    # only the two 0.95 AI essays are caught: 2 of 4 = 0.5 recall.
    y_score = STUDENTS + [0.6] + AI
    m = compute_metrics([0] * 101 + [1] * 4, [0] * 105, y_score=y_score)
    assert m["ai_recall_at_1pct_fpr"] == 0.5
