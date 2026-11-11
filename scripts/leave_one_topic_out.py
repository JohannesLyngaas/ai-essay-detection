"""Leave-one-topic-out cross-validation for the two classical models.

    uv run python scripts/leave_one_topic_out.py

The topic split holds out three topics and tests on those. That answers "how well does
the model do on unseen topics?" with a sample size of three, which is thin. This script
repeats the same idea fifteen times: for each topic in turn, train on the other fourteen
and test on the held-out one. Every topic gets to be the test set exactly once, so we
can see whether the errors on the topic split are a property of the method or of the
three topics that happened to be held out.

Settings are deliberately fixed, not tuned:

- TF-IDF features come from configs/lr_topic.yaml, so they match the reported runs.
- C is fixed to the value each model chose on the topic split's validation set
  (1.0 for logistic regression, 0.1 for the linear SVM). Re-tuning C inside every fold
  would need a validation topic per fold and would make the folds harder to compare.
- The default threshold is used (decision value > 0), the same as in the reported runs.

Writes results/loto/per_topic.csv and prints two tables: every topic sorted by false
positive rate, and the false positive rate pooled by prompt type.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from essaydetect.classical import make_classifier
from essaydetect.data import LABEL_COL, TEXT_COL, TOPIC_COL, load_split
from essaydetect.evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent

# C per model, fixed to what each one picked on the topic split's validation set.
MODELS = {"lr": ("logreg", 1.0), "svm": ("svm", 0.1)}
SEED = 42

# The seven PERSUADE prompts that give the student a source text to write about
# ("source-based" in the PERSUADE papers, also called text-dependent). The other eight
# are independent prompts: the student writes from their own opinion, with no text.
#
# Where this list comes from, in order of how directly it is evidence:
#
# 1. The DAIGT v2 CSV itself. data/raw/train_v2_drcat_02.csv has a column
#    `RDizzl3_seven` that is 1 for exactly these seven prompts and 0 for the other
#    eight. (The column is named after the DAIGT competition subset, not after
#    PERSUADE, but it selects exactly the source-based set - see points 2 and 3.)
# 2. The PERSUADE 1.0 paper (Crossley et al. 2022, PMC9767095) states that the corpus
#    has seven source-based (text-dependent) prompts and eight independent prompts,
#    fifteen in total. Our seven and eight match those counts exactly. The paper does
#    not print the prompt names, so it confirms the split sizes, not the membership.
# 3. A check on the essays themselves: the share of STUDENT essays that mention
#    "the article", "the author", "the passage", "the text" or "paragraph" is 19-83 %
#    for these seven topics and 0.1-8.5 % for the other eight. The gap is clean, and
#    it puts the same seven topics on the source-based side.
#
# So the membership is taken from the dataset (1) and cross-checked two ways (2, 3).
# It is not copied from a published table of prompt names; if the report needs to cite
# one, check the prompt files shipped with the PERSUADE 2.0 corpus itself.
TEXT_DEPENDENT = {
    '"A Cowboy Who Rode the Waves"',
    "Car-free cities",
    "Does the electoral college work?",
    "Driverless cars",
    "Exploring Venus",
    "Facial action coding system",
    "The Face on Mars",
}


def load_all_essays() -> pd.DataFrame:
    """The whole cleaned dataset: the three parts of the random split put back together."""
    parts = load_split("random", ROOT / "data/splits")
    return pd.concat(parts.values(), ignore_index=True)


def load_feature_settings() -> dict:
    """The TfidfVectorizer settings from the configs, so the folds match the real runs."""
    with open(ROOT / "configs/lr_topic.yaml") as f:
        features = yaml.safe_load(f)["features"]
    with open(ROOT / "configs/svm_topic.yaml") as f:
        other = yaml.safe_load(f)["features"]
    # Both classical configs are supposed to use identical features, so that the two
    # models differ only in the classifier. Fail loudly if that ever stops being true.
    assert features == other, "lr_topic.yaml and svm_topic.yaml have different features"
    return features


def run_one_fold(train: pd.DataFrame, test: pd.DataFrame, features: dict) -> dict:
    """Train both classifiers on `train`, score them on `test`.

    The vectorizer is fitted once per fold and shared, so both classifiers see exactly
    the same features and any difference comes from the classifier alone. It is fitted
    on the training topics only, so no vocabulary from the held-out topic leaks in.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(features["ngram_range"]),
        min_df=features["min_df"],
        max_features=features["max_features"],
        sublinear_tf=features["sublinear_tf"],
    )
    X_train = vectorizer.fit_transform(train[TEXT_COL])
    X_test = vectorizer.transform(test[TEXT_COL])

    results = {}
    for short, (classifier_name, C) in MODELS.items():
        classifier = make_classifier(classifier_name, C, SEED)
        classifier.fit(X_train, train[LABEL_COL])
        y_pred = classifier.predict(X_test)
        y_score = classifier.decision_function(X_test)  # works for both models
        # The same scoring function every other experiment uses, so these numbers are
        # directly comparable to results/<run>/metrics.json.
        results[short] = compute_metrics(test[LABEL_COL], y_pred, y_score)
    return results


def main() -> None:
    essays = load_all_essays()
    features = load_feature_settings()
    topics = sorted(essays[TOPIC_COL].unique())

    unknown = set(TEXT_DEPENDENT) - set(topics)
    assert not unknown, f"These text-dependent topics are not in the data: {unknown}"
    print(f"{len(essays):,} essays, {len(topics)} topics, {len(TEXT_DEPENDENT)} text-dependent\n")

    rows, counts = [], []
    for i, topic in enumerate(topics, start=1):
        test = essays[essays[TOPIC_COL] == topic]
        train = essays[essays[TOPIC_COL] != topic]
        n_ai = int(test[LABEL_COL].sum())
        n_student = len(test) - n_ai

        print(f"[{i:>2}/{len(topics)}] {topic}  ({n_student:,} student, {n_ai:,} AI)", flush=True)
        results = run_one_fold(train, test, features)

        row = {
            "topic": topic,
            "prompt_type": "text-dependent" if topic in TEXT_DEPENDENT else "independent",
            "n_student": n_student,
            "n_ai": n_ai,
        }
        for short, metrics in results.items():
            row[f"{short}_fpr"] = metrics["false_positive_rate"]
            row[f"{short}_ai_recall"] = metrics["ai_recall"]
            row[f"{short}_roc_auc"] = metrics["roc_auc"]
            print(
                f"        {short:<4} FPR {metrics['false_positive_rate']:.3f}   "
                f"AI recall {metrics['ai_recall']:.3f}   ROC-AUC {metrics['roc_auc']:.3f}"
            )
        rows.append(row)
        # Raw counts, kept for the pooled numbers below. Pooling rates would weight a
        # 1,600-essay topic the same as a 4,700-essay one; pooling counts does not.
        counts.append(
            {
                "topic": topic,
                "prompt_type": row["prompt_type"],
                **{
                    f"{short}_{name}": results[short]["confusion_matrix"][name]
                    for short in MODELS
                    for name in ("tn", "fp", "fn", "tp")
                },
            }
        )

    table = pd.DataFrame(rows)
    out = ROOT / "results/loto"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_topic.csv", index=False)

    print("\n" + "=" * 78)
    print("Every topic, sorted by logistic regression's false positive rate")
    print("(the share of that topic's student essays wrongly flagged as AI)")
    print("=" * 78)
    print(table.sort_values("lr_fpr", ascending=False).to_string(index=False))

    print("\n" + "=" * 78)
    print("False positive rate pooled by prompt type")
    print("=" * 78)
    pooled = (
        pd.DataFrame(counts)
        .groupby("prompt_type")[[f"{short}_{name}" for short in MODELS for name in ("fp", "tn")]]
        .sum()
    )
    summary = pd.DataFrame(index=pooled.index)
    summary["n_topics"] = pd.DataFrame(counts).groupby("prompt_type").size()
    summary["n_student"] = pooled["lr_fp"] + pooled["lr_tn"]
    for short in MODELS:
        summary[f"{short}_fpr"] = pooled[f"{short}_fp"] / (
            pooled[f"{short}_fp"] + pooled[f"{short}_tn"]
        )
    print(summary.round(4).to_string())

    print(f"\nWrote {(out / 'per_topic.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
