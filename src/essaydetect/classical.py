"""TF-IDF features + logistic regression or linear SVM.

How it works:
1. TfidfVectorizer turns each essay into a long, mostly-zero vector with one number
   per word or word pair (n-gram). It learns the vocabulary from the TRAINING essays
   only, so nothing about the test set leaks into the features.
2. A linear classifier learns one weight per n-gram. Positive weights push an essay
   towards "AI", negative weights towards "student".
3. The regularisation strength C is chosen on the validation set. Small C = simpler
   model with smaller weights, large C = fits the training data more closely.

Both classifiers get the exact same features, so any difference in the results
comes from the classifier, not the features. Both are also solved by the same
library (LIBLINEAR), which keeps the comparison clean.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from essaydetect.data import LABEL_COL, TEXT_COL


def make_classifier(name: str, C: float, seed: int):
    if name == "logreg":
        return LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=seed)
    if name == "svm":
        return LinearSVC(C=C, dual=True, max_iter=5000, random_state=seed)
    raise ValueError(f"Unknown classifier {name!r}, expected 'logreg' or 'svm'.")


def top_features(vectorizer: TfidfVectorizer, classifier, n: int = 25) -> dict:
    """The n-grams with the largest weights towards each class.

    A cheap way to see what the model actually uses as evidence. Useful for the
    error analysis: if the top "AI" n-grams are topic words, the model has learned
    topics, not style.
    """
    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    order = np.argsort(weights)
    return {
        "ai": [(names[i], round(float(weights[i]), 3)) for i in order[::-1][:n]],
        "student": [(names[i], round(float(weights[i]), 3)) for i in order[:n]],
    }


def train_classical(config: dict, splits: dict) -> tuple:
    """Fit TF-IDF + classifier, pick C on val, predict test.

    Returns (y_pred, y_score, details) where y_score is the decision value per test
    essay (higher = more likely AI) and details holds the tuning results.
    """
    train, val, test = splits["train"], splits["val"], splits["test"]
    feats = config["features"]

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(feats["ngram_range"]),
        min_df=feats["min_df"],
        max_features=feats["max_features"],
        sublinear_tf=feats["sublinear_tf"],
    )
    X_train = vectorizer.fit_transform(train[TEXT_COL])  # learn vocabulary + weights
    X_val = vectorizer.transform(val[TEXT_COL])  # reuse them, never refit
    X_test = vectorizer.transform(test[TEXT_COL])
    print(f"TF-IDF: {X_train.shape[0]:,} essays x {X_train.shape[1]:,} n-grams")

    # Try each C, keep the one with the best macro-F1 on the validation set.
    tuning, best = [], None
    for C in config["C_grid"]:
        clf = make_classifier(config["classifier"], C, config["seed"])
        clf.fit(X_train, train[LABEL_COL])
        val_f1 = f1_score(val[LABEL_COL], clf.predict(X_val), average="macro")
        tuning.append({"C": C, "val_macro_f1": round(float(val_f1), 4)})
        print(f"  C={C:<6} val macro-F1 = {val_f1:.4f}")
        if best is None or val_f1 > best[0]:
            best = (val_f1, C, clf)

    best_f1, best_C, clf = best
    print(f"Best C = {best_C}")

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)  # works for both models

    details = {
        "best_C": best_C,
        "tuning": tuning,
        "n_features": X_train.shape[1],
        "top_features": top_features(vectorizer, clf),
    }
    return y_pred, y_score, details
