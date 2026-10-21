"""Score every model in exactly the same way.  (Step 2: not written yet.)

Planned functions:

    compute_metrics(y_true, y_pred) -> dict
        accuracy, F1 for the AI class, macro-F1, the false positive rate
        (share of student essays flagged as AI) and the confusion matrix.

    save_run(name, test_df, y_pred, y_score, metrics, results_dir="results")
        writes results/<name>/metrics.json and results/<name>/predictions.csv.
        The predictions file keeps the essay id, topic, true label, predicted
        label and score, so the error analysis never needs to retrain anything.

Keeping this in one shared file is what makes the comparison between the
classical models and the transformers fair.
"""
