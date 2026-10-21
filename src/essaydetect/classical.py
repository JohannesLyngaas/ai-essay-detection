"""TF-IDF features + logistic regression or linear SVM.  (Step 2: not written yet.)

Planned function:

    train_classical(config, splits) -> (y_pred, y_score)
        1. Fit a TfidfVectorizer on the TRAINING essays only.
        2. Try each C in config["C_grid"], score each on the validation set.
        3. Refit with the best C and predict the test set.

Both classifiers use the same features, so any difference in the results
comes from the classifier, not the features.
"""
