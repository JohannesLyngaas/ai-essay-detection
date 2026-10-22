# Results

Everything here is committed, so every number in the report can be traced to a file.

```
results/
├── splits/split_summary.json   which topics are in each split, sizes, class balance
└── <experiment name>/          one folder per config in configs/
    ├── metrics.json            accuracy, F1, macro-F1, AI recall (also at 1 % FPR),
    │                           false positive rate, ROC-AUC, confusion matrix
    └── predictions.csv         id, topic, true label, predicted label, score per test essay
```

Model weights are **not** stored here. They are too large for GitHub.
