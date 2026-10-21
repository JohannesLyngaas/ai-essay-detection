"""Fine-tune a transformer encoder for binary classification.  (Step 3: not written yet.)

Planned function:

    train_transformer(config, splits) -> (y_pred, y_score, details)
        1. Tokenise the essays, cut at config["max_length"] tokens.
        2. Load config["model_name"] with a fresh 2-class classification head.
        3. Fine-tune with the Hugging Face Trainer, keep the checkpoint
           with the best validation macro-F1.
        4. Predict the test set.

Works for any Hugging Face encoder: switching from DistilBERT to ModernBERT
is a one-line change in the config file.
"""


def train_transformer(config: dict, splits: dict) -> tuple:
    raise NotImplementedError("Transformer training is step 3. Run a classical config for now.")
