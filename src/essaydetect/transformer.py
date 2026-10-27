"""Fine-tune a transformer encoder for binary classification.

What "fine-tuning" means here: we download an encoder that has already been
pretrained on a lot of text (so it knows English), put a fresh 2-class
classification head on top, and keep training the whole thing on our essays.
Unlike the TF-IDF models, which only see which n-grams occur, the encoder sees the
words in order and in context.

The steps below:
    1. Tokenise the essays, cutting them at config["max_length"] tokens.
    2. Load config["model_name"] with a new 2-class head.
    3. Fine-tune with the Hugging Face Trainer, evaluating on val after every epoch
       and keeping the checkpoint with the best validation macro-F1.
    4. Predict the test set and return the AI probability per essay.

Works for any Hugging Face encoder: switching from DistilBERT to ModernBERT is a
one-line change in the config file.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from essaydetect.data import LABEL_COL, TEXT_COL


def macro_f1_for_trainer(eval_pred) -> dict:
    """Macro-F1 for the Trainer's own evaluation after each epoch.

    The Trainer hands us the raw model outputs (logits) and the true labels. The
    predicted class is the column with the larger logit. The name of the returned
    key, "macro_f1", becomes "eval_macro_f1" in the logs, which is what
    metric_for_best_model points at below.

    The final test scores do NOT come from here: train.py scores the test set with
    evaluate.compute_metrics, the same function every other model goes through.
    """
    logits, labels = eval_pred
    predictions = np.asarray(logits).argmax(axis=-1)
    return {"macro_f1": f1_score(labels, predictions, average="macro")}


def tokenise_split(df, tokenizer, max_length: int) -> Dataset:
    """Turn one DataFrame into a tokenised Dataset the Trainer can read.

    Essays longer than max_length tokens are cut off at the end (truncation). We do
    NOT pad here: padding is done per batch by the data collator, so a batch of short
    essays stays short instead of being stretched to max_length. That is "dynamic
    padding", and it saves a lot of time.
    """
    frame = df[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"})
    data = Dataset.from_pandas(frame, preserve_index=False)
    return data.map(
        lambda batch: tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=[TEXT_COL],  # the collator cannot stack raw strings
    )


def share_truncated(texts, tokenizer, max_length: int) -> float:
    """Share of essays longer than max_length tokens, i.e. the share that gets cut off.

    Worth reporting: if a quarter of the test essays lose their ending, the model is
    judged on incomplete essays, and that is part of the explanation for its errors.
    """
    encoded = tokenizer(list(texts), truncation=False, verbose=False)["input_ids"]
    return float(np.mean([len(ids) > max_length for ids in encoded]))


def train_transformer(config: dict, splits: dict) -> tuple:
    """Fine-tune one encoder, keep the best epoch by val macro-F1, predict test.

    Returns (y_pred, y_score, details) where y_score is the model's probability that
    the essay is AI-generated, so it goes straight into ROC-AUC like the decision
    values from the classical models.
    """
    train, val, test = splits["train"], splits["val"], splits["test"]

    # Seeds the shuffling, the dropout and the new classification head, so a rerun
    # with the same config gives the same model.
    set_seed(config["seed"])

    # fp16 ("half precision") stores numbers in 16 instead of 32 bits. On a GPU that
    # is faster and uses less memory; on a CPU it is unsupported, so we switch it off
    # whatever the config says. That is what lets the same config run on the laptop.
    use_gpu = torch.cuda.is_available()
    fp16 = bool(config.get("fp16")) and use_gpu
    print(f"Device: {'GPU' if use_gpu else 'CPU'}, fp16 = {fp16}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,  # 2 classes: student (0) and AI (1)
    )

    max_length = config["max_length"]
    datasets = {
        name: tokenise_split(part, tokenizer, max_length)
        for name, part in (("train", train), ("val", val), ("test", test))
    }

    # How many optimiser steps the whole run takes. One step uses batch_size essays
    # times gradient_accumulation_steps (the gradients of several small batches are
    # summed before the weights are updated, which imitates a larger batch on a
    # small GPU).
    effective_batch = config["batch_size"] * config["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train) / effective_batch)
    total_steps = steps_per_epoch * config["epochs"]
    # Warmup: start at learning rate 0 and ramp up over the first warmup_ratio of the
    # run, so the first few updates do not wreck the pretrained weights.
    # TrainingArguments accepted a warmup_ratio until transformers 5, which only takes
    # warmup_steps, so we convert it here. That works on both versions.
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    print(
        f"{len(train):,} train essays, effective batch {effective_batch}, "
        f"{total_steps} steps, {warmup_steps} of them warmup"
    )

    args = TrainingArguments(
        output_dir=f"checkpoints/{config['name']}",  # gitignored: weights are large
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,  # no gradients, so more fits
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        weight_decay=config["weight_decay"],
        warmup_steps=warmup_steps,
        fp16=fp16,
        # Score the val set after every epoch, save a checkpoint at the same moments,
        # and at the end put back the weights from the epoch with the best val
        # macro-F1. Without this we would keep the last epoch, which may be overfitted.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,  # keep one checkpoint on disk (the best one is protected)
        logging_steps=50,
        report_to="none",  # no Weights & Biases or other online tracking
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),  # dynamic padding per batch
        compute_metrics=macro_f1_for_trainer,
    )
    trainer.train()

    # The Trainer logs one row per evaluation. Pull out the val score per epoch, so the
    # report can show the learning curve and which epoch was kept.
    per_epoch = [
        {"epoch": round(row["epoch"], 2), "val_macro_f1": round(float(row["eval_macro_f1"]), 4)}
        for row in trainer.state.log_history
        if "eval_macro_f1" in row
    ]
    best = max(per_epoch, key=lambda row: row["val_macro_f1"])
    print(f"Best epoch: {best['epoch']} (val macro-F1 {best['val_macro_f1']:.4f})")

    output = trainer.predict(datasets["test"])
    logits = output.predictions
    if isinstance(logits, tuple):  # some models return extra outputs
        logits = logits[0]
    # softmax turns the two logits into two probabilities that sum to 1.
    probabilities = torch.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32), dim=-1)

    y_pred = probabilities.argmax(dim=-1).numpy()
    y_score = probabilities[:, 1].numpy()  # column 1 = probability of "AI-generated"

    details = {
        "model_name": config["model_name"],
        "max_length": max_length,
        "effective_batch_size": effective_batch,
        "warmup_steps": warmup_steps,
        "fp16": fp16,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "val_macro_f1_per_epoch": per_epoch,
        "test_share_truncated": round(share_truncated(test[TEXT_COL], tokenizer, max_length), 4),
    }
    return y_pred, y_score, details
