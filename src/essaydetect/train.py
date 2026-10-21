"""Run one experiment from a config file.  (Step 2: not written yet.)

Planned usage:

    uv run python -m essaydetect.train --config configs/lr_random.yaml

What it will do:
    1. Read the YAML config.
    2. Load the right split with data.load_split(config["split"]).
    3. Call classical.train_classical or transformer.train_transformer,
       depending on config["type"].
    4. Score the test predictions with evaluate.compute_metrics and save
       everything to results/<config name>/ with evaluate.save_run.
"""
