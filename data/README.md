# Data

Nothing in this folder is committed to git, except this file.

## Get the dataset

The project uses the [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) from Kaggle.

**Option A, in the browser (simplest):** open the dataset page, click *Download*, and unzip the file into `data/raw/`.

**Option B, from the terminal:** set up a [Kaggle API token](https://www.kaggle.com/docs/api), then run:

```bash
uv run bash scripts/download_data.sh
```

Either way, you should end up with `data/raw/train_v2_drcat_02.csv`.

## Make the splits

```bash
uv run python -m essaydetect.data
```

This creates `data/splits/random/` and `data/splits/topic/`, each with `train.csv`, `val.csv` and `test.csv`. The split is seeded, so it is the same on every machine.
