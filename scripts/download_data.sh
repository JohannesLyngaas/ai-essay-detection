#!/usr/bin/env bash
# Download the DAIGT v2 dataset from Kaggle into data/raw/.
#
# Needs a Kaggle API token first (see data/README.md). Run from the repo root:
#     uv run bash scripts/download_data.sh
#
# No token? Download the zip from the dataset page in your browser instead
# and unzip it into data/raw/.

set -euo pipefail

mkdir -p data/raw
kaggle datasets download -d thedrcat/daigt-v2-train-dataset -p data/raw --unzip
echo "Done. Files in data/raw/:"
ls -lh data/raw
