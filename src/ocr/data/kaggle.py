"""
Run with `uv run --with kagglehub src/ocr/data/kaggle.py`
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("gregoryeritsyan/im2latex-230k")

print("Path to dataset files:", path)

