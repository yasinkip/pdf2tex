"""
Run with `uv run --with python 3.12 --with kagglehub src/ocr/data/kaggle.py` from project root.
"""

from pathlib import Path

import kagglehub

IM2LATEX_PATH = Path(__file__).resolve().parents[0] / "im2latex-230k"

# Download latest version
if __name__ == "__main__":
    IM2LATEX_PATH.mkdir(exist_ok=True)
    saved_path = kagglehub.dataset_download(
        "gregoryeritsyan/im2latex-230k", output_dir=str(IM2LATEX_PATH)
    )
