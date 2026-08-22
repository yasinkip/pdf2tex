"""
Load data into custom torch dataset.
"""

from os import PathLike
from typing import Any

from torchvision.io import decode_image
from torch.utils.data import Dataset


class Im2Latex230kDataset(Dataset):
    def __init__(
        self,
        root_path: PathLike,
        num: int | None = None,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.path = root_path
        self.num = num
        self.transform = transform
        self.target_transform = target_transform

        self.pairs = self._load_pairs(num)

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx: int):
        png_path, formula = self.pairs[idx]
        image = decode_image(png_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            formula = self.target_transform(image)

        return image, formula

    def _load_pairs(self, num: int | None = None) -> tuple[list[str], list[str]]:
        im2latex_dir = self.path / "PRINTED_TEX_230k"
        png_dir = im2latex_dir / "generated_png_images"

        with open(im2latex_dir / "corresponding_png_images.txt") as png_map:
            png_lines = [line.rstrip("\r\n") for line in png_map.readlines()]
            png_paths = [png_dir / png for png in png_lines]
            if num is not None:
                png_paths = png_paths[:num]

        with open(im2latex_dir / "final_png_formulas.txt") as png_formulas:
            formulas = [line.rstrip("\r\n") for line in png_formulas.readlines()]
            if num is not None:
                formulas = formulas[:num]

        return list(zip(png_paths, formulas))
