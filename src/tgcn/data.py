"""UAM loader: original sample order and preprocessing with lossless index masks."""

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .transforms import (
    Compose,
    ResizeWithPaddingCV,
    ToTensor,
    RandomScale,
    RandomTranslate,
    RandomShear,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomBrightnessContrast,
    RandomGammaCorrection,
)


def transform(training=False):
    operations = []
    if training:
        operations = [
            RandomScale((0.7, 1.3), p=0.5),
            RandomTranslate((0.05, 0.05), p=0.5),
            RandomShear(0.1, p=0.3),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(20, p=0.5),
            RandomBrightnessContrast((0.7, 1.3), (0.7, 1.3), p=0.5),
            RandomGammaCorrection((0.7, 1.5), p=0.5),
        ]
    # This source resize swaps RGB back to BGR; preserve it for existing weights.
    return Compose(operations + [ResizeWithPaddingCV((224, 224)), ToTensor()])


class UAMDataset(Dataset):
    def __init__(self, root, split="test", training=False):
        self.root = Path(root)
        self.split = split
        order = self.root / "splits" / f"{split}.json"
        if not order.is_file():
            raise FileNotFoundError(
                f"Missing {order}. Extract the complete UAM release asset first."
            )
        self.names = json.loads(order.read_text())
        self.transform = transform(training)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.root / self.split / "images" / f"{name}.jpg"))
        mask = cv2.imread(
            str(self.root / self.split / "masks" / f"{name}.png"), cv2.IMREAD_UNCHANGED
        )
        if image is None or mask is None:
            raise FileNotFoundError(
                f"Missing/unreadable UAM sample: {self.split}/{name}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {name}: {image.shape} / {mask.shape}"
            )
        if mask.dtype != np.uint8 or mask.max() > 2:
            raise ValueError(f"Expected uint8 class-index mask (0,1,2): {name}")
        return self.transform({"image": image, "mask": mask, "name": name})


def loader(
    root, split="test", batch_size=10, training=False, workers=0, drop_last=False
):
    return DataLoader(
        UAMDataset(root, split, training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=True,
    )
