"""Mandatory scratch-CNN baseline for the eval label budgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config


class FrameDataset(Dataset):
    """Small image dataset for the scratch-CNN baseline."""

    def __init__(self, records: list[dict[str, object]], label_to_idx: dict[str, int]) -> None:
        self.records = records
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.open(config.ROOT_DIR / str(record["frame_path"])).convert("RGB")
        image = image.resize((config.EVAL_IMAGE_SIZE, config.EVAL_IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        label = torch.tensor(self.label_to_idx[str(record["label"])], dtype=torch.long)
        return tensor, label


class ScratchCNN(nn.Module):
    """Tiny CNN trained from scratch; no pretrained weights."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (config.EVAL_IMAGE_SIZE // 4) * (config.EVAL_IMAGE_SIZE // 4), num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def train_and_eval(
    train_records: list[dict[str, object]], test_records: list[dict[str, object]]
) -> float:
    """Train the scratch CNN on selected labels and return test accuracy."""
    config.set_seed()
    labels = sorted({str(record["label"]) for record in train_records + test_records})
    label_to_idx = {label: index for index, label in enumerate(labels)}
    train_loader = DataLoader(
        FrameDataset(train_records, label_to_idx),
        batch_size=config.EVAL_CNN_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        FrameDataset(test_records, label_to_idx),
        batch_size=config.EVAL_CNN_BATCH_SIZE,
        shuffle=False,
    )
    model = ScratchCNN(len(labels)).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.EVAL_CNN_EPOCHS):
        for images, labels_tensor in train_loader:
            images = images.to(config.DEVICE)
            labels_tensor = labels_tensor.to(config.DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels_tensor)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels_tensor in test_loader:
            logits = model(images.to(config.DEVICE))
            predictions = logits.argmax(dim=1).cpu()
            correct += int((predictions == labels_tensor).sum())
            total += int(labels_tensor.numel())
    return correct / total if total else 0.0
