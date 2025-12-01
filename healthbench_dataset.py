from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class HealthBenchDataset:
    """
    Utility for loading HealthBench jsonl files and exposing a deterministic
    train/validation split.
    """

    def __init__(
        self,
        filename: str,
        train_fraction: float = 0.9,
        seed: int = 0,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive)")

        self.filename = filename
        self.train_fraction = train_fraction
        self.seed = seed

        self._dataset_path = Path(__file__).resolve().parent / "healthbench" / self.filename
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {self._dataset_path}")

        self._examples = self._load_examples()
        if len(self._examples) < 2:
            raise ValueError(
                f"Expected at least two examples in {self._dataset_path} to make a split"
            )

        self.train_examples, self.val_examples = self._split_examples()

    def _load_examples(self) -> list[dict[str, Any]]:
        with self._dataset_path.open("r", encoding="utf-8") as infile:
            return [json.loads(line) for line in infile if line.strip()]

    def _split_examples(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        indices = list(range(len(self._examples)))
        random.Random(self.seed).shuffle(indices)
        split_index = int(len(indices) * self.train_fraction)
        split_index = min(len(indices) - 1, max(1, split_index))
        train_idx = indices[:split_index]
        val_idx = indices[split_index:]
        train = [self._examples[i] for i in train_idx]
        val = [self._examples[i] for i in val_idx]
        return train, val

    def __len__(self) -> int:
        return len(self._examples)
