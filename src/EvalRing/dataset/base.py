"""
Base classes for datasets in the EvalRing framework.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DataSample:
    """Represents a single data sample for evaluation."""

    id: str
    input_text: str
    target_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "input_text": self.input_text,
            "target_output": self.target_output,
            "metadata": self.metadata,
        }


class BaseDataset(ABC):
    """
    Abstract base class for all datasets in EvalRing.

    This class provides a standardized interface for loading and accessing
    different types of datasets (JSON, CSV, DataFrame, etc.).
    """

    def __init__(self, name: str, description: str | None = None, version: str = "1.0"):
        self.name = name
        self.description = description or f"Dataset: {name}"
        self.version = version
        self._samples: list[DataSample] = []
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def load_data(
        self,
        source: str | Path | pd.DataFrame,
        text_field: str = "text",
        label_field: str = "label",
        id_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load samples from the given source.

        Args:
            source: File path or in-memory frame, depending on the subclass.
            text_field: Column or key holding the model input.
            label_field: Column or key holding the ground-truth label.
            id_field: Column or key holding a unique per-sample ID. When
                ``None`` the row index is used, which is only safe for datasets
                that are never resumed or retried.
            **kwargs: Reader-specific options (for example ``pandas.read_csv``
                keyword arguments).
        """
        pass

    @abstractmethod
    def validate_data(self) -> bool:
        """
        Validate the loaded data for consistency and correctness.

        Returns:
            True if data is valid, False otherwise
        """
        pass

    def assert_unique_ids(
        self, *, expected_count: int | None = None, context: str = "dataset"
    ) -> None:
        """Raise if sample IDs are not unique.

        Unique per-sample IDs are required for:
        - resuming partial runs
        - retrying failures
        - merging per-sample metrics
        """
        ids = [str(s.id) for s in self._samples]
        unique_ids = set(ids)

        if expected_count is None:
            expected_count = len(ids)

        if len(ids) != expected_count:
            raise ValueError(
                f"[{context}] Expected {expected_count} samples, but found {len(ids)}."
            )

        if len(unique_ids) != len(ids):
            # Provide a small, actionable duplicate preview.
            from collections import Counter

            counts = Counter(ids)
            dupes = [k for k, v in counts.items() if v > 1]
            preview = dupes[:10]
            raise ValueError(
                f"[{context}] Duplicate sample IDs detected: unique={len(unique_ids)} total={len(ids)} "
                f"(first {len(preview)} duplicates: {preview}). "
                "Fix by using a per-row question/message ID (e.g., an explicit 'ID' column or row index), not a user ID."
            )

    def add_sample(self, sample: DataSample) -> None:
        """Add a single sample to the dataset."""
        self._samples.append(sample)

    def get_sample(self, index: int) -> DataSample:
        """Get a sample by index."""
        return self._samples[index]

    def get_sample_by_id(self, sample_id: str) -> DataSample | None:
        """Get a sample by its ID."""
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)

    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over samples in the dataset."""
        return iter(self._samples)

    def __getitem__(self, index: int | slice) -> DataSample | list[DataSample]:
        """Get sample(s) by index or slice."""
        return self._samples[index]

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        if not self._samples:
            return {"total_samples": 0}

        # Basic statistics
        stats = {
            "total_samples": len(self._samples),
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }

        # Try to infer additional statistics
        try:
            input_lengths = [len(sample.input_text) for sample in self._samples]
            stats.update(
                {
                    "avg_input_length": sum(input_lengths) / len(input_lengths),
                    "min_input_length": min(input_lengths),
                    "max_input_length": max(input_lengths),
                }
            )
        except Exception:
            pass

        return stats

    def split(
        self, train_ratio: float = 0.8, seed: int | None = None
    ) -> tuple["BaseDataset", "BaseDataset"]:
        """
        Split the dataset into train and test sets.

        Args:
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random

        if seed is not None:
            random.seed(seed)

        indices = list(range(len(self._samples)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_dataset = self.__class__(name=f"{self.name}_train", version=self.version)
        test_dataset = self.__class__(name=f"{self.name}_test", version=self.version)

        for idx in train_indices:
            train_dataset.add_sample(self._samples[idx])

        for idx in test_indices:
            test_dataset.add_sample(self._samples[idx])

        return train_dataset, test_dataset

    def save(self, filepath: str | Path) -> None:
        """Save dataset to file."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self._metadata,
            "samples": [sample.to_dict() for sample in self._samples],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str | Path) -> None:
        """Load dataset from file."""
        with open(filepath) as f:
            data = json.load(f)

        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)
        self.version = data.get("version", self.version)
        self._metadata = data.get("metadata", {})

        self._samples = []
        for sample_data in data.get("samples", []):
            sample = DataSample(
                id=sample_data["id"],
                input_text=sample_data["input_text"],
                target_output=sample_data["target_output"],
                metadata=sample_data.get("metadata", {}),
            )
            self._samples.append(sample)
