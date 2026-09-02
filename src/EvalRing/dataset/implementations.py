"""
Concrete implementations of datasets.
"""

import json
from pathlib import Path

import pandas as pd

from .base import BaseDataset, DataSample


class JSONDataset(BaseDataset):
    """Dataset loaded from a JSON file."""

    def load_data(
        self,
        source: str | Path | pd.DataFrame,
        text_field: str = "text",
        label_field: str = "label",
        id_field: str | None = "id",
        **kwargs,
    ) -> None:
        """
        Load data from a JSON file.

        Args:
            source: Path to JSON file
            text_field: Key for input text
            label_field: Key for target output
            id_field: Key for sample ID (auto-generated if None)
        """
        if isinstance(source, pd.DataFrame):
            raise TypeError("JSONDataset expects a file path; use DataFrameDataset instead.")

        filepath = Path(source)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of objects")

        for i, item in enumerate(data):
            sample_id = str(item.get(id_field, i)) if id_field else str(i)

            # Extract metadata (everything except text and label)
            metadata = {
                k: v for k, v in item.items() if k not in [text_field, label_field, id_field]
            }

            sample = DataSample(
                id=sample_id,
                input_text=item.get(text_field, ""),
                target_output=item.get(label_field),
                metadata=metadata,
            )
            self.add_sample(sample)

    def validate_data(self) -> bool:
        """Validate that all samples have input text and target output."""
        for sample in self._samples:
            if not sample.input_text or sample.target_output is None:
                return False
        return True


class CSVDataset(BaseDataset):
    """Dataset loaded from a CSV file."""

    def load_data(
        self,
        source: str | Path | pd.DataFrame,
        text_field: str = "text",
        label_field: str = "label",
        id_field: str | None = None,
        **kwargs,
    ) -> None:
        """
        Load data from a CSV file.

        Args:
            source: Path to CSV file
            text_field: Column name for input text
            label_field: Column name for target output
            id_field: Column name for sample ID (auto-generated if None)
        """
        if isinstance(source, pd.DataFrame):
            raise TypeError("CSVDataset expects a file path; use DataFrameDataset instead.")

        filepath = Path(source)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, **kwargs)

        if text_field not in df.columns:
            raise ValueError(f"Text field '{text_field}' not found in CSV columns")

        if label_field not in df.columns:
            raise ValueError(f"Label field '{label_field}' not found in CSV columns")

        for i, row in df.iterrows():
            sample_id = str(row[id_field]) if id_field and id_field in df.columns else str(i)

            # Extract metadata
            metadata_cols = [c for c in df.columns if c not in [text_field, label_field, id_field]]
            metadata = {col: row[col] for col in metadata_cols}

            sample = DataSample(
                id=sample_id,
                input_text=str(row[text_field]),
                target_output=row[label_field],
                metadata=metadata,
            )
            self.add_sample(sample)

    def validate_data(self) -> bool:
        """Validate that all samples have input text and target output."""
        for sample in self._samples:
            if not sample.input_text or pd.isna(sample.target_output):
                return False
        return True


class DataFrameDataset(BaseDataset):
    """Dataset loaded from a pandas DataFrame."""

    def load_data(
        self,
        source: str | Path | pd.DataFrame,
        text_field: str = "text",
        label_field: str = "label",
        id_field: str | None = None,
        **kwargs,
    ) -> None:
        """
        Load data from a pandas DataFrame.

        Args:
            source: pandas DataFrame
            text_field: Column name for input text
            label_field: Column name for target output
            id_field: Column name for sample ID (auto-generated if None)
        """
        if not isinstance(source, pd.DataFrame):
            raise ValueError("Source must be a pandas DataFrame")

        df = source

        if text_field not in df.columns:
            raise ValueError(f"Text field '{text_field}' not found in DataFrame columns")

        if label_field not in df.columns:
            raise ValueError(f"Label field '{label_field}' not found in DataFrame columns")

        for i, row in df.iterrows():
            sample_id = str(row[id_field]) if id_field and id_field in df.columns else str(i)

            # Extract metadata
            metadata_cols = [c for c in df.columns if c not in [text_field, label_field, id_field]]
            metadata = {col: row[col] for col in metadata_cols}

            sample = DataSample(
                id=sample_id,
                input_text=str(row[text_field]),
                target_output=row[label_field],
                metadata=metadata,
            )
            self.add_sample(sample)

    def validate_data(self) -> bool:
        """Validate that all samples have input text and target output."""
        for sample in self._samples:
            if not sample.input_text or pd.isna(sample.target_output):
                return False
        return True
