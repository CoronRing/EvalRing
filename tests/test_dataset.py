"""Dataset loading, validation, and the unique-ID guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from EvalRing.dataset import CSVDataset, DataFrameDataset, DataSample, JSONDataset


class TestJSONDataset:
    def test_loads_every_record(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label")
        assert len(dataset) == 3

    def test_uses_the_id_field_for_sample_ids(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label", id_field="id")
        assert [s.id for s in dataset] == ["a1", "a2", "a3"]

    def test_unmapped_columns_land_in_metadata(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label", id_field="id")
        assert dataset[0].metadata["source"] == "web"

    def test_falls_back_to_row_index_without_an_id_field(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label", id_field=None)
        assert [s.id for s in dataset] == ["0", "1", "2"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        dataset = JSONDataset(name="missing")
        with pytest.raises(FileNotFoundError):
            dataset.load_data(tmp_path / "nope.json")

    def test_non_list_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "object.json"
        path.write_text(json.dumps({"text": "a", "label": "b"}), encoding="utf-8")
        dataset = JSONDataset(name="object")
        with pytest.raises(ValueError, match="list of objects"):
            dataset.load_data(path)

    def test_validation_fails_when_a_label_is_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps([{"text": "a", "label": None}]), encoding="utf-8")
        dataset = JSONDataset(name="partial")
        dataset.load_data(path)
        assert dataset.validate_data() is False


class TestCSVDataset:
    def test_loads_every_row(self, csv_dataset_file: Path) -> None:
        dataset = CSVDataset(name="reviews")
        dataset.load_data(csv_dataset_file, text_field="text", label_field="label", id_field="id")
        assert len(dataset) == 3
        assert dataset[1].target_output == "negative"

    def test_unknown_text_column_raises(self, csv_dataset_file: Path) -> None:
        dataset = CSVDataset(name="reviews")
        with pytest.raises(ValueError, match="Text field"):
            dataset.load_data(csv_dataset_file, text_field="absent", label_field="label")

    def test_unknown_label_column_raises(self, csv_dataset_file: Path) -> None:
        dataset = CSVDataset(name="reviews")
        with pytest.raises(ValueError, match="Label field"):
            dataset.load_data(csv_dataset_file, text_field="text", label_field="absent")


class TestDataFrameDataset:
    def test_loads_from_a_dataframe(self, sample_records: list[dict[str, object]]) -> None:
        dataset = DataFrameDataset(name="frame")
        dataset.load_data(pd.DataFrame(sample_records), text_field="text", label_field="label")
        assert len(dataset) == 3

    def test_non_dataframe_source_is_rejected(self) -> None:
        dataset = DataFrameDataset(name="frame")
        with pytest.raises(ValueError, match="pandas DataFrame"):
            dataset.load_data([{"text": "a", "label": "b"}])  # type: ignore[arg-type]


class TestUniqueIds:
    """Resume, retry, and per-sample merges all key on the sample ID."""

    def _dataset_with_ids(self, ids: list[str]) -> JSONDataset:
        dataset = JSONDataset(name="ids")
        for sample_id in ids:
            dataset.add_sample(DataSample(id=sample_id, input_text="text", target_output="label"))
        return dataset

    def test_unique_ids_pass(self) -> None:
        self._dataset_with_ids(["a", "b", "c"]).assert_unique_ids()

    def test_duplicate_ids_raise_and_name_the_duplicate(self) -> None:
        dataset = self._dataset_with_ids(["a", "b", "a"])
        with pytest.raises(ValueError, match="Duplicate sample IDs"):
            dataset.assert_unique_ids()

    def test_unexpected_count_raises(self) -> None:
        dataset = self._dataset_with_ids(["a", "b"])
        with pytest.raises(ValueError, match="Expected 5 samples"):
            dataset.assert_unique_ids(expected_count=5)


class TestRoundTrip:
    def test_save_then_load_preserves_samples(
        self, json_dataset_file: Path, tmp_path: Path
    ) -> None:
        original = JSONDataset(name="reviews", description="d", version="2.0")
        original.load_data(json_dataset_file, text_field="text", label_field="label", id_field="id")

        out = tmp_path / "saved.json"
        original.save(out)

        restored = JSONDataset(name="placeholder")
        restored.load_from_file(out)

        assert restored.name == "reviews"
        assert restored.version == "2.0"
        assert [s.id for s in restored] == [s.id for s in original]
        assert [s.target_output for s in restored] == [s.target_output for s in original]

    def test_split_is_reproducible_for_a_fixed_seed(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label", id_field="id")

        first_train, first_test = dataset.split(train_ratio=0.67, seed=7)
        second_train, second_test = dataset.split(train_ratio=0.67, seed=7)

        assert [s.id for s in first_train] == [s.id for s in second_train]
        assert [s.id for s in first_test] == [s.id for s in second_test]
        assert len(first_train) + len(first_test) == len(dataset)

    def test_statistics_report_sample_count(self, json_dataset_file: Path) -> None:
        dataset = JSONDataset(name="reviews")
        dataset.load_data(json_dataset_file, text_field="text", label_field="label")
        stats = dataset.get_statistics()
        assert stats["total_samples"] == 3
        assert stats["max_input_length"] >= stats["min_input_length"]
