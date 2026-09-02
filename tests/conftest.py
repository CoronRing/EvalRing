"""Shared fixtures.

Every test in this suite runs offline. Nothing here contacts a model provider;
agents under test are the deterministic in-process ones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from EvalRing.utils.global_cache import GlobalCache

#: Credential variables that must not leak from the developer's shell into a
#: test run, so behaviour does not depend on who is running the suite.
_CREDENTIAL_VARS = (
    "EVALRING_API_KEY",
    "EVALRING_BASE_URL",
    "EVALRING_MODEL",
    "EVALRING_WORKSPACE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "OPEN_ROUTER_KEY",
    "OPEN_ROUTER_MODEL",
    "RADIUM_API_KEY",
    "RADIUM_BASE_URL",
    "RADIUM_MODEL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials from the environment for every test."""
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point cache and run artifacts at a temporary directory.

    The cache is a process-wide singleton, so it is reset on both sides of the
    test to stop one test's database from leaking into the next.
    """
    GlobalCache.reset_instance()
    monkeypatch.setenv("EVALRING_WORKSPACE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    GlobalCache.reset_instance()


@pytest.fixture
def sample_records() -> list[dict[str, object]]:
    """Three labelled records used across the dataset and evaluator tests."""
    return [
        {"id": "a1", "text": "this product is wonderful", "label": "positive", "source": "web"},
        {"id": "a2", "text": "terrible, i want a refund", "label": "negative", "source": "web"},
        {"id": "a3", "text": "it arrived on tuesday", "label": "neutral", "source": "email"},
    ]


@pytest.fixture
def json_dataset_file(tmp_path: Path, sample_records: list[dict[str, object]]) -> Path:
    """Write ``sample_records`` to a JSON file and return its path."""
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(sample_records), encoding="utf-8")
    return path


@pytest.fixture
def csv_dataset_file(tmp_path: Path, sample_records: list[dict[str, object]]) -> Path:
    """Write ``sample_records`` to a CSV file and return its path."""
    import pandas as pd

    path = tmp_path / "dataset.csv"
    pd.DataFrame(sample_records).to_csv(path, index=False)
    return path
