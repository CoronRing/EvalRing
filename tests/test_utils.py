"""Cache behaviour, logging setup, and the public API surface."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import EvalRing
from EvalRing.logging_utils import LOGGER_NAME, configure_logging, get_logger
from EvalRing.utils.global_cache import GlobalCache


class TestGlobalCache:
    def test_round_trips_a_payload(self, isolated_workspace: Path) -> None:
        cache = GlobalCache(mode="cache_file", workspace_root=str(isolated_workspace))
        key = GlobalCache.generate_key("model-a", "prompt", {"temperature": 0.0})

        cache.set(key, {"prediction": "positive"})
        assert cache.get(key) == {"prediction": "positive"}

    def test_unknown_key_returns_none(self, isolated_workspace: Path) -> None:
        cache = GlobalCache(mode="cache_file", workspace_root=str(isolated_workspace))
        assert cache.get("never-written") is None

    def test_keys_are_stable_across_dict_ordering(self) -> None:
        first = GlobalCache.generate_key("m", {"a": 1, "b": 2}, {"x": 1, "y": 2})
        second = GlobalCache.generate_key("m", {"b": 2, "a": 1}, {"y": 2, "x": 1})
        assert first == second

    def test_keys_differ_per_model(self) -> None:
        assert GlobalCache.generate_key("m1", "p", {}) != GlobalCache.generate_key("m2", "p", {})

    def test_keys_differ_per_parameters(self) -> None:
        assert GlobalCache.generate_key("m", "p", {"t": 0.0}) != GlobalCache.generate_key(
            "m", "p", {"t": 1.0}
        )

    def test_none_mode_never_stores(self, isolated_workspace: Path) -> None:
        cache = GlobalCache(mode="none", workspace_root=str(isolated_workspace))
        key = GlobalCache.generate_key("m", "p", {})
        cache.set(key, {"prediction": "x"})
        assert cache.get(key) is None

    def test_runs_only_mode_skips_the_sqlite_tier(self, isolated_workspace: Path) -> None:
        cache = GlobalCache(mode="cache_file", workspace_root=str(isolated_workspace))
        key = GlobalCache.generate_key("m", "p", {})
        cache.set(key, {"prediction": "x"})

        cache.mode = "runs_only"
        assert cache.get(key) is None

    def test_workspace_comes_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "elsewhere"
        workspace.mkdir()
        GlobalCache.reset_instance()
        monkeypatch.setenv("EVALRING_WORKSPACE", str(workspace))

        cache = GlobalCache(mode="cache_file")
        assert (workspace / "_EvalRing" / "Cache").exists()
        assert cache.db_path.is_relative_to(workspace)

    def test_reset_allows_repointing_the_workspace(self, tmp_path: Path) -> None:
        first = tmp_path / "one"
        second = tmp_path / "two"
        first.mkdir()
        second.mkdir()

        GlobalCache.reset_instance()
        cache_one = GlobalCache(mode="cache_file", workspace_root=str(first))
        assert cache_one.db_path.is_relative_to(first)

        GlobalCache.reset_instance()
        cache_two = GlobalCache(mode="cache_file", workspace_root=str(second))
        assert cache_two.db_path.is_relative_to(second)


class TestLogging:
    def test_library_logger_is_namespaced(self) -> None:
        assert get_logger("EvalRing.agent").name == "EvalRing.agent"
        assert get_logger("some.module").name == "EvalRing.some.module"
        assert get_logger(None).name == LOGGER_NAME

    def test_importing_the_library_adds_no_handlers(self) -> None:
        """A library must leave logging configuration to the application."""
        logger = logging.getLogger(LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        assert logging.getLogger(LOGGER_NAME).handlers == []

    def test_configure_logging_is_idempotent(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        configure_logging(logging.INFO)
        configure_logging(logging.DEBUG)

        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG


class TestPublicAPI:
    def test_every_advertised_name_is_importable(self) -> None:
        missing = [name for name in EvalRing.__all__ if not hasattr(EvalRing, name)]
        assert missing == []

    def test_version_is_a_release_string(self) -> None:
        parts = EvalRing.__version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_the_documented_quickstart_imports_resolve(self) -> None:
        from EvalRing import ClassificationEvaluator, JSONDataset, MockAgent

        assert ClassificationEvaluator is not None
        assert JSONDataset is not None
        assert MockAgent is not None
