"""The ``evalring`` console script.

Only offline subcommands are exercised; ``run-suite`` and ``models`` reach the
network and are covered by their own modules' unit tests instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from EvalRing import __version__
from EvalRing.cli import build_parser, main


class TestParser:
    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_every_subcommand_is_registered(self) -> None:
        parser = build_parser()
        for command in ("info", "check", "models", "run-suite"):
            assert parser.parse_args([command, *_required_args(command)]).command == command

    def test_version_flag_exits_cleanly(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


def _required_args(command: str) -> list:
    if command == "info":
        return ["--dataset", "x.json"]
    if command == "run-suite":
        return ["--eval-script", "s.py", "--models-file", "m.json"]
    return []


class TestInfoCommand:
    def test_reports_statistics_as_json(
        self, json_dataset_file: Path, capsys: pytest.CaptureFixture
    ) -> None:
        exit_code = main(["info", "--dataset", str(json_dataset_file), "--id-field", "id"])
        assert exit_code == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["total_samples"] == 3
        assert payload["valid"] is True

    def test_reads_csv_as_well_as_json(
        self, csv_dataset_file: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["info", "--dataset", str(csv_dataset_file)]) == 0
        assert json.loads(capsys.readouterr().out)["total_samples"] == 3

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        assert main(["info", "--dataset", str(tmp_path / "absent.json")]) == 1

    def test_unsupported_extension_exits_nonzero(self, tmp_path: Path) -> None:
        path = tmp_path / "data.parquet"
        path.write_bytes(b"")
        assert main(["info", "--dataset", str(path)]) == 1


class TestCheckCommand:
    def test_reports_missing_credentials_with_nonzero_exit(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        exit_code = main(["check"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 1
        assert payload["api_key_found"] is False
        assert "EVALRING_API_KEY" in payload["recognized_key_variables"]

    def test_reports_a_resolved_key_without_printing_it(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("EVALRING_API_KEY", "sk-super-secret-value")

        exit_code = main(["check"])
        out = capsys.readouterr().out
        payload = json.loads(out)

        assert exit_code == 0
        assert payload["api_key_found"] is True
        assert payload["api_key_source"] == "$EVALRING_API_KEY"
        # The key itself must never reach the terminal.
        assert "sk-super-secret-value" not in out

    def test_reports_the_resolved_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("EVALRING_API_KEY", "k")
        monkeypatch.setenv("EVALRING_BASE_URL", "https://gateway.example/v1")

        main(["check"])
        assert json.loads(capsys.readouterr().out)["base_url"] == "https://gateway.example/v1"
