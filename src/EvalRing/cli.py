"""
Command-line interface for EvalRing.

Installed as the ``evalring`` console script. Every subcommand is a thin
wrapper over the public Python API, so anything reachable here is also
reachable from code.

Run ``evalring --help`` for the full listing, or see ``docs/CLI.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import CREDENTIAL_ENV_VARS, resolve_credentials, resolve_model_name
from .dataset.base import BaseDataset
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

SUPPORTED_DATASET_SUFFIXES = (".json", ".csv")


def _load_dataset(
    path: Path,
    text_field: str,
    label_field: str,
    id_field: str | None,
) -> BaseDataset:
    """Load a dataset from disk, picking the reader from the file extension.

    Args:
        path: Path to a ``.json`` or ``.csv`` dataset file.
        text_field: Column or key holding the model input.
        label_field: Column or key holding the ground-truth label.
        id_field: Column or key holding a unique sample ID, or ``None`` to
            fall back to the row index.

    Returns:
        A loaded :class:`~EvalRing.dataset.base.BaseDataset` subclass instance.

    Raises:
        ValueError: If the file extension has no registered reader.
    """
    from .dataset import CSVDataset, JSONDataset

    suffix = path.suffix.lower()
    dataset: BaseDataset
    if suffix == ".json":
        dataset = JSONDataset(name=path.stem)
    elif suffix == ".csv":
        dataset = CSVDataset(name=path.stem)
    else:
        raise ValueError(
            f"Unsupported dataset extension {suffix!r}. "
            f"Supported: {', '.join(SUPPORTED_DATASET_SUFFIXES)}"
        )

    dataset.load_data(
        path,
        text_field=text_field,
        label_field=label_field,
        id_field=id_field,
    )
    return dataset


def _cmd_info(args: argparse.Namespace) -> int:
    """Print dataset statistics as JSON.

    Args:
        args: Parsed arguments for the ``info`` subcommand.

    Returns:
        Process exit code: 0 on success, 1 when the dataset cannot be read.
    """
    path = Path(args.dataset)
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        return 1

    try:
        dataset = _load_dataset(path, args.text_field, args.label_field, args.id_field)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return 1

    stats = dataset.get_statistics()
    stats["valid"] = dataset.validate_data()
    stats["path"] = str(path)
    print(json.dumps(stats, indent=2, default=str))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Report the resolved provider configuration without revealing secrets.

    Args:
        args: Parsed arguments for the ``check`` subcommand.

    Returns:
        Process exit code: 0 when a credential was resolved, 1 otherwise.
    """
    credentials = resolve_credentials()
    report = {
        "evalring_version": __version__,
        "python": sys.version.split()[0],
        "api_key_found": bool(credentials.api_key),
        "api_key_source": credentials.source,
        "provider": credentials.provider,
        "base_url": credentials.base_url or "<provider default>",
        "model": resolve_model_name(default="<unset>"),
        "recognized_key_variables": list(CREDENTIAL_ENV_VARS),
    }

    # find_spec only inspects the import system, so this stays fast even when
    # a heavyweight optional package such as litellm is installed.
    report["optional_packages"] = {
        module: importlib.util.find_spec(module) is not None
        for module in ("litellm", "openai", "matplotlib", "nest_asyncio")
    }

    print(json.dumps(report, indent=2))
    if not credentials.api_key:
        logger.warning("No API key resolved. Set one of: %s", ", ".join(CREDENTIAL_ENV_VARS))
        return 1
    return 0


def _cmd_run_suite(args: argparse.Namespace) -> int:
    """Run an evaluation script across every model in a model list.

    Args:
        args: Parsed arguments for the ``run-suite`` subcommand.

    Returns:
        Process exit code propagated from
        :func:`EvalRing.utils.suite_runner.run_suite`.
    """
    from .utils.suite_runner import run_suite

    return run_suite(
        eval_script=args.eval_script,
        model_list_path=args.models_file,
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
        host_model=args.host_model,
        role_models_json=args.role_models_json,
        max_host_iterations=args.max_host_iterations,
        continue_runs=args.continue_runs,
        cache=args.cache,
        cache_mode=args.cache_mode,
        ignore_errors=args.ignore_errors,
        out_dir=args.out_dir,
        data_path=args.data_path,
        no_confirm=args.yes,
    )


def _cmd_models(args: argparse.Namespace) -> int:
    """Generate a model-list JSON file from the OpenRouter catalogue.

    Args:
        args: Parsed arguments for the ``models`` subcommand.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    from .utils.generate_model_list import generate_model_list

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_model_list(output)
    except Exception as exc:
        logger.error("Failed to generate model list: %s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        The fully configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="evalring",
        description="Evaluate agents and models across providers.",
    )
    parser.add_argument("--version", action="version", version=f"evalring {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log warnings and errors.")

    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Show statistics for a dataset file.")
    info.add_argument("--dataset", required=True, help="Path to a .json or .csv dataset.")
    info.add_argument("--text-field", default="text", help="Input text column/key.")
    info.add_argument("--label-field", default="label", help="Ground-truth column/key.")
    info.add_argument("--id-field", default=None, help="Unique sample ID column/key.")
    info.set_defaults(func=_cmd_info)

    check = sub.add_parser(
        "check",
        help="Report the resolved provider configuration (no secrets printed).",
    )
    check.set_defaults(func=_cmd_check)

    models = sub.add_parser("models", help="Generate a model-list JSON file.")
    models.add_argument(
        "--output",
        default="model_list.json",
        help="Destination path for the model list.",
    )
    models.set_defaults(func=_cmd_models)

    suite = sub.add_parser(
        "run-suite",
        help="Run an evaluation script across every model in a model list.",
    )
    suite.add_argument("--eval-script", required=True, help="Path to the evaluation script.")
    suite.add_argument("--models-file", required=True, help="Path to the model list JSON.")
    suite.add_argument("--n-samples", type=int, default=10)
    suite.add_argument("--max-workers", type=int, default=5)
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--agent-mode", default="single-class")
    suite.add_argument("--base-class", default=None)
    suite.add_argument("--host-model", default=None)
    suite.add_argument("--role-models-json", default=None)
    suite.add_argument("--max-host-iterations", type=int, default=10)
    suite.add_argument("--continue-runs", action="store_true")
    suite.add_argument("--cache", default=None)
    suite.add_argument(
        "--cache-mode",
        default="both",
        choices=["runs_only", "cache_file", "both", "none"],
    )
    suite.add_argument("--ignore-errors", action="store_true")
    suite.add_argument("--out-dir", default=None)
    suite.add_argument("--data-path", default=None)
    suite.add_argument(
        "-y", "--yes", action="store_true", help="Skip the pre-run cache confirmation."
    )
    suite.set_defaults(func=_cmd_run_suite)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``evalring`` console script.

    Args:
        argv: Argument vector to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    configure_logging(level, verbose=args.verbose)

    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
