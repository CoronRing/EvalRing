# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-01

First release prepared for public use. The framework's behaviour is largely
unchanged; this release makes the package honest about what it does, removes
vendor-specific defaults, and adds the tests and CI a published library needs.

### Added

- `evalring` command-line interface with four subcommands: `info` (dataset
  statistics), `check` (report the resolved provider configuration without
  printing secrets), `models` (generate a model list), and `run-suite`
  (multi-model evaluation). Previously the entry point was declared but the
  module did not exist.
- `EvalRing.config` with `resolve_credentials()` and `resolve_model_name()`, a
  single documented precedence order for API keys, endpoints, and model names
  across every component.
- `EvalRing.logging_utils` with `get_logger()` and `configure_logging()`.
- `py.typed` marker, so downstream projects get type information.
- Test suite covering credentials, datasets, agents, evaluation, metrics,
  caching, and the CLI. Every test runs offline.
- CI across Python 3.10-3.13 on Linux, plus Windows and macOS: lint, format,
  types, tests, a core-only install check, and a wheel-contents assertion.
- Release workflow publishing to PyPI via Trusted Publishing on a version tag.
- `GlobalCache.reset_instance()` and the `EVALRING_WORKSPACE` variable, so the
  cache location can be controlled instead of always following the working
  directory.
- Documentation: `CONFIGURATION.md`, `API_REFERENCE.md`, `CLI.md`, `DATA.md`,
  `RELEASING.md`, plus `AGENTS.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and
  `.env.example`.

### Changed

- **Provider resolution is vendor-neutral.** `EVALRING_API_KEY` and
  `EVALRING_BASE_URL` are the recommended variables. `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`, `OPEN_ROUTER_KEY`, and `RADIUM_API_KEY` still work, in
  that order. No private endpoint is used as a default any more; a provider's
  base URL applies only when that provider's key is the one selected.
- **The library logs instead of printing.** Modules under `EvalRing/` emit
  records on the `EvalRing` logger hierarchy and no longer write to stdout or
  configure the root logger. Entry points call `configure_logging()`.
- Examples moved from `src/example/` to `examples/` and are no longer packaged.
  Installing EvalRing previously also installed a top-level `example` package.
- Dependencies now reflect what the code imports. Core: `pandas`, `tqdm`,
  `python-dotenv`, `requests`. Optional extras: `llm` (litellm, openai, httpx),
  `viz` (matplotlib), `datagen` (openai, nest-asyncio), `all`, `dev`.
- Minimum Python is 3.10. The previously declared 3.8 floor was never valid:
  `dataset/base.py` uses PEP 604 annotations evaluated at runtime.
- `BaseDataset.load_data()` declares `text_field`, `label_field`, and
  `id_field` so concrete readers no longer widen the base signature.
- `run_suite()` takes `base_class=None` by default and forwards
  `--base-class` only when set, instead of defaulting to a task-specific label.
- `run_suite()` skips its interactive confirmation when stdin is not a TTY,
  so automated runs cannot hang.
- Package metadata: distribution renamed to `evalring` (the import name stays
  `EvalRing`), real repository URLs, single version source in
  `EvalRing.__version__`.

### Removed

- `setup.py` and `requirements.txt`. `pyproject.toml` is the single source of
  build configuration and dependencies.
- `EvalRing.examples`, a module that could not be imported: it referenced
  `EvalRing.base` and `EvalRing.implementations`, neither of which exists.
- The `sys.path` manipulation that `EvalRing.utils.suite_runner` performed at
  import time, and the equivalent bootstrapping in the example scripts. Install
  the package instead: `pip install -e .`.
- Unused declared dependencies: `numpy`, `scikit-learn`, `seaborn`, `pyyaml`.
  The wrong `dotenv` distribution was replaced with `python-dotenv`.

### Fixed

- The README documented an API that did not exist (`Evaluator`,
  `ClassificationMetrics`, `load_dataset`, `create_agent`, `EvalRing.agents`,
  `EvalRing.datasets`, and a CLI). It now documents the real API, verified by a
  test.
- A hardcoded absolute developer path in the suicide-detection suite runner.
- A duplicated `--ignore-errors` flag in the suite runner's subprocess command.
- `zip()` over ground truth and predictions is now strict, so a length mismatch
  raises instead of silently truncating the metric computation.
- Trailing-whitespace, unused-variable, and un-narrowed-type issues across the
  package; `ruff check`, `ruff format --check`, and `mypy` all pass.

### Security

- `.gitignore` no longer relies on a blanket `*.csv` rule that silently hid
  files; credential files used by examples are ignored by explicit path.
- `evalring check` reports which variable supplied the API key without printing
  the key, and a test asserts the key never reaches stdout.

[Unreleased]: https://github.com/CoronRing/EvalRing/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/CoronRing/EvalRing/releases/tag/v0.2.0
