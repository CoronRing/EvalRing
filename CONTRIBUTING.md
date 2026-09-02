# Contributing to EvalRing

Thanks for your interest. This document covers how to get set up, what the code
should look like, and what a reviewable change looks like.

## Getting set up

EvalRing requires Python 3.10 or newer.

```bash
git clone https://github.com/CoronRing/EvalRing.git
cd EvalRing

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

`[dev]` pulls in the full runtime (`llm`, `viz`, `datagen`) plus pytest, ruff,
mypy, and the build tooling.

## The checks CI runs

Run these before opening a pull request. They are exactly what
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs.

```bash
ruff check src tests          # lint
ruff format --check src tests # formatting
mypy                          # types
pytest                        # tests
```

`ruff format src tests` applies the formatting rather than just checking it.

### Tests must run offline

The whole suite runs without an API key and without network access. Provider
credentials are stripped from the environment by an autouse fixture in
[`tests/conftest.py`](tests/conftest.py), and cache and run artifacts are
redirected into a temporary directory.

If you add a test that genuinely needs the network, mark it:

```python
@pytest.mark.network
def test_fetches_the_live_model_catalogue() -> None:
    ...
```

Those tests are excluded with `pytest -m "not network"`.

To test code that calls a model, subclass `BaseAgent` with a scripted response
rather than mocking an SDK — see `ScriptedAgent` in
[`tests/test_evaluator.py`](tests/test_evaluator.py).

## Design principles

**Push shared logic down into the framework, not up into each task.** If two
evaluation tasks both need a behaviour, it belongs in `EvalRing/`, not copied
into two scripts under `examples/`. Dataset-level and evaluator-level
abstractions exist so a new task is a thin script, not a fork of the framework.

**Never hard-code a provider.** All credentials and endpoints resolve through
[`EvalRing/config.py`](src/EvalRing/config.py). A new backend adds an entry to
`_CREDENTIAL_SOURCES`; it does not read `os.environ` directly and it never bakes
in a base URL as a default argument.

**The library logs, the entry point prints.** Modules under `EvalRing/` use
`get_logger(__name__)` and never call `print()` or configure the root logger.
Only `EvalRing/cli.py` and scripts under `examples/` call
`configure_logging()`. The one exception is machine-readable output the user
asked for, which goes to stdout via `print()` in the CLI.

**Optional dependencies stay optional.** `import EvalRing` must succeed with
only the core dependencies installed. Import `openai`, `litellm`, `matplotlib`,
and `nest_asyncio` inside the function that needs them, and raise an
`ImportError` naming the extra to install. CI enforces this in the
`minimal-install` job.

## Code style

- Formatting and linting are ruff's job; do not hand-format around it.
- Public functions, classes, and methods need a docstring with `Args:`,
  `Returns:`, and `Raises:` sections where they apply. Docstrings render in
  Pylance, so keep the argument names accurate.
- Annotate public signatures. `EvalRing.config`, `EvalRing.logging_utils`, and
  `EvalRing.cli` are checked under stricter mypy settings; new modules should
  be added to that override list in `pyproject.toml` once they are fully typed.
- Do not reference previous versions of the code in comments. Comments explain
  what the code does now and why, not what it used to be.

## Documentation is part of the change

Docs ship with the code so that a reader — human or agent — can trust them
without reading the source. A change that alters behaviour is not complete
until the affected document is updated in the same pull request:

| If you change | Update |
| --- | --- |
| Architecture, components, data flow | [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md) and bump its version |
| Environment variables, provider resolution | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| The public Python API | [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) |
| CLI flags or subcommands | [`docs/CLI.md`](docs/CLI.md) |
| How runs are operated | [`docs/USAGE.md`](docs/USAGE.md) |
| Anything user-visible | [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased` |

[`AGENTS.md`](AGENTS.md) is the entry point for coding agents working in this
repository; keep it pointing at the right documents.

## Data and secrets

- Never commit an API key. Credentials come from the environment or a local
  `.env`, both of which are gitignored. See [`.env.example`](.env.example).
- Never commit a dataset you do not have redistribution rights to. Datasets
  used by examples are documented in [`docs/DATA.md`](docs/DATA.md) with their
  source and licence; add an entry there before adding data.
- The suicide-detection and clinical examples involve sensitive material. Read
  [`docs/DATA.md`](docs/DATA.md) before working on them.

## Pull requests

1. Branch from `main`.
2. Keep the change focused; unrelated formatting churn makes review harder.
3. Add or update tests covering the behaviour you changed.
4. Update the documents listed above.
5. Make sure the four checks pass locally.

Commits do not need to follow a strict convention, but a subject line that
names the behaviour that changed helps the changelog.

## Reporting bugs and requesting features

Open an issue using the templates under
[`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE). For anything with security
implications, follow [`SECURITY.md`](SECURITY.md) instead of opening a public
issue.
