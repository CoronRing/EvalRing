# Working in this repository

Orientation for coding agents and for anyone reading the codebase for the first
time. The documents under [`docs/`](docs/) are the authoritative description of
this system and ship with the code — read them before the source, and update
them in the same change that alters behaviour.

## Where to read first

| Question | Document |
| --- | --- |
| What is this and how is it built? | [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md) |
| What can I call from Python? | [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) |
| How do I point it at a model? | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| What does the CLI do? | [`docs/CLI.md`](docs/CLI.md) |
| How do I run a real evaluation? | [`docs/USAGE.md`](docs/USAGE.md) |
| Which datasets exist and what may I do with them? | [`docs/DATA.md`](docs/DATA.md) |
| How do I contribute a change? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| How do I cut a release? | [`docs/RELEASING.md`](docs/RELEASING.md) |

If a document disagrees with the code, the code is what runs — fix the document
and say so in your summary.

## Layout

```
src/EvalRing/        the installable package; the only thing that ships
  config.py          credential and model resolution for every component
  logging_utils.py   logger namespace and entry-point logging setup
  cli.py             the `evalring` console script
  dataset/           BaseDataset, DataSample, JSON/CSV/DataFrame readers
  agent/             BaseAgent, AgentResponse, Mock/RuleBased/OpenAI agents,
                     classification parsing, error classification
  evaluator/         BaseEvaluator, ClassificationEvaluator, llm_judge/
  utils/             suite runner, response cache, model lists, charts
tests/               offline test suite
examples/            runnable evaluations; NOT packaged, NOT importable
docs/                authoritative documentation
```

## Ground rules

These are enforced by CI, so a change that breaks one will fail before review.

1. **`import EvalRing` must work with core dependencies only.** `openai`,
   `litellm`, `matplotlib`, and `nest_asyncio` are optional. Import them inside
   the function that needs them and raise an `ImportError` naming the extra.
2. **No `print()` inside `src/EvalRing/`**, except machine-readable output the
   user explicitly asked the CLI for. Use `get_logger(__name__)`.
3. **No provider hard-coded anywhere.** Credentials and base URLs come from
   `EvalRing.config.resolve_credentials()`. Never read a key from `os.environ`
   directly, and never put an endpoint in a default argument.
4. **Tests stay offline.** No network in the default suite. Use a scripted
   `BaseAgent` subclass, not SDK mocks.
5. **Nothing but `EvalRing` goes in the wheel.** `examples/` and `tests/` are
   excluded; CI asserts the wheel's top-level contents.
6. **Do not commit datasets or credentials.** See
   [`docs/DATA.md`](docs/DATA.md) and [`SECURITY.md`](SECURITY.md).

## Local commands

```bash
pip install -e ".[dev]"

ruff check src tests
ruff format src tests
mypy
pytest
```

Use `.agent_temp/` for scratch files and `.log/` for log output. Both are
gitignored. Do not leave working files in the repository root.

## Sensitive material

`examples/suicide_detection/` and `examples/sample/med_note/` involve
suicide-risk classification and clinical notes. Read
[`docs/DATA.md`](docs/DATA.md) before touching them: the datasets carry
redistribution conditions, and the outputs are not clinical tools.
