# EvalRing

[![CI](https://github.com/CoronRing/EvalRing/actions/workflows/ci.yml/badge.svg)](https://github.com/CoronRing/EvalRing/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A framework for evaluating agents across models and across versions of the same
agent, with reproducible artifacts and resumable runs.

EvalRing exists for the case where you have one task and many candidates: six
models, three prompt variants, two agent architectures — and you need a
comparison you can rerun next month and still trust. It gives you a common
interface for datasets, agents, and scoring; a response cache so a rerun costs
almost nothing; and per-run artifacts that record exactly what was asked and
what came back.

## What it does

- **One interface, many backends.** Any OpenAI-compatible endpoint works —
  OpenAI, OpenRouter, a self-hosted gateway, vLLM, Ollama — through one set of
  environment variables. Nothing is hard-coded to a vendor.
- **Suites over one-offs.** Run the same evaluation across a list of models in
  parallel subprocesses, and get a combined report plus per-model artifacts.
- **Runs you can resume.** Every response is cached by model, prompt, and
  parameters. Interrupt a run, fix a bug, rerun: only the uncached samples cost
  money. Failed samples can be retried in place from a previous run's metadata.
- **Classification and LLM-as-a-judge.** Accuracy, macro precision/recall/F1,
  plus latency and token accounting; or rubric-based scoring with a judge model
  when there is no single right answer.
- **Artifacts, not just numbers.** Each run writes per-sample CSV/JSONL, a
  Markdown report, and a `Meta.json` recording the configuration that produced
  it.

## Install

```bash
pip install evalring          # core: datasets, evaluation, caching
pip install "evalring[llm]"   # + call models (litellm, openai, httpx)
pip install "evalring[all]"   # + charts and synthetic data generation
```

Python 3.10 or newer. From source:

```bash
git clone https://github.com/CoronRing/EvalRing.git
cd EvalRing
pip install -e ".[dev]"
```

## Configure a provider

One variable is enough. `EVALRING_API_KEY` is the vendor-neutral form; set
`EVALRING_BASE_URL` too if you are not using OpenAI directly.

```bash
export EVALRING_API_KEY="your-key"
export EVALRING_BASE_URL="https://openrouter.ai/api/v1"   # optional
export EVALRING_MODEL="anthropic/claude-sonnet-4"         # optional default
```

`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `OPEN_ROUTER_KEY` are also
recognized. Check what EvalRing resolved — it never prints the key itself:

```console
$ evalring check
{
  "evalring_version": "0.2.0",
  "api_key_found": true,
  "api_key_source": "$EVALRING_API_KEY",
  "provider": "evalring",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4",
  ...
}
```

Full precedence table: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Quick start

This runs offline — no API key, no network:

```python
from EvalRing import ClassificationEvaluator, JSONDataset, RuleBasedAgent

# 1. Load a dataset. Construct, then load; `id_field` matters for resuming.
dataset = JSONDataset(name="reviews")
dataset.load_data("reviews.json", text_field="text", label_field="label", id_field="id")

# 2. Pick or write an agent.
agent = RuleBasedAgent(
    name="keyword-baseline",
    rules={"positive": ["great", "love"], "negative": ["awful", "terrible"]},
    default_output="neutral",
)

# 3. Evaluate.
evaluator = ClassificationEvaluator(output_dir="results")
result = evaluator.evaluate(agent, dataset, task_name="sentiment")

print(f"accuracy: {result.metrics.get_metric('accuracy'):.3f}")
print(f"f1:       {result.metrics.get_metric('f1_score'):.3f}")
evaluator.save_results(result)
```

`reviews.json` is a list of objects:

```json
[
  {"id": "1", "text": "this is great, I love it", "label": "positive"},
  {"id": "2", "text": "awful experience, terrible", "label": "negative"}
]
```

CSV and pandas work the same way through `CSVDataset` and `DataFrameDataset`.

## Evaluating a real model

```python
from EvalRing import ClassificationEvaluator, CSVDataset, OpenAIAgent

dataset = CSVDataset(name="reviews")
dataset.load_data("reviews.csv", text_field="review", label_field="sentiment", id_field="row_id")

agent = OpenAIAgent(
    name="sentiment-classifier",
    model_name="gpt-4o",                  # or leave unset to use $EVALRING_MODEL
    system_prompt="Reply with exactly one word: positive, negative, or neutral.",
    temperature=0.0,
)

result = ClassificationEvaluator(output_dir="results").evaluate(
    agent, dataset, task_name="sentiment", max_workers=16, max_retries=3
)
```

Rate limits, transient 5xx, and stalled streams are retried with backoff;
samples that never succeed are recorded as failures rather than aborting the
run, and can be retried later with `retry_failed_cases()`.

## Writing your own agent

Subclass `BaseAgent` and implement two methods. Anything you can call from
Python can be evaluated — a local model, a multi-step pipeline, a REST service.

```python
from EvalRing import AgentResponse, BaseAgent

class MyAgent(BaseAgent):
    def initialize(self, **kwargs) -> None:
        self._model = load_my_model()
        self._is_initialized = True

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        label, score = self._model(input_text)
        return AgentResponse(
            input_id="",              # the evaluator fills this in
            input_text=input_text,
            output=label,
            confidence=score,
        )
```

`output` may be a label string or a `{label: score}` mapping; EvalRing resolves
the top class either way and keeps the full distribution in the per-sample
metrics.

## Comparing models

Write your evaluation as a script that accepts the standard flags, list the
models in a JSON file, and run the suite:

```bash
evalring models --output model_list.json
evalring run-suite \
    --eval-script my_eval.py \
    --models-file model_list.json \
    --n-samples 500 \
    --max-workers 32 \
    --yes
```

Each model runs in its own subprocess, so one model failing does not stop the
others. Output lands in `_EvalRing/run_suite_<timestamp>/`: per-model run
directories, a combined Markdown report, a JSON summary, and charts if
`matplotlib` is installed. See [docs/USAGE.md](docs/USAGE.md).

## LLM-as-a-judge

For open-ended outputs with no single correct answer:

```python
from EvalRing import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator.from_rubric(
    rubric="Score 1-5 on whether the response answers the question accurately.",
    criteria="Factual accuracy relative to the reference answer",
    judge_model="gpt-4o",
    threshold=0.6,
)
result = evaluator.evaluate(agent, dataset, task_name="qa-quality")
```

Rubrics can be plain strings or structured `Rubric` objects with per-level
descriptions, and you can score several weighted criteria at once.

## Command line

```
evalring check                       Report the resolved provider config
evalring info --dataset data.csv     Dataset statistics and validation
evalring models --output list.json   Generate a model list
evalring run-suite ...               Run an evaluation across many models
```

See [docs/CLI.md](docs/CLI.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/DESIGN_SPEC.md](docs/DESIGN_SPEC.md) | Architecture, components, data flow, persistence |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Every public class and function |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Environment variables and provider setup |
| [docs/CLI.md](docs/CLI.md) | Command-line reference |
| [docs/USAGE.md](docs/USAGE.md) | Running suites, artifacts, retries |
| [docs/DATA.md](docs/DATA.md) | Dataset provenance, licensing, responsible use |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and standards |
| [AGENTS.md](AGENTS.md) | Orientation for coding agents |

## Examples

Runnable evaluations live in [`examples/`](examples/). They are not part of the
installed package; clone the repository to use them.

- [`examples/hle/`](examples/hle/) — Humanity's Last Exam, ARC-Challenge, and
  GPQA with a judge-based answer checker
- [`examples/suicide_detection/`](examples/suicide_detection/) — multi-class
  risk classification, including one-vs-rest and multi-role agent modes
- [`examples/semantic_analysis/`](examples/semantic_analysis/) — a minimal
  custom agent and dataset

The suicide-detection and clinical-note examples involve sensitive material and
datasets with redistribution conditions. Read [docs/DATA.md](docs/DATA.md)
before using them. EvalRing is a research and engineering tool; none of its
outputs are clinical or diagnostic instruments.

## Status

Pre-1.0. The core abstractions (`BaseDataset`, `BaseAgent`, `BaseEvaluator`)
are stable in shape, but minor releases may still change signatures. Pin a
version if you depend on it. Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE).
