# Suicide Detection Sample Guide

This document is a complete usage guide for the suicide detection sample under EvalRing.

It covers:

- environment setup
- single-run evaluation
- all three prediction modes
- retry-in-place workflow
- model-suite workflow
- output artifacts
- adaptation guidance for future tasks

## 1. Scope and Files

Core package and sample files:

- [src/EvalRing](../../../src/EvalRing)
- [examples/suicide_detection/llm_agent.py](../../../examples/suicide_detection/llm_agent.py)
- [examples/suicide_detection/evaluate_rsd15k_main.py](../../../examples/suicide_detection/evaluate_rsd15k_main.py)
- [examples/suicide_detection/run_model_suite.py](../../../examples/suicide_detection/run_model_suite.py)
- [examples/suicide_detection/data/data_translator.py](../../../examples/suicide_detection/data/data_translator.py)
- [examples/suicide_detection/data/rsd_15k.csv](../../../examples/suicide_detection/data/rsd_15k.csv)

One can access original dataset from: https://drive.google.com/file/d/1DrWVF28hEj70x3Yxtfonk3pzS0hPOQBT/view?usp=sharing

Alternative datasets: https://drive.google.com/drive/folders/1c8wL7woT-td9SFVr78wqiIYMCjXHRsoh?usp=drive_link

Reusable classification helpers used by this sample:

- [src/EvalRing/agent/classification.py](../../../src/EvalRing/agent/classification.py)

## 2. Dataset Translator

Translator file:

- [examples/suicide_detection/data/data_translator.py](../../../examples/suicide_detection/data/data_translator.py)

Purpose:

- keep original dataset unchanged
- create `*_simple.csv` with only users that appear once
- create `*_multi_round.csv` with only users that appear more than once

Default command:

```bash
python examples/suicide_detection/data/data_translator.py
```

Dry-run command (no file writing):

```bash
python examples/suicide_detection/data/data_translator.py --dry-run
```

With custom source path:

```bash
python examples/suicide_detection/data/data_translator.py --input-csv examples/suicide_detection/data/rsd_15k.csv
```

Default generated files:

- `examples/suicide_detection/data/rsd_15k_simple.csv`
- `examples/suicide_detection/data/rsd_15k_multi_round.csv`

## 3. Environment Setup

From repository root:

```bash
Set-Location "EvalRing"
```

Activate virtual environment (example path):

```bash
source .venv/bin/activate   # Windows: .venv\Scriptsctivate
```

Required environment key (at least one):

- `OPEN_ROUTER_KEY`
- `OPENAI_API_KEY`

Optional model override keys:

- `OPEN_ROUTER_MODEL`
- `OPENAI_MODEL`

The runner will prefer OpenRouter when `OPEN_ROUTER_KEY` is available.

## 4. Standard Evaluation Command

Main runner:

- [examples/suicide_detection/evaluate_rsd15k_main.py](../../../examples/suicide_detection/evaluate_rsd15k_main.py)

Basic command:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

Supported arguments:

- `--n-samples`
- `--max-workers`
- `--max-retries`
- `--seed`
- `--agent-mode`
- `--base-class`
- `--retry-failed`
- `--meta-path`
- `--out-dir` (Groups output run directories into a specific parent folder)
- `--continue` (Resumes a run if it failed midway, skipping already-completed items)
- `--host-model` (Specifies the host model for `multi-agent-host` mode)
- `--role-models-json` (Provides JSON mapping of role names to model endpoints for `multi-agent-host` mode)
- `--max-host-iterations` (Caps the number of rounds for multi-agent mode, default 10)

## 5. Agent Modes

### 4.1 single-class

Model returns one label.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode single-class
```

### 4.2 multi-class-chance

Model returns one JSON object with probabilities for all four classes.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-class-chance
```

Expected JSON style:

```json
{"ideation": 0.80, "behavior": 0.10, "indicator": 0.05, "attempt": 0.05}
```

### 4.3 base-vs-rest-binary

One class is declared as base, and the agent runs three binary decisions: `base vs each non-base class`.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Indicator
```

Other base-class examples:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Attempt
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Behavior
```

Behind the scenes:

1. agent runs one binary call per non-base class
2. pairwise probabilities are collected
3. core aggregation converts pairwise probabilities into one multi-class probability distribution
4. evaluator resolves top class for standard metrics

### 4.4 multi-agent-host

A more complex architecture where a "host" agent negotiates with specialized sub-agents to reach a consensus label.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-agent-host --host-model "openai/gpt-4o" --role-models-json '{"expert":"anthropic/claude-3.5-sonnet", "critic":"google/gemini-1.5-pro"}' --max-host-iterations 5
```

Key Multi-Agent Arguments:
* `--host-model`: Specifies which model drives the main consensus loop.
* `--role-models-json`: Passes a JSON string defining the roles and the sub-models populating those roles. 
* `--max-host-iterations`: Sets a limit on how many questioning rounds the host can perform.

## 6. Retry Failed Rows In Place & Continuing Iterations

### Retry Mode

Retry mode avoids rerunning all rows when only a few API/provider calls failed. It explicitly searches for `Error` results from a previous iteration and re-evaluates them using patched local memory instead of rewriting to a new directory base.

Retry latest run:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

Retry specific run:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed --meta-path examples/suicide_detection/_EvalRing/run_YYYYMMDD_HHMMSS/Meta.json
```

Behavior summary:

1. load previous `Meta.json`
2. identify rows with `Error` prediction or non-empty `error`
3. rerun only those rows
4. patch the original run folder files in place

### Continue / Resume mode

To resume a partial run from operations that stopped prematurely before conclusion or completion, `--continue` skips rows functionally logged as valid without executing them again, appending to the last output trace logs instead of starting a fresh cycle.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --continue
```

## 7. Multi-Model Suite

Runner:

- [examples/suicide_detection/run_model_suite.py](../../../examples/suicide_detection/run_model_suite.py)

Default script behavior is controlled in that file's `__main__` block invoking the `run_suite(...)` call.

Programmatic examples include parameters corresponding to evaluating suites across massive parallel chunks and complex multi-agent workflows across the target models:

```python
from example.suicide_detection.run_model_suite import run_suite

run_suite(
    n_samples=10000,
    max_workers=1000,
    seed=42,
    agent_mode="multi-agent-host",
    base_class="Indicator",
    host_model="openai/gpt-4o-mini",
    role_models_json='{"expert":"anthropic/claude-3-haiku"}',
    max_host_iterations=10,
    continue_runs=True
)
```

### Suite Folder Isolation

When run, the suite natively isolates its outputs. It groups all individual model runs executed during its sequence into an encapsulated folder formatted as: `_EvalRing/run_suite_YYYYMMDD_HHMMSS`. 

The suite auto-generates two summary files locally inside this target folder linking insights across all model testing iterations:
- **Combined Markdown (`run_suite_{timestamp}.md`)**: A concatenated performance review documenting logic, accuracy, execution errors, F1-scores, and stripped down components extracted dynamically from every sub-model `result.md`.
- **Suite Report JSON (`suite_report_{timestamp}.json`)**: Structured dictionary mapping the global suite configuration metrics (n_samples, workers, base_class) paired with array performance attributes for programmable tracking.

Suite stop conditions:

- subprocess exits non-zero
- run directory not found
- `execution_failures` found in `Meta.json`
- `prediction == Error` or non-empty `error` found in `all_cases.csv`

## 8. Artifacts Produced Per Run

Run directories are created at:

- [examples/suicide_detection/_EvalRing](../../../examples/suicide_detection/_EvalRing)

Each run contains:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt` (if failures exist)
- `Meta.json`

Important fields for structured modes:

- `prediction_confidence`
- `class_scores`
- `model_config.agent_mode`
- `model_config.base_class`

## 9. How Metrics Are Computed With Structured Outputs

The evaluator accepts either:

- plain string labels
- mapping outputs (class to score)

Core resolver in [src/EvalRing/agent/classification.py](../../../src/EvalRing/agent/classification.py) selects the top class for accuracy/precision/recall/F1 while preserving confidence and class scores in per-sample metadata.

## 10. Practical Experiment Recipes

Quick smoke test:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Structured mode sanity:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode multi-class-chance
```

Base-vs-rest sanity:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode base-vs-rest-binary --base-class Indicator
```

Recover transient errors:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

## 11. Adapting This Pattern to Other Tasks

For a new classification task (for example emotion categorization):

1. create a task-specific agent class similar to [examples/suicide_detection/llm_agent.py](../../../examples/suicide_detection/llm_agent.py)
2. define class labels and output prompt format
3. optionally add base-vs-rest decomposition if binary comparisons are preferred
4. return either label strings or class-score mapping outputs
5. reuse [src/EvalRing/evaluator/implementations.py](../../../src/EvalRing/evaluator/implementations.py) without changing metric logic

This keeps the evaluation flow stable while allowing task-specific prompting and parsing.
