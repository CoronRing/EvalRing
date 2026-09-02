# EvalRing Usage Guide

How to operate EvalRing for a real evaluation: running a single model, running
a suite, reading the artifacts, and retrying failures. For the Python API see
[API_REFERENCE.md](API_REFERENCE.md); for the command line see [CLI.md](CLI.md).

## 1. Environment

Install the package first — the example scripts import `EvalRing` and no longer
manipulate `sys.path`:

```bash
pip install -e ".[all]"
```

Configure one provider. `EVALRING_API_KEY` is the vendor-neutral variable;
`EVALRING_BASE_URL` is only needed when the endpoint is not OpenAI:

```bash
export EVALRING_API_KEY="your-key"
export EVALRING_BASE_URL="https://openrouter.ai/api/v1"
export EVALRING_MODEL="anthropic/claude-sonnet-4"
```

`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPEN_ROUTER_KEY`, and `RADIUM_API_KEY`
are also recognized, in that order. Copy [`.env.example`](../.env.example) to
`.env` for a local setup; the example scripts load it, the library does not.

Confirm what was resolved before spending anything — this prints no secrets:

```bash
evalring check
```

The complete precedence tables are in [CONFIGURATION.md](CONFIGURATION.md).

---

## 2. Main Entry Points

### 2.1 Standard evaluation runner

File:

- [examples/suicide_detection/evaluate_rsd15k_main.py](../examples/suicide_detection/evaluate_rsd15k_main.py)

Use for:

- one-model benchmark runs
- report generation
- in-place retry of failed error rows
- switching prediction modes

### 2.2 Sequential model suite runner

File:

- [examples/suicide_detection/run_model_suite.py](../examples/suicide_detection/run_model_suite.py)

Use for:

- testing multiple models one after another
- stopping the sequence when a model run is not error-free

### 2.3 Dataset translator for user-round splits

File:

- [examples/suicide_detection/data/data_translator.py](../examples/suicide_detection/data/data_translator.py)

Use for:

- creating `*_simple.csv` (users appearing exactly once)
- creating `*_multi_round.csv` (users appearing more than once)
- preserving original source dataset unchanged

Command:

```bash
python examples/suicide_detection/data/data_translator.py
```

---

## 3. Prediction Modes

The suicide detection runner supports multiple modes via `--agent-mode`.

### 3.1 `single-class`

- Model returns one label directly.
- Expected labels: `Indicator`, `Ideation`, `Behavior`, `Attempt`.

### 3.2 `multi-class-chance`

- Model returns structured JSON probabilities for all four classes.
- Example:

```json
{"ideation": 0.8, "behavior": 0.1, "indicator": 0.05, "attempt": 0.05}
```

- Evaluator resolves top class automatically for accuracy/F1 metrics.

### 3.3 `base-vs-rest-binary`

- You declare one class as `--base-class`.
- The agent runs a knock-out tournament. It evaluates `base vs target_1`, taking the winner, and evaluates `winner vs target_2`, and so on.
- Output is a sequence of single-word knockout decisions until a final winner is determined.
- Evaluator returns the final tournament surviving class natively.

### 3.4 `per-class-score`

- Model returns independent scores from 1 to 10 for each of the four classes based on a rubric.
- Example:

```json
{"ideation": 8, "behavior": 2, "indicator": 1, "attempt": 1}
```

- Evaluator resolves the top class automatically by picking the highest-scored class.

---

## 4. Standard Evaluation Usage

From the EvalRing root:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 100 --max-workers 10
```

### Supported flags

- `--n-samples`: limit dataset size for the run
- `--max-workers`: number of parallel worker threads
- `--max-retries`: retries per sample inside evaluator execution
- `--seed`: recorded in metadata for reproducibility
- `--agent-mode`: `single-class`, `multi-class-chance`, `base-vs-rest-binary`, `multi-agent-host`, or `per-class-score`
- `--base-class`: used by `base-vs-rest-binary`; one of `Indicator`, `Ideation`, `Behavior`, `Attempt`
- `--retry-failed`: rerun only error cases from an existing run
- `--meta-path`: specific `Meta.json` to use with `--retry-failed`

---

## 4.1 Sample IDs (CRITICAL)

EvalRing assumes every dataset row maps to exactly one evaluation request.

- **Use `ID` (per-row question/message ID) as the dataset `id_field`.**
- **Do NOT use `users` as `id_field`.** `users` is an author identifier and is not unique.

Why this matters:

- Resume logic and report merging key off `sample_id`.
- Non-unique IDs can cause incorrect resume/merge behavior (e.g., skipping remaining rows or overwriting/merging unrelated rows).

RSD_15K in this repo is maintained with an explicit leading `ID` column:

- [examples/suicide_detection/data/rsd_15k.csv](../examples/suicide_detection/data/rsd_15k.csv)

---

## 4.2 Sampling Policy (Deterministic)

When `--n-samples N` is set, the runners take the first `N` rows in file order (equivalent to `head(N)`):

- Sample IDs should be `0..N-1` for RSD_15K
- After sampling, the runner verifies there are no duplicate sample IDs (set size must equal `N`)
- Even with parallel workers, per-sample results are emitted in deterministic order (sorted by `sample_id`) so runs do not *look* randomly sampled.

### Examples

Single-class:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode single-class
```

Multi-class chance:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode multi-class-chance
```

Base-vs-rest with Indicator as base:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Indicator
```

Base-vs-rest with Attempt as base:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 50 --agent-mode base-vs-rest-binary --base-class Attempt
```

---

## 5. Output Files

Each fresh run creates a directory under:

- [examples/suicide_detection/_EvalRing](../examples/suicide_detection/_EvalRing)

Example layout:

```text
run_YYYYMMDD_HHMMSS/
|-- result.md
|-- all_cases.txt
|-- all_cases.csv
|-- failed_cases.txt
`-- Meta.json
```

### 5.1 `result.md`

Human-readable run summary:

- model used
- mode used
- base class (when relevant)
- sample count
- aggregate accuracy / precision / recall / F1
- per-class metrics
- confusion matrix
- per-sample summary table

### 5.2 `all_cases.csv`

Primary machine-readable table for downstream analysis.

Key columns include:

- `sample_id`
- `ground_truth`
- `prediction`
- `correct`
- `prediction_confidence`
- `class_scores`
- `ttft`
- `tps`
- `total_time`
- `generation_time`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `error`
- `text`

### 5.3 `failed_cases.txt`

Convenience text file listing failed predictions that remain after the run.

### 5.4 `Meta.json`

Run manifest and reproducibility record.

Contains:

- `run_config`
- `model_config` (includes `agent_mode`, `base_class`)
- `dataset_config`
- `aggregate_metrics`
- `per_class_metrics`
- `confusion_matrix`
- `retry_history`

---

## 6. Retry Failed Cases In Place

This is the preferred workflow when a small number of samples failed due to provider/API instability.

### What it does

`--retry-failed`:

1. loads an existing run's `Meta.json`
2. reuses that run's configuration
3. reads original `all_cases.csv`
4. selects only rows where:
   - prediction is `Error`, or
   - `error` is non-empty
5. reruns only those rows
6. updates the original run directory in place

### Retry latest run

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

### Retry a specific run

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed --meta-path examples/suicide_detection/_EvalRing/run_20260313_185114/Meta.json
```

### Files updated during retry

In the original run directory:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt`
- `Meta.json`

---

## 7. Sequential Multi-Model Suite Usage

Runner utility:

- [src/EvalRing/utils/suite_runner.py](../src/EvalRing/utils/suite_runner.py)

### CLI Usage

```bash
evalring run-suite     --eval-script examples/suicide_detection/evaluate_rsd15k_main.py     --models-file examples/suicide_detection/model_list.json     --n-samples 100     --max-workers 10
```

Add `--yes` to skip the pre-run cache-summary confirmation. The prompt is
skipped automatically when stdin is not a terminal, so a suite launched from a
job runner cannot hang. Every flag is documented in [CLI.md](CLI.md).

### Automatic Visualizations

If the suite completes, the runner automatically delegates the generated `suite_report_*.json` into the visualization utility which produces comparison graphs and a composite Markdown presentation inside a bounded `visuals/` directory.

---

## 8. LLM Judge Evaluation

Runner file:

- [examples/suicide_detection/evaluate_rsd15k_llm_judge.py](../examples/suicide_detection/evaluate_rsd15k_llm_judge.py)

### Usage

Use the LLM-as-a-Judge API to score another language model's reasoning and classification alignment against the ground truth. This mode requires configuring rubrics (like "reasoning_quality", "classification_correctness", etc.) to dictate how a master LLM judge should grade responses.

Run the judge evaluator:

```bash
python examples/suicide_detection/evaluate_rsd15k_llm_judge.py
```

---

## 8.5 Generative-benchmark evaluation (HLE, ARC-Challenge, …)

A second example under [examples/hle](../examples/hle) benchmarks models on
generative datasets using a two-stage *generate → LLM-judge* pipeline (free-form
answers graded by semantic equivalence rather than exact match).

- Runner/flags: [examples/hle/README.md](../examples/hle/README.md)
- Providers & keys: [docs/providers/](providers/)
- Datasets (one file each): [docs/dataset/](dataset/) —
  [HLE](dataset/hle.md), [ARC-Challenge](dataset/arc_challenge.md)

Quick start:

```bash
# Ingest a dataset (HLE is gated; needs HF_TOKEN)
python examples/hle/ingest_hle.py
python examples/hle/ingest_arc.py

# Run the model suite (any dataset via --data-path)
python examples/hle/run_hle_suite.py --n-samples 20 --max-workers 50 --data-path examples/hle/data/arc_challenge.csv
```

---

## 9. Reusable Core Behavior for Future Tasks

Core module:

- [src/EvalRing/agent/classification.py](../src/EvalRing/agent/classification.py)

Reusable utilities include:

- `resolve_classification_prediction`
- `normalize_probability_distribution`
- `aggregate_base_vs_rest_probabilities`

Implication for future domains:

- You can return either label strings or class-score mappings from agents.
- Evaluator-level metric calculation can stay unchanged.
- Base-vs-rest strategies can be adapted to emotion/intent/categorization tasks with minimal code changes.

---

## 10. Suggested Commands

Fast sanity run:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Structured chance run:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode multi-class-chance
```

Base-vs-rest run:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --n-samples 20 --max-workers 4 --agent-mode base-vs-rest-binary --base-class Indicator
```

Retry broken rows only:

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```
