# EvalRing Design Specification

**Spec version:** 2.0 — applies to EvalRing 0.2.0.

## 1. Purpose

EvalRing is a lightweight evaluation framework for model and agent benchmarking.

The system has two layers:

1. **Core framework layer** in [src/EvalRing](../src/EvalRing) — the installable
   package, and the only code that ships to PyPI.
2. **Task-specific evaluation applications** in [examples](../examples), such as
   [examples/suicide_detection](../examples/suicide_detection) and
   [examples/hle](../examples/hle). These are not packaged and are not
   importable; they are runnable scripts that depend on the installed package.

The framework standardizes:

- how datasets are loaded
- how agents are called
- how evaluation metrics are computed
- how per-sample and aggregate results are persisted
- how failed executions are retried without rerunning the whole experiment

---

## 2. System Goals

### Primary goals

- Provide a common interface for benchmarking heterogeneous agents
- Preserve reproducibility through saved run configuration and metadata
- Support structured result persistence for research workflows
- Support incremental recovery from transient model/API failures

### Non-goals

- Full experiment orchestration platform
- Dataset labeling or annotation tooling
- Dashboarding or visualization server
- Distributed job execution manager

---

## 3. High-Level Architecture

## 3.1 Core modules

### Agent layer

Location: [src/EvalRing/agent](../src/EvalRing/agent)

Responsibilities:

- define the agent contract
- wrap model backends
- normalize prediction outputs into a common response object

Key public types:

- `BaseAgent`
- `AgentResponse`
- `ClassificationPrediction`
- `MockAgent`
- `RuleBasedAgent`
- `OpenAIAgent`

### Dataset layer

Location: [src/EvalRing/dataset](../src/EvalRing/dataset)

Responsibilities:

- load raw evaluation data
- normalize rows into `DataSample`
- expose iterable samples to the evaluator

Key public types:

- `BaseDataset`
- `DataSample`
- `JSONDataset`
- `CSVDataset`
- `DataFrameDataset`

### Evaluator layer

Location: [src/EvalRing/evaluator](../src/EvalRing/evaluator)

Responsibilities:

- run agents against datasets
- coordinate concurrency and retries
- compute task metrics
- emit evaluation result objects
- resolve top-class labels from either plain labels or structured class-score outputs

Key public types:

- `BaseEvaluator`
- `EvaluationMetrics`
- `EvaluationResult`
- `ClassificationEvaluator`

### LLM-as-a-Judge layer

Location: [src/EvalRing/evaluator/llm_judge](../src/EvalRing/evaluator/llm_judge)

Responsibilities:

- rubric-based scoring
- templated judgment prompts
- evaluation pipelines where an LLM judges another output

Key public types exported through [src/EvalRing/evaluator/__init__.py](../src/EvalRing/evaluator/__init__.py):

- `Rubric`
- `RubricLevel`
- `ScoringCriteria`
- `JudgeVerdict`
- `EvalSteps`
- `JudgeTemplate`
- `JudgeMetric`
- `LLMJudge`
- `OpenAIJudge`
- `LLMJudgeEvaluator`

---

## 4. Object Model

## 4.1 `DataSample`

Canonical dataset row representation.

Fields:

- `id`: unique or semi-unique sample identifier
- `input_text`: model input text
- `target_output`: ground-truth label or expected output
- `metadata`: row-level auxiliary fields

## 4.2 `AgentResponse`

Canonical prediction response returned by agents.

Fields:

- `input_id`
- `input_text`
- `output`
- `confidence`
- `metadata`
- `processing_time`
- `error`

For model-backed agents, `metadata` may include runtime telemetry such as:

- `raw_output`
- `model`
- `base_url`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `ttft`
- `generation_time`
- `tps`
- `total_time`

## 4.3 `EvaluationResult`

Top-level result container returned by evaluators.

Fields:

- `agent_name`
- `dataset_name`
- `metrics`
- `duration`
- `timestamp`
- `task_name`
- `version`
- `metadata`

---

## 5. Runtime Flow

Standard classification flow:

1. Load dataset into a `BaseDataset` implementation
2. Initialize the agent
3. Create evaluator instance
4. Run `evaluate()`
5. Collect per-sample predictions
6. Compute aggregate metrics
7. Persist run artifacts

In the suicide detection example, this flow is implemented in [examples/suicide_detection/evaluate_rsd15k_main.py](../examples/suicide_detection/evaluate_rsd15k_main.py).

---

## 6. Concurrency and Retry Model

## 6.1 First-pass evaluation

`ClassificationEvaluator` uses a `ThreadPoolExecutor` to process samples concurrently.

Configurable controls:

- `max_workers`
- `max_retries`

Retry behavior during first pass:

- each sample can be retried up to `max_retries`
- exponential backoff is used for repeated failures
- failed predictions are recorded as `prediction = "Error"`

## 6.2 In-place retry mode

The suicide detection application adds a second retry layer beyond the evaluator:

- CLI flag: `--retry-failed`
- optional pointer: `--meta-path <path to Meta.json>`

Behavior:

1. load the original run’s `Meta.json`
2. reuse original run config (`n_samples`, `max_workers`, `max_retries`, `seed`)
3. scan original `all_cases.csv`
4. select only rows with:
   - `prediction == Error`, or
   - non-empty `error`
5. rerun only those rows
6. patch original run artifacts in place

This prevents rerunning the entire experiment for one or two transient API failures.

---

## 7. Provider and Model Abstraction

## 7.1 Credential resolution

All credential and endpoint resolution is centralized in
[src/EvalRing/config.py](../src/EvalRing/config.py). No component reads
`os.environ` for a key directly, and no endpoint appears as a default argument.

`resolve_credentials(api_key=None, base_url=None, env=None)` returns a
`ProviderCredentials` carrying the key, the base URL, the provider identifier,
and a printable description of which variable supplied the key. The key itself
never appears in that description, so it is safe to log.

Precedence, first non-blank wins: an explicit `api_key=` argument, then
`EVALRING_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPEN_ROUTER_KEY`,
`RADIUM_API_KEY`. A provider's default base URL applies only when that
provider's own key is the one selected. The full table, including base URL and
model precedence, is in [CONFIGURATION.md](CONFIGURATION.md).

Adding a backend means adding a row to `_CREDENTIAL_SOURCES`, not reading a new
variable somewhere in the call path.

## 7.1.1 Generic model client

`OpenAIAgent` in [src/EvalRing/agent/implementations.py](../src/EvalRing/agent/implementations.py)
is the common model-backed client. It resolves credentials at construction but
raises only at `initialize()`, so constructing an agent never requires a key —
which is what lets the test suite exercise agent configuration offline.

Transport is selected by `EVALRING_LLM_TRANSPORT`: `litellm` (default)
normalizes parameters across providers and silently drops ones a given model
rejects; `openai` uses the OpenAI SDK directly. If LiteLLM is not installed the
SDK path is used regardless.

## 7.2 Task-specific agent

The suicide detection task uses `OpenAISuicideDetectionAgent` in [examples/suicide_detection/llm_agent.py](../examples/suicide_detection/llm_agent.py).

Responsibilities:

- define task prompt
- define label space
- construct user messages
- support multiple output modes:
  - single-class label
  - multi-class probabilities
  - base-vs-rest binary decomposition

### 7.2.1 Suicide detection modes

Current `OpenAISuicideDetectionAgent` modes:

- `single-class`
- `multi-class-chance`
- `base-vs-rest-binary`

`base-vs-rest-binary` takes a configurable `base_class`, runs one binary inference per non-base label, then aggregates the pairwise probabilities into one multi-class distribution.

### 7.2.2 Reusable classification output utilities

Location: [src/EvalRing/agent/classification.py](../src/EvalRing/agent/classification.py)

Core reusable functions:

- `resolve_classification_prediction`
- `normalize_probability_distribution`
- `aggregate_base_vs_rest_probabilities`

Design intent:

- future tasks can emit structured class scores without changing evaluator interfaces
- future tasks can reuse base-vs-rest decomposition and aggregation logic

## 7.3 Default model resolution

`resolve_model_name()` in [src/EvalRing/config.py](../src/EvalRing/config.py)
applies one order for every component: an explicit `model_name=` argument, then
`EVALRING_MODEL`, `OPENAI_MODEL`, `OPENROUTER_MODEL`, `OPEN_ROUTER_MODEL`,
`RADIUM_MODEL`, then the component's own default (`gpt-4o` for agents and
judges).

The suite runner sets every one of these per subprocess when iterating a model
list, so an evaluation script written against any single variable works inside
a suite.

---

## 8. Suicide Detection Application Design

Location: [examples/suicide_detection](../examples/suicide_detection)

This application provides a concrete research workflow around the RSD_15K dataset.

### Inputs

- dataset: `data/rsd_15k.csv`
- environment keys from `EvalRing/.env`
- optional model override via environment variables

### Outputs per run

Created under:

- [examples/suicide_detection/_EvalRing](../examples/suicide_detection/_EvalRing)

Each run produces a timestamped folder containing:

- `result.md`
- `all_cases.txt`
- `all_cases.csv`
- `failed_cases.txt` (only when failures remain)
- `Meta.json`

### Metadata recorded

`Meta.json` stores:

- run configuration
- model configuration
- mode configuration (`agent_mode`, `base_class`)
- dataset configuration
- aggregate metrics
- per-class metrics
- confusion matrix
- estimated token cost
- retry history

---

## 9. Sequential Multi-Model Runner

Location: [examples/suicide_detection/run_model_suite.py](../examples/suicide_detection/run_model_suite.py)

Purpose:

- benchmark multiple models in sequence
- stop immediately when a model run is not error-free

Current guarantees:

- models are run one-by-one in declared order
- next model runs only if previous model has:
  - zero process failures
  - zero `Meta.json` execution failures
  - zero per-sample prediction errors in `all_cases.csv`

This script is intended for controlled benchmarking rather than maximum throughput.

---

## 9.5 Humanity's Last Exam (HLE) Application

Location: [examples/hle](../examples/hle)

A second task application that exercises a different evaluation shape than the
classification example: HLE answers are free-form, so grading uses an LLM judge
rather than exact-match against a fixed label set.

Components:

- `HLEAgent` (subclass of `OpenAIAgent`): LiteLLM-backed answerer with per-model
  provider routing (OpenAI / Gemini / Radium in one suite), a configurable
  `reasoning_effort` ("thinking" level), and `litellm.drop_params` so
  heterogeneous providers degrade gracefully.
- `HLEJudge`: LLM grader returning `{extracted_final_answer, correct, confidence}`.
- `HLEEvaluator` (subclass of `BaseEvaluator`): parallel *generate → judge*
  pipeline with the same retry / partial-callback / error-callback contract as
  `ClassificationEvaluator`. Aggregates accuracy, RMS calibration error,
  per-category accuracy, and token/latency roll-ups.
- `run_hle_suite.py`: runs a mixed-provider model list concurrently.

Design significance: this demonstrates the "Add a new evaluator" extension point
(§12) — a judge-based, generative evaluator plugged into the same dataset/agent
primitives and the same run-artifact layout (`result.md`, `all_cases.csv`,
`all_cases_partial.jsonl`, `Meta.json`).

---

## 10. Persistence Contract

## 10.1 `all_cases.csv`

Machine-readable source of truth for sample-level results.

Columns:

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

## 10.2 `Meta.json`

Machine-readable source of truth for run-level state.

Used for:

- reproducibility
- downstream analysis
- retry recovery
- suite-level validation

## 10.3 `result.md`

Human-readable report containing:

- run summary
- aggregate metrics
- per-class metrics
- confusion matrix
- per-sample summary table
- retry update summary when applicable

---

## 11. Known Constraints

### Duplicate sample IDs

The suicide detection dataset currently uses `users` as sample ID, which is not globally unique in all observed rows.

Impact:

- duplicate identifiers may appear in reports
- row position is sometimes a more reliable retry key than dataset-provided ID

The retry system therefore uses row position from `all_cases.csv` for in-place error recovery.

### The response cache is a process-wide singleton

`GlobalCache` fixes its workspace root on first construction; later
constructions return the same object. One evaluator per process is the intended
shape. `GlobalCache.reset_instance()` exists for tests and for long-lived
processes that switch workspaces, and `EVALRING_WORKSPACE` moves the root away
from the current working directory.

### The suite runner orchestrates subprocesses, not objects

`run_suite()` invokes an evaluation *script* per model rather than calling into
Python objects. That is what gives per-model crash isolation, but it means the
script must accept the flags the runner forwards, and the contract between them
is positional rather than typed. The flags are enumerated in the `run_suite()`
docstring and in [CLI.md](CLI.md).

---

## 12. Extension Points

### Add a new dataset loader

Implement a new `BaseDataset` subclass and normalize all rows to `DataSample`.

### Add a new agent backend

Implement a new `BaseAgent` subclass or adapt `OpenAIAgent` for a different API surface.

### Add a new evaluator

Implement a `BaseEvaluator` subclass for tasks such as:

- generation quality
- ranking
- retrieval evaluation
- judge-based rubric scoring

### Add a new task application

Pattern to follow:

1. create a task-specific folder under [examples](../examples)
2. add a task-specific agent wrapper if needed
3. add an application runner that loads data, runs evaluation, and writes structured outputs
4. optionally add retry and suite orchestration

Behaviour that two tasks both need belongs in `src/EvalRing/`, not duplicated
across two scripts. See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## 13. Design Principles

The current implementation follows these practical principles:

- **simple contracts** over framework complexity
- **file-based reproducibility** over database dependence
- **recoverability** over full rerun cost
- **research traceability** over minimal logging
- **task-specific applications** built on top of reusable primitives
- **no vendor in the core**: any OpenAI-compatible endpoint, resolved through
  one documented precedence order
- **optional dependencies stay optional**: `import EvalRing` succeeds with the
  core dependencies alone; provider SDKs and plotting are extras
- **the library logs, the entry point prints**: nothing under `EvalRing/`
  writes to stdout or configures the root logger
