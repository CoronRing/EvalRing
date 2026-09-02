# API Reference

Everything importable from the top-level `EvalRing` namespace. Names not listed
here are internal and may change without notice.

```python
import EvalRing
EvalRing.__all__   # the complete public surface, asserted by a test
```

---

## Datasets

A dataset is constructed first and loaded second. `load_data()` is separate
from `__init__` so the same object can be built from a file, a frame, or
sample-by-sample.

### `DataSample`

One evaluation record.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Unique per sample. Resuming, retrying, and per-sample merges all key on it. |
| `input_text` | `str` | What the agent receives. |
| `target_output` | `Any` | Ground truth. |
| `metadata` | `dict[str, Any]` | Every column not mapped to the fields above. |

`to_dict()` returns a JSON-serializable dictionary.

### `BaseDataset`

Abstract base. Subclass it to support a new source.

```python
class BaseDataset(ABC):
    def __init__(self, name: str, description: str | None = None, version: str = "1.0")

    @abstractmethod
    def load_data(self, source, text_field="text", label_field="label",
                  id_field=None, **kwargs) -> None: ...

    @abstractmethod
    def validate_data(self) -> bool: ...
```

Concrete behaviour provided for you:

| Method | Purpose |
| --- | --- |
| `add_sample(sample)` | Append a `DataSample`. |
| `get_sample(index)` / `get_sample_by_id(id)` | Direct access. |
| `assert_unique_ids(expected_count=None, context="dataset")` | Raise if IDs repeat, naming the duplicates. The evaluator calls this before any model request. |
| `get_statistics()` | Sample count and input-length summary. |
| `split(train_ratio=0.8, seed=None)` | Reproducible train/test split. |
| `save(path)` / `load_from_file(path)` | JSON round-trip preserving name, version, and metadata. |
| `len()`, iteration, indexing, slicing | Standard sequence behaviour. |

**Unique IDs matter.** With `id_field=None` the row index is used, which breaks
resume and retry as soon as the dataset is reordered or filtered. Pass a real
per-row identifier — and note it must identify the *row*, not the author or
document a row belongs to.

### `JSONDataset`, `CSVDataset`, `DataFrameDataset`

```python
dataset = JSONDataset(name="reviews")
dataset.load_data("reviews.json", text_field="text", label_field="label", id_field="id")

dataset = CSVDataset(name="reviews")
dataset.load_data("reviews.csv", text_field="review", label_field="sentiment",
                  id_field="row_id", encoding="utf-8")   # extra kwargs reach pandas

dataset = DataFrameDataset(name="reviews")
dataset.load_data(df, text_field="text", label_field="target")
```

`JSONDataset` expects a JSON array of objects and defaults `id_field` to
`"id"`. `CSVDataset` and `DataFrameDataset` default it to `None`. All three
raise `ValueError` when the named text or label column is absent, and
`FileNotFoundError` for a missing file.

---

## Agents

### `AgentResponse`

| Field | Type | Meaning |
| --- | --- | --- |
| `input_id` | `str` | Set by the evaluator; return `""`. |
| `input_text` | `str` | Echo of the input. |
| `output` | `Any` | A label string, or a `{label: score}` mapping. |
| `confidence` | `float \| None` | Optional. |
| `metadata` | `dict[str, Any]` | Merged into the per-sample metrics — put timing and token counts here. |
| `processing_time` | `float \| None` | Optional. |
| `error` | `str \| None` | Non-`None` marks the response as failed. |

`is_successful()` returns `error is None`.

### `BaseAgent`

```python
class BaseAgent(ABC):
    def __init__(self, name: str, version: str = "1.0",
                 description: str | None = None, **kwargs)

    @abstractmethod
    def initialize(self, **kwargs) -> None: ...          # must set self._is_initialized = True

    @abstractmethod
    def predict(self, input_text: str, **kwargs) -> AgentResponse: ...
```

`predict_batch()`, `validate_input()`, `get_info()`, and `save_config()` are
provided. The evaluator calls `initialize()` for you if you have not.

### `MockAgent`

```python
MockAgent(name="mock_agent", fixed_response=None,
          possible_outputs=["positive", "negative", "neutral"], delay=0.1)
```

Returns `fixed_response` if given, otherwise a random choice. Use `delay=0` in
tests.

### `RuleBasedAgent`

```python
RuleBasedAgent(name="rule_based_agent",
               rules={"positive": ["great", "love"]},
               default_output="unknown")
```

Case-insensitive substring matching, first rule wins. A useful floor to compare
model results against.

### `OpenAIAgent`

Streaming client for any OpenAI-compatible endpoint.

```python
OpenAIAgent(
    name="openai-agent",
    model_name=None,               # falls back to $EVALRING_MODEL, then "gpt-4o"
    api_key=None,                  # resolved from the environment
    base_url=None,                 # resolved alongside the key
    system_prompt="You are a helpful assistant.",
    temperature=0.0,
    max_completion_tokens=256,     # <= 0 means no cap
    reasoning_effort=None,         # forwarded where supported, dropped elsewhere
    error_on_empty=True,           # empty completion counts as an error
)
```

Subclass it and set `self.system_prompt` for a task-specific agent; override
`_parse_output()` to clean the raw text.

On success, `metadata` carries `raw_output`, `model`, `base_url`, TTFT, tokens
per second, `prompt_tokens`, `completion_tokens`, `total_tokens`,
`reasoning_tokens`, and `reasoning_content` where the provider streams it.

Credentials resolve at construction; the error is raised by `initialize()`, so
building an agent never requires a key. See
[CONFIGURATION.md](CONFIGURATION.md).

### `MultiRoleHostOrchestrator` and `RoleConfig`

A host model consults several persona models and decides a final label.

```python
RoleConfig(name, persona, model_name, temperature=0.0, max_completion_tokens=400)

MultiRoleHostOrchestrator(
    client=..., labels=[...], task_name=..., task_instructions=...,
    host_model_name=..., role_configs=[...],
    host_temperature=0.0, host_max_completion_tokens=500, max_iterations=10,
)
```

### Classification helpers

| Function | Purpose |
| --- | --- |
| `resolve_classification_prediction(output, valid_labels=None, label_aliases=None)` | Normalize a string or `{label: score}` output into a `ClassificationPrediction`. Ties break by `valid_labels` order, then alphabetically. |
| `normalize_probability_distribution(scores)` | Clamp negatives to zero and rescale to sum to 1. |
| `aggregate_base_vs_rest_probabilities(base_label=..., target_vs_base_probs=..., all_labels=None)` | Convert pairwise one-vs-base probabilities into a full multi-class distribution via odds. |
| `parse_json_object(raw_text)` | Extract the first JSON object from model text, or `None`. |

`ClassificationPrediction` has `label`, `confidence`, and `class_scores`.

### Error classification

```python
result = classify_error(message)
result.is_rate_limit   # dedicated backoff
result.is_transient    # worth retrying
result.is_terminal     # retrying only wastes tokens
```

Terminal conditions win over transient-looking text: an empty response that
also mentions a timeout is terminal. `ErrorClass` is the frozen result type.

---

## Evaluators

### `EvaluationMetrics`

`metrics: dict[str, float]`, `per_sample_metrics: list[dict]`, and
`metadata: dict`. Use `add_metric(name, value)` and
`get_metric(name, default=None)`.

### `EvaluationResult`

`agent_name`, `dataset_name`, `metrics`, `duration`, `timestamp`, `task_name`,
`version`, `metadata`. `to_dict()` and `save(path)` serialize it.

### `BaseEvaluator`

```python
BaseEvaluator(name="evaluator", output_dir=None, **kwargs)   # cache_mode kwarg accepted
```

`validate_inputs()` checks types, rejects an empty dataset, runs
`validate_data()`, and enforces unique IDs. `save_results()` writes a
timestamped JSON file and returns its path.

### `ClassificationEvaluator`

```python
result = evaluator.evaluate(
    agent, dataset, task_name,
    version="1.0",
    max_workers=5,
    max_retries=3,
    show_progress=False,
    exit_on_first_error=False,   # abort if the very first request fails
    partial_cb=None,             # called with each per-sample metric as it lands
    error_cb=None,               # called with a diagnostic dict on each failed attempt
)
```

Samples run on a thread pool. Rate limits sleep 10s and retry up to 30 times;
other errors back off exponentially up to `max_retries`. A sample that never
succeeds is recorded in `metadata["execution_failures"]` and scored 0 — the run
continues.

Metrics produced: `accuracy`, macro `precision`, `recall`, `f1_score`, plus
`avg_ttft`, `std_ttft`, `avg_tps`, `std_tps`, `avg_generation_time`,
`avg_total_time`, and token totals when the agent reports them in `metadata`.

`per_sample_metrics` is ordered to match dataset iteration order, not
completion order.

```python
evaluator.retry_failed_cases(meta_file_path, agent, dataset)
```

Reads a previous run's `Meta.json` and re-evaluates only the failed samples.

### `LLMJudgeEvaluator`

Two phases: collect agent responses, then score them with one or more judge
metrics.

```python
evaluator = LLMJudgeEvaluator.from_rubric(
    rubric="Score 1-5 on factual accuracy.",   # str, Rubric, or {name: rubric}
    criteria="Accuracy against the reference answer",
    judge_model=None,          # falls back to $EVALRING_MODEL, then "gpt-4o"
    api_key=None,
    base_url=None,
    judge_temperature=0.0,
    judge_max_tokens=512,
    weights=None,              # {metric_name: weight}
    threshold=0.5,
    strict_mode=False,         # binary 0/1 scoring
    max_workers=5,
    max_retries=3,
)
result = evaluator.evaluate(agent, dataset, task_name="qa-quality")
```

Supporting types: `Rubric(name, levels, score_range=None, description="")`,
`RubricLevel(score, label, description)`,
`ScoringCriteria(name, criteria, evaluation_steps=None, rubric=None, weight=1.0)`,
`JudgeVerdict(score, reason, metadata)`, `EvalSteps`, `JudgeTemplate`,
`JudgeMetric`, and the `LLMJudge` / `OpenAIJudge` backends.

Subclass `LLMJudge` and implement `generate(prompt) -> str` plus
`get_model_name()` to use a provider the OpenAI SDK cannot reach.

---

## Suite tooling

### `run_suite()`

```python
run_suite(
    eval_script, model_list_path,
    n_samples=2, max_workers=32, seed=0,
    agent_mode="single-class", base_class=None,
    host_model=None, role_models_json=None, max_host_iterations=10,
    continue_runs=False, cache=None, cache_mode="both",
    ignore_errors=False, out_dir=None, data_path=None, no_confirm=False,
) -> int
```

Runs `eval_script` once per model in a separate subprocess with the model
pinned through the environment. Returns `0` when the suite ran, `1` when it
could not start; individual model failures appear in the report. The evaluation
script must accept the flags listed in the function's docstring.

The interactive confirmation is skipped automatically when stdin is not a TTY.

### `GlobalCache`

Process-wide singleton over a SQLite database in
`<workspace>/_EvalRing/Cache/`. The workspace is `EVALRING_WORKSPACE`, then the
current directory.

| Member | Purpose |
| --- | --- |
| `GlobalCache(mode="both", workspace_root=None)` | `"both"`, `"cache_file"`, `"runs_only"`, or `"none"`. |
| `generate_key(model_name, payload, params)` | Stable SHA-256 key; dict ordering does not affect it. |
| `get(key)` / `set(key, payload)` | Direct access. |
| `lookup(...)` | Multi-tier: SQLite, then previous run directories, migrating hits into SQLite. |
| `reset_instance()` | Drop the singleton so the next construction re-reads its workspace. Tests and long-lived processes only. |

### `generate_model_list(output_path, requested_mapping=None)`

Writes the model-list JSON that `run_suite()` consumes, pulling pricing and
metadata from the OpenRouter catalogue. Requires network access.

### `generate_suite_visuals(report_path, model_list_path)`

Renders comparison charts next to the suite report. Requires the `viz` extra;
check `EvalRing.utils.VISUALS_AVAILABLE` first.

---

## Configuration and logging

| Name | Purpose |
| --- | --- |
| `resolve_credentials(api_key=None, base_url=None, env=None)` | Returns `ProviderCredentials(api_key, base_url, provider, source)`. |
| `ProviderCredentials.require_key()` | Returns the key or raises `MissingCredentialsError`. |
| `resolve_model_name(model_name=None, default=None, env=None)` | Model precedence resolution. |
| `MissingCredentialsError` | Carries the list of variables the user can set. |
| `get_logger(name)` | Logger inside the `EvalRing` namespace. |
| `configure_logging(level=logging.INFO, verbose=False)` | Attach one stderr handler. Idempotent. For entry points only. |

The library never configures the root logger. To silence it in your
application:

```python
import logging
logging.getLogger("EvalRing").setLevel(logging.WARNING)
```
