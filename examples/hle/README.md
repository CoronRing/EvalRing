# Generative-benchmark runner (HLE, ARC-Challenge, …)

A two-stage *generate → LLM-judge* evaluation harness. A candidate model answers
each question; an LLM judge decides whether the answer matches the gold answer
(semantic equivalence), so free-form answers work without exact string match.

Originally built for [`cais/hle`](https://huggingface.co/datasets/cais/hle); it
runs any dataset in the shared CSV schema via `--data-path`.

- **Providers & keys:** see [docs/providers/](../../docs/providers/) — one file per provider
  ([openai](../../docs/providers/openai.md), [gemini](../../docs/providers/gemini.md),
  [openrouter](../../docs/providers/openrouter.md),
  [litellm + Radium](../../docs/providers/litellm.md)).
- **Datasets:** see [docs/dataset/](../../docs/dataset/) — one file per dataset
  ([HLE](../../docs/dataset/hle.md), [ARC-Challenge](../../docs/dataset/arc_challenge.md)).

## Layout

| File | Purpose |
|---|---|
| `ingest_hle.py` / `ingest_arc.py` | Download a dataset → CSV (shared schema). |
| `hle_agent.py` | `HLEAgent` — thin subclass of core `OpenAIAgent`; adds the answer contract + answer/confidence extraction. |
| `hle_judge.py` | `HLEJudge` — LLM grader returning `{extracted_final_answer, correct, confidence}`. |
| `hle_evaluator.py` | `HLEEvaluator` — parallel generate→judge loop; accuracy, RMS calibration, per-category accuracy, token/latency + reasoning-token roll-ups. |
| `evaluate_hle_main.py` | Single-model runner → timestamped run dir (`result.md`, `all_cases.csv`, `all_cases.txt`, `all_cases_partial.jsonl`, `Meta.json`). |
| `run_hle_suite.py` | Runs the whole `model_list.json` suite concurrently → `suite_report_*.json` + markdown, plus a `visuals/` dir and `visuals_summary.md` (accuracy / errors / reasoning tokens / calibration charts). Use `--report-only <suite_dir>` to (re)build the report + visuals from existing runs. |
| `model_list.json` | Model suite + judge config, with per-model LiteLLM routing and `max_workers`. |

> Reusable model-call behaviour (streaming telemetry, `finish_reason`,
> reasoning-token capture, empty/timeout→error, complete error messages,
> `reasoning_effort`) lives in the core `EvalRing.agent` — this example is only
> the task-specific prompt, judge, and reporting.

## Run

Whole suite (parallel across models; `basic` mode, medium thinking):

```bash
# HLE (gated — needs HF_TOKEN; text-only by default)
python examples/hle/run_hle_suite.py --n-samples 10 --max-workers 5

# ARC-Challenge (public; 50 workers/model)
python examples/hle/run_hle_suite.py `
    --n-samples 20 --max-workers 50 --data-path examples/hle/data/arc_challenge.csv
```

Single model:

```bash
python examples/hle/evaluate_hle_main.py `
    --n-samples 10 --max-workers 5 --agent-mode basic --reasoning-effort medium `
    --litellm-model openai/gpt-5.5 --api-key-env OPENAI_API_KEY
```

### Key flags (`evaluate_hle_main.py`)

- `--n-samples N` — first N entries (deterministic, file order).
- `--max-workers K` — **parallel** worker threads.
- `--data-path` — dataset CSV (defaults to the HLE data).
- `--reasoning-effort` — "thinking" level (dropped if a provider doesn't support it).
- `--max-completion-tokens` — output budget. **Default `0` = no cap** (reasoning models finish naturally instead of truncating to empty).
- `--request-timeout-s` — per-request timeout (default 600s). See [litellm.md](../../docs/providers/litellm.md#timeouts).
- `--include-images` — include multimodal questions (default: text-only).
- `--litellm-model / --api-key-env / --api-base-env` — candidate routing.
- `--judge-litellm-model / --judge-api-key-env / --judge-api-base-env` — judge routing.

## Notes

- HLE is intentionally brutal; low accuracy on a small run is expected.
- Datasets' large archives and `_EvalRing/` run outputs are git-ignored.
