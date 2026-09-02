# Command-line reference

Installing EvalRing provides the `evalring` command. Every subcommand is a thin
wrapper over the Python API described in [API_REFERENCE.md](API_REFERENCE.md).

```
usage: evalring [-h] [--version] [-v] [-q] {info,check,models,run-suite} ...
```

| Global flag | Effect |
| --- | --- |
| `--version` | Print the installed version and exit. |
| `-v`, `--verbose` | Debug logging, with timestamps and logger names. |
| `-q`, `--quiet` | Warnings and errors only. |

Log records go to **stderr**; machine-readable output goes to **stdout**, so
`evalring info ... \| jq` works.

---

## `evalring check`

Report the provider configuration EvalRing resolved. Run this first when a run
fails to authenticate or reaches the wrong endpoint.

```console
$ evalring check
{
  "evalring_version": "0.2.0",
  "python": "3.12.4",
  "api_key_found": true,
  "api_key_source": "$EVALRING_API_KEY",
  "provider": "evalring",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4",
  "recognized_key_variables": [
    "EVALRING_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPEN_ROUTER_KEY",
    "RADIUM_API_KEY"
  ],
  "optional_packages": {
    "litellm": true,
    "openai": true,
    "matplotlib": true,
    "nest_asyncio": false
  }
}
```

The API key is never printed — only which variable supplied it. Exit code is
`0` when a key was resolved and `1` when none was, which makes it usable as a
precondition check in a script.

---

## `evalring info`

Dataset statistics and validation, without running anything.

```console
$ evalring info --dataset data/reviews.csv --text-field review --label-field sentiment --id-field row_id
{
  "total_samples": 1500,
  "name": "reviews",
  "version": "1.0",
  "avg_input_length": 412.7,
  "min_input_length": 18,
  "max_input_length": 4096,
  "valid": true,
  "path": "data/reviews.csv"
}
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | required | Path to a `.json` or `.csv` file. |
| `--text-field` | `text` | Column or key holding the input. |
| `--label-field` | `label` | Column or key holding the ground truth. |
| `--id-field` | none | Column or key holding a unique sample ID. |

`"valid": false` means at least one sample has an empty input or a missing
label. Exit code is `1` if the file is missing or unreadable.

---

## `evalring models`

Generate the model-list JSON that `run-suite` consumes, using the OpenRouter
catalogue for identifiers and pricing. Requires network access.

```bash
evalring models --output model_list.json
```

Edit the result to select the models you actually want; entries with
`"available": false` are skipped by the suite runner.

---

## `evalring run-suite`

Run one evaluation script once per model, each in its own subprocess.

```bash
evalring run-suite \
    --eval-script examples/hle/evaluate_hle_main.py \
    --models-file examples/hle/model_list.json \
    --n-samples 500 \
    --max-workers 32 \
    --out-dir results/ \
    --yes
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--eval-script` | required | Evaluation script to invoke per model. |
| `--models-file` | required | Model list JSON. |
| `--n-samples` | `10` | Samples per model. |
| `--max-workers` | `5` | Request concurrency within each model's run. |
| `--seed` | `42` | Sampling seed forwarded to the script. |
| `--agent-mode` | `single-class` | Mode string forwarded to the script. |
| `--base-class` | none | Base class for one-vs-rest modes. Forwarded only when set. |
| `--host-model` | none | Orchestrator model for multi-role runs. |
| `--role-models-json` | none | JSON mapping role names to models. |
| `--max-host-iterations` | `10` | Iteration cap for multi-role orchestration. |
| `--continue-runs` | off | Resume from previous partial runs. |
| `--cache` | none | Path to a cache database or run directory. |
| `--cache-mode` | `both` | `runs_only`, `cache_file`, `both`, or `none`. |
| `--ignore-errors` | off | Re-run samples whose cached result was an error. |
| `--out-dir` | none | Parent directory for output. Defaults to `_EvalRing/` beside the script. |
| `--data-path` | none | Dataset path forwarded to the script. |
| `-y`, `--yes` | off | Skip the pre-run cache-summary confirmation. |

Without `--yes`, the runner first scans how many samples are already cached,
prints the totals, and asks before spending anything. That prompt is skipped
automatically when stdin is not a terminal, so CI never hangs.

Your evaluation script must accept `--n-samples`, `--max-workers`, `--seed`,
`--agent-mode`, `--max-host-iterations`, `--out-dir`, `-nc`, and `--info-only`,
plus the optional flags above when you use them. The model is pinned through
the environment (`EVALRING_MODEL` and the provider-specific aliases). See
[USAGE.md](USAGE.md) and the scripts under `examples/` for a working shape.

### Output

```
_EvalRing/run_suite_<timestamp>_<mode>/
├── run_<timestamp>_<model>_<mode>/     one directory per model
│   ├── all_cases.csv                   per-sample results
│   ├── all_cases_partial.jsonl         streamed during the run
│   ├── incorrect_cases.txt
│   ├── Meta.json                       configuration and failures
│   └── result.md
├── .log/run_<model>.log                stdout and stderr per model
├── suite_progress.txt                  live progress
├── suite_report_<timestamp>.json
├── run_suite_<timestamp>.md            combined report
└── visuals/                            charts, when matplotlib is installed
```

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The suite ran. Individual model failures are in the report and logged as errors. |
| `1` | The suite could not start: no models loaded, script missing, or no API key. |

---

## Scripting notes

`evalring check` is the cheap precondition:

```bash
evalring check > /dev/null || { echo "no API key configured" >&2; exit 1; }
evalring run-suite --eval-script eval.py --models-file models.json --yes
```

Keep artifacts out of your working directory by setting `EVALRING_WORKSPACE`,
or pass `--out-dir` explicitly.
