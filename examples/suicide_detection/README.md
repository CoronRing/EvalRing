# Suicide Detection Evaluation Pipeline

A complete pipeline for running and evaluating suicide-risk classification with
the **EvalRing** framework.

> **Read [docs/DATA.md](../../docs/DATA.md) first.** The RSD-15k corpus is
> sensitive and is not distributed with this repository. Nothing produced here
> is a clinical, screening, or triage instrument.

## 1. Environment Setup

1. **Install EvalRing.** These scripts import the installed package; they do not
   manipulate `sys.path`.

   ```bash
   pip install -e ".[all]"     # from the repository root
   ```

2. **Configure a provider.** One key is enough:

   ```bash
   export EVALRING_API_KEY="your-key"
   export EVALRING_BASE_URL="https://openrouter.ai/api/v1"   # omit for OpenAI
   ```

   `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPEN_ROUTER_KEY`, and
   `RADIUM_API_KEY` are also recognized, in that order. Confirm what was
   resolved with `evalring check`; the full precedence table is in
   [docs/CONFIGURATION.md](../../docs/CONFIGURATION.md).

## 2. Dataset Preparation

The original dataset must be parsed and split into single-turn (`simple`) and multi-turn (`multi_round`) formats for structured evaluation and to keep the original source immutable.

**Run the Data Translator:**
```bash
python examples/suicide_detection/data_translator.py
```

By default, this will scan for the source data and output two processed `.csv` files into the `examples/suicide_detection/data/` directory:
- `rsd_15k_simple.csv`
- `rsd_15k_multi_round.csv`

*(If using a custom data location, you can direct it via the `--input-csv` argument).*

## 3. Running the Model Suite

The primary evaluation execution is wrapped by `run_model_suite.py`. This script acts as an automated harness, leveraging `EvalRing.utils.suite_runner` to systematically traverse through a list of LLMs configured in a JSON file.

### Basic Suite Execution
To test models using default parameters:

```bash
python examples/suicide_detection/run_model_suite.py
```

### Advanced Suite Execution
For formal publication runs, you will likely customize parameters like sample size and thread pool.

```bash
python examples/suicide_detection/run_model_suite.py \
    --n-samples 100 \
    --max-workers 50 \
    --agent-mode single-class \
  --models-file model_list.json
```

**Key Parameters:**
- `--n-samples`: Number of samples to evaluate. Omit or set appropriately for full dataset runs.
- `--max-workers`: Maximum concurrent workers for API calls (adjust carefully based on your API rate limits).
- `--seed`: Random seed for reproducibility (default `42`).
- `--agent-mode`: Specifies the classification strategy. Supported patterns:
  - `single-class`: Model outputs a single textual label.
  - `multi-class-chance`: Model provides probabilities for multiple classes.
  - `base-vs-rest-binary`: Agent runs pairwise comparisons against a base class.
  - `multi-agent-host`: A designated host model coordinates with specialized roles (e.g. an "expert" and "critic" model).
- `--base-class`: Applicable only when using `base-vs-rest-binary` (e.g., `Indicator`, `Attempt`).
- `--models-file`: The target JSON file indexing all the models you wish to benchmark in the suite.
- `--no-cache`: Force restarts without applying run caches.

## 4. Single-Model Granular Evaluation

To troubleshoot a specific model or debug agent logic, use the inner script `evaluate_rsd15k_main.py` directly. This evaluates whatever the default environment configuration is defined as, instead of sweeping across the model JSON file.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py \
    --n-samples 50 \
    --agent-mode multi-class-chance
```

## 5. Resuming and Retrying Runs

In extensive programmatic runs, API instability can occur. The EvalRing framework handles persistent state tracking. A `Meta.json` artifact within the output directory acts as the run ledger.

### Auto-Continue
To resume operations that were interrupted, appending the `--continue` flag will bypass already successful evaluations from the previous payload, ensuring cost efficiency.

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --continue
```
*Note: `run_model_suite.py` safely evaluates with `--continue` logic out-of-the-box unless explicitly bypassed with `--no-cache`.*

### Target Failed API Calls
If you only need to rerun inferences that resulted in an `Error` state (bypassing the cache mapping):

```bash
python examples/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

## 6. Output Artifacts and Analysis

Upon completion, output traces are bundled under the `_EvalRing/` output cache directory structure (timestamped per run).

**Contents typically include:**
- `Meta.json`: Comprehensive configuration tracing the parameters, inputs, constraints, and total latency.
- Event JSON lines logging the prompts submitted, API replies generated, and token metadata.
- Generated probability matrices (if `multi-class-chance` is the active topology).

These detailed payload traces integrate downstream into `EvalRing.utils.visualizations` and significance verification libraries for final formal charting and statistics reporting.