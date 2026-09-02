# Configuration

Every EvalRing component that calls a model — `OpenAIAgent`, `OpenAIJudge`,
`AIDataGenerator` — resolves its credentials through one function,
[`EvalRing.config.resolve_credentials()`](../src/EvalRing/config.py). Nothing
reads `os.environ` directly and no endpoint is hard-coded as a default. This
page is the reference for what it reads and in what order.

## The short version

```bash
export EVALRING_API_KEY="your-key"
```

That is enough for OpenAI. For anything else, add the endpoint:

```bash
export EVALRING_API_KEY="your-key"
export EVALRING_BASE_URL="https://openrouter.ai/api/v1"
```

Confirm what was resolved. The key itself is never printed:

```bash
evalring check
```

## API key precedence

The first variable that holds a non-blank value wins. An `api_key=` argument
passed in code beats all of them.

| Order | Variable | Default base URL when this key is selected |
| --- | --- | --- |
| 1 | `EVALRING_API_KEY` | none (OpenAI default) unless `EVALRING_BASE_URL` is set |
| 2 | `OPENAI_API_KEY` | none (OpenAI default) |
| 3 | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| 4 | `OPEN_ROUTER_KEY` | `https://openrouter.ai/api/v1` |
| 5 | `RADIUM_API_KEY` | `https://api.radium.cloud/v1` |

Entries 3-5 are compatibility aliases for environments that already set them.
New setups should use `EVALRING_API_KEY` plus `EVALRING_BASE_URL`, which
express the same thing without tying your configuration to a particular vendor.

A provider's default base URL applies **only** when that provider's key is the
one selected. Setting `OPENAI_API_KEY` never routes traffic anywhere but
OpenAI.

## Base URL precedence

| Order | Source |
| --- | --- |
| 1 | `base_url=` argument in code |
| 2 | `EVALRING_BASE_URL` |
| 3 | The variable paired with the selected key (`OPENAI_BASE_URL`, `OPENROUTER_BASE_URL`, `RADIUM_BASE_URL`) |
| 4 | The selected provider's default from the table above |

`None` means the OpenAI SDK's own default, `https://api.openai.com/v1`.

## Model precedence

| Order | Source |
| --- | --- |
| 1 | `model_name=` argument in code |
| 2 | `EVALRING_MODEL` |
| 3 | `OPENAI_MODEL` |
| 4 | `OPENROUTER_MODEL`, then `OPEN_ROUTER_MODEL` |
| 5 | `RADIUM_MODEL` |
| 6 | The component's own default (`gpt-4o` for agents and judges) |

The suite runner sets all of these per subprocess when iterating a model list,
so an evaluation script that reads any one of them works inside a suite.

## Behaviour variables

| Variable | Default | Effect |
| --- | --- | --- |
| `EVALRING_LLM_TRANSPORT` | `litellm` | `litellm` routes through LiteLLM, which normalizes parameters across providers. `openai` uses the OpenAI SDK directly. If LiteLLM is not installed, the OpenAI SDK is used regardless. |
| `OPENAI_REQUEST_TIMEOUT_S` | `120` | Per-request timeout in seconds. `0` or negative disables it. A stalled stream would otherwise block a worker forever. |
| `EVALRING_WORKSPACE` | current directory | Directory that holds `_EvalRing/` — the response cache and run artifacts. Set it to keep artifacts out of wherever you happen to launch from. |

## Worked examples

**OpenAI**

```bash
export EVALRING_API_KEY="sk-..."
export EVALRING_MODEL="gpt-4o"
```

**OpenRouter** — model names carry the provider prefix.

```bash
export EVALRING_API_KEY="sk-or-v1-..."
export EVALRING_BASE_URL="https://openrouter.ai/api/v1"
export EVALRING_MODEL="anthropic/claude-sonnet-4"
```

**A self-hosted or gateway endpoint** — anything that speaks the
chat-completions API.

```bash
export EVALRING_API_KEY="internal-token"
export EVALRING_BASE_URL="https://llm-gateway.internal/v1"
export EVALRING_MODEL="llama-3.3-70b"
```

**Ollama, locally.** It ignores the key but the SDK requires one to be present.

```bash
export EVALRING_API_KEY="ollama"
export EVALRING_BASE_URL="http://localhost:11434/v1"
export EVALRING_MODEL="llama3.3"
```

**Two providers in one script** — pass credentials explicitly when the process
needs more than one endpoint, for instance a cheap judge scoring an expensive
agent.

```python
agent = OpenAIAgent(name="candidate", model_name="gpt-4o", api_key=OPENAI_KEY)
judge = OpenAIJudge(
    model_name="anthropic/claude-sonnet-4",
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
)
```

## Using a .env file

Copy [`.env.example`](../.env.example) to `.env` and fill it in. `.env` is
gitignored.

The library does **not** load `.env` on import — that would mean importing
EvalRing silently mutates your process environment. `run_suite()` loads it, and
example scripts call `load_dotenv()` themselves. In your own code, load it
explicitly before constructing agents:

```python
from dotenv import load_dotenv

load_dotenv()
```

## Resolving credentials in your own code

```python
from EvalRing import resolve_credentials

creds = resolve_credentials()
print(creds.provider)  # "evalring", "openai", "openrouter", "radium", "none"
print(creds.source)  # "$EVALRING_API_KEY" — safe to log
print(creds.base_url)  # None means the OpenAI default
key = creds.require_key()  # raises MissingCredentialsError with the variable list
```

`resolve_credentials()` accepts an `env=` mapping, which makes configuration
logic testable without touching the real environment.

## Keeping keys out of artifacts

- `evalring check` prints which variable supplied the key, never the value. A
  test asserts this.
- `LLMJudge.get_info()` strips `api_key` from the recorded configuration.
- Run artifacts under `_EvalRing/` contain prompts and model responses but no
  credentials. They do contain your dataset text — see
  [SECURITY.md](../SECURITY.md).
