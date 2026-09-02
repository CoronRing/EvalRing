# Transport: LiteLLM (and the Radium gateway)

LiteLLM is EvalRing's default model transport. One call style reaches every
provider; the provider is selected by the model-id prefix, and OpenAI-compatible
gateways (like **Radium**) are reached by adding an explicit `api_base`.

Set `EVALRING_LLM_TRANSPORT=litellm` (the default). If LiteLLM is not installed,
the core agent falls back to the OpenAI SDK.

## How routing works

The core `OpenAIAgent` calls `litellm.completion(model=..., api_key=..., api_base=..., ...)`:

- `openai/<model>` → OpenAI (or any OpenAI-compatible endpoint when `api_base` is set)
- `gemini/<model>` → Google Gemini (see [gemini.md](gemini.md))
- `openrouter/<model>` → OpenRouter (see [openrouter.md](openrouter.md))

Shared behaviour the transport enables for every provider:

- **`reasoning_effort`** (`low`/`medium`/`high`) forwarded when set.
- **`drop_params = True`** — params a given model doesn't support (e.g.
  `reasoning_effort` on a non-reasoning model, `temperature` on some reasoning
  models) are dropped instead of erroring, so a mixed suite never hard-fails.
- **Reasoning-token capture** — hidden "thinking" tokens are read from
  `usage.completion_tokens_details.reasoning_tokens` (falling back to an estimate
  from streamed `reasoning_content`). Exposed as `reasoning_tokens`,
  `answer_tokens`, `reasoning_chars` per sample and aggregated
  (`total/avg_reasoning_tokens`, `reasoning_token_fraction`).
- **Empty/timeout responses → errors** — a completion that streams only hidden
  reasoning and no visible answer (e.g. cut at the request timeout) is reported
  as an explicit error, not a silent empty answer.

## Radium gateway

Radium is an OpenAI-compatible gateway serving custom models (`hal-1.0`,
`tycho-1.0`, `clarke-1.0`). It has **no dedicated SDK** — it is reached purely as
an OpenAI-compatible endpoint via LiteLLM, so it does **not** need its own
provider doc.

| Setting | Value |
|---|---|
| Env key | `EVALRING_API_KEY` (preferred), or the `RADIUM_API_KEY` alias |
| Base URL | `EVALRING_BASE_URL`, or `RADIUM_BASE_URL` (which defaults to `https://api.radium.cloud/v1` when `RADIUM_API_KEY` is the selected credential) |
| LiteLLM model id | `openai/<model>` with `api_base` set to the gateway (e.g. `openai/hal-1.0`) |

This is the general pattern for **any** OpenAI-compatible gateway — an internal
proxy, vLLM, Ollama: point `EVALRING_BASE_URL` at it and use an `openai/`-prefixed
model id. The `RADIUM_*` variables are recognized as an alias so existing
environments keep working; see [../CONFIGURATION.md](../CONFIGURATION.md).

### Operational notes

- The gateway can **drop long, highly concurrent streams** ("Connection error").
  Keep per-model concurrency modest for heavy reasoners (the HLE model list caps
  Radium models at `max_workers: 2`); connection/5xx errors are retried patiently
  by the evaluator (`classify_error` → transient).
- Radium models are strong reasoners and may stream tens of thousands of hidden
  reasoning tokens; give them a generous `--request-timeout-s` and no output cap,
  or accept timeout errors on the hardest items.

### Model-list entry

```json
{ "name": "hal-1.0", "litellm_model": "openai/hal-1.0", "provider": "radium",
  "api_base_env": "RADIUM_BASE_URL", "api_key_env": "RADIUM_API_KEY",
  "reasoning_effort": "medium", "max_workers": 2 }
```

## Timeouts

`OPENAI_REQUEST_TIMEOUT_S` (default 120s) bounds each request; runners expose it
as `--request-timeout-s`. Note LiteLLM does not always enforce this as a hard cap
on a live stream — a provider's own stream limit may cut first. Either way, a
resulting empty response is captured as an error.
