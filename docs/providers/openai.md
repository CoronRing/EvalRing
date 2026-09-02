# Provider: OpenAI

Native OpenAI models (GPT-5 family, o-series, GPT-4o, …), reached directly.

## Configuration

| Setting | Value |
|---|---|
| Env key | `EVALRING_API_KEY` (preferred), or `OPENAI_API_KEY` |
| Base URL | native (none required) |
| LiteLLM model id | `openai/<model>` (e.g. `openai/gpt-5.5`, `openai/gpt-5.4-mini`) |

## Notes

- Reasoning models (GPT-5 family, o-series) honour `reasoning_effort`
  (`low` / `medium` / `high`) and report hidden thinking usage via
  `usage.completion_tokens_details.reasoning_tokens` — captured automatically
  (see [litellm.md](litellm.md) for how the agent records reasoning tokens).
- Reasoning models can stay **silent for a long time** before the first visible
  token; keep the request timeout generous (`OPENAI_REQUEST_TIMEOUT_S`, or the
  runner's `--request-timeout-s`).
- Some reasoning models reject a custom `temperature`; `drop_params` (enabled by
  the LiteLLM transport) drops it silently rather than erroring.

## Model-list entry

```json
{ "name": "gpt-5.5", "litellm_model": "openai/gpt-5.5", "provider": "openai",
  "api_key_env": "OPENAI_API_KEY", "reasoning_effort": "medium" }
```
