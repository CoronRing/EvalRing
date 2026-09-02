# Provider: OpenRouter

Aggregator that fronts many vendors' models behind one OpenAI-compatible API.
Used by the suicide-detection example's model list; reachable through the core agent like any other OpenAI-compatible endpoint.

## Configuration

| Setting | Value |
|---|---|
| Env key | `EVALRING_API_KEY` (preferred), or `OPENROUTER_API_KEY` / `OPEN_ROUTER_KEY` |
| Base URL | `https://openrouter.ai/api/v1`. Applied automatically when an `OPENROUTER_API_KEY` / `OPEN_ROUTER_KEY` is the selected credential; set `EVALRING_BASE_URL` explicitly when using `EVALRING_API_KEY`. |
| Model id | OpenRouter ids, e.g. `openai/gpt-4o-mini`, `meta-llama/llama-3.3-70b-instruct` |
| LiteLLM model id | `openrouter/<model>` when routing explicitly through LiteLLM's OpenRouter provider |
| Optional | `OPEN_ROUTER_MODEL` (default model override) |

## Notes

- Credential precedence when no key is passed explicitly: `EVALRING_API_KEY` →
  `OPENAI_API_KEY` → `OPENROUTER_API_KEY` → `OPEN_ROUTER_KEY` → `RADIUM_API_KEY`.
  Full table in [../CONFIGURATION.md](../CONFIGURATION.md).
- Live per-model pricing is fetched from `https://openrouter.ai/api/v1/models`
  by the suicide-detection reporter for cost estimates.
- Model lists generated for OpenRouter carry an `openrouter_id` field
  (see `src/EvalRing/utils/generate_model_list.py`).
