# Provider: Google Gemini

Native Google Gemini models (e.g. `gemini-3.5-flash`), routed through LiteLLM's
`gemini/` provider.

## Configuration

| Setting | Value |
|---|---|
| Env key | `GEMINI_API_KEY` (LiteLLM reads this for `gemini/*` models) |
| Base URL | native (none required) |
| LiteLLM model id | `gemini/<model>` (e.g. `gemini/gemini-3.5-flash`) |

## Notes

- The project also keeps `GOOGLE_API_KEY`; mirror its value into `GEMINI_API_KEY`
  so LiteLLM's Gemini provider picks it up (both are set in `.env`).
- Gemini 3+ accepts `temperature`/`top_p`/`top_k` but warns they are deprecated;
  this is harmless. `drop_params` handles anything unsupported.
- Gemini "thinking" tokens are normalised by LiteLLM into
  `usage.completion_tokens_details.reasoning_tokens` and captured automatically.

## Model-list entry

```json
{ "name": "gemini-3.5-flash", "litellm_model": "gemini/gemini-3.5-flash",
  "provider": "gemini", "api_key_env": "GEMINI_API_KEY", "reasoning_effort": "medium" }
```
