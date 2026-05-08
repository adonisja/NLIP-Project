# Angel Filter Notes

This project is a Python/FastAPI proxy that calls multiple hosted AI models,
normalizes their answers, and returns ranked results.

## Current Provider Setup

The active providers are registered in `angel_filter/server.py`:

- `ClaudeProvider`
- `OpenAIProvider`
- `GeminiProvider`

All hosted model API logic lives in `angel_filter/providers/ai_models.py`.
Provider classes must return `ProviderResult` objects from
`angel_filter/providers/base.py`.

## Environment Variables

Required for live API calls:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Optional model overrides:

- `CLAUDE_MODEL`
- `OPENAI_MODEL`
- `GEMINI_MODEL`
- `CLAUDE_MODEL_PRESET`
- `OPENAI_MODEL_PRESET`
- `GEMINI_MODEL_PRESET`

## Conventions

- Do not hardcode secrets.
- Keep provider HTTP/API code in `providers/ai_models.py` unless there is a
  strong reason to split it again.
- Keep no-key providers separate from hosted model API code.
- Keep the FastAPI fallback path in `server.py` working.
- Tests should not call live hosted APIs.
- `MockProvider` is kept for local/offline tests.

## Commands

```powershell
poetry install
poetry run python -m angel_filter.server
poetry run pytest
```
