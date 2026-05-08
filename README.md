# Angel Filter

A local proxy that asks multiple hosted AI models for answers, normalizes their
responses, and returns the ranked results through the existing FastAPI/NLIP
server.

The current default fan-out calls:

- Claude
- OpenAI
- Gemini

The project still keeps the orchestrator/ranker structure, so the API-provider
setup is separate from any later judging or ranking changes.

## How It Works

```text
User question
  -> server.py (/query or NLIP message)
  -> Orchestrator
  -> ClaudeProvider, OpenAIProvider, GeminiProvider
  -> ProviderResult list
  -> Ranker
  -> JSON response / NLIP text response
```

Each AI provider returns one normalized result:

```json
{
  "title": "Claude answer",
  "snippet": "Model answer text...",
  "url": null,
  "provider": "claude",
  "score": 0.82,
  "rationale": "...",
  "sponsored": false
}
```

## Setup

Prereqs: Python 3.10+, Poetry, and API keys for the model providers you want to
call.

```powershell
cd C:\Users\pc\Desktop\NLPproject\NLIP-Project_ib
poetry install
```

Set API keys in the same terminal that will run the server:

```powershell
$env:ANTHROPIC_API_KEY="..."
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
```

Optional model overrides:

```powershell
$env:CLAUDE_MODEL="claude-sonnet-4-20250514"
$env:OPENAI_MODEL="gpt-5.4-mini"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

You can also use model presets instead of exact model names:

```powershell
$env:CLAUDE_MODEL_PRESET="fast"
$env:OPENAI_MODEL_PRESET="low_cost"
$env:GEMINI_MODEL_PRESET="fast_free_tier"
```

Run the server:

```powershell
poetry run python -m angel_filter.server
```

Then open:

- <http://localhost:8000> for the simple UI
- <http://localhost:8000/docs> for Swagger
- <http://localhost:8000/health> for provider status

## Providers

Hosted model API code lives in:

```text
angel_filter/providers/ai_models.py
```

That file contains:

- `AIAnswerProvider`: shared API-key, HTTP POST, and normalization helper
- `MODEL_OPTIONS`: model presets for Claude, OpenAI, and Gemini
- `ClaudeProvider`: calls Anthropic Messages API
- `OpenAIProvider`: calls OpenAI Responses API
- `GeminiProvider`: calls Gemini `generateContent`

Current presets:

| Provider | Preset | Model |
|---|---|---|
| Claude | `default` | `claude-sonnet-4-20250514` |
| Claude | `fast` | `claude-3-5-haiku-20241022` |
| OpenAI | `default` | `gpt-5.4-mini` |
| OpenAI | `low_cost` | `gpt-5-mini` |
| OpenAI | `legacy_low_cost` | `gpt-4.1-mini` |
| Gemini | `default` | `gemini-2.5-flash` |
| Gemini | `free_tier` | `gemini-2.5-flash` |
| Gemini | `fast_free_tier` | `gemini-2.5-flash-lite` |

Offline/free utility providers live separately:

- `DuckDuckGoProvider`: no-key search provider, not registered by default
- `MockProvider`: deterministic local results for tests and offline runs

If a key is missing or one API call fails, the orchestrator records that
provider in `providers_failed` and still returns any successful provider
answers.

## Useful Files

```text
angel_filter/
  server.py             # FastAPI/NLIP routes and provider registration
  orchestrator.py       # parallel provider fan-out and failure isolation
  ranker.py             # existing ranking layer
  providers/
    ai_models.py        # Claude/OpenAI/Gemini API calls
    base.py             # provider interface and ProviderResult shape
    duckduckgo.py       # free/no-key DuckDuckGo provider
    mock.py             # test/offline provider
static/
  index.html            # simple local UI
tests/
  test_orchestrator.py  # existing orchestrator tests
```

## Tests

```powershell
poetry run pytest
```

The current tests use the mock provider and do not call live AI APIs.

## Notes

- API keys are read from environment variables only. Do not hardcode secrets.
- The default server provider list is in `_build_orchestrator()` in
  `angel_filter/server.py`.
- Ollama/ranking behavior is intentionally separate from the provider API setup.
