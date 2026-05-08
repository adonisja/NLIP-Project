# Dev Fallback Mode

`angel_filter/server.py` tries to start as an NLIP-backed FastAPI app. If the
NLIP libraries are not installed, it falls back to a plain FastAPI app with the
same local routes.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `static/index.html` |
| `/query` | POST | Calls the registered providers and returns ranked results |
| `/health` | GET | Shows mode, uptime, and active providers |
| `/metrics` | GET | Prometheus metrics |

## POST /query

Request:

```json
{
  "query": "Explain retrieval augmented generation",
  "preference": "concise and accurate"
}
```

Response shape:

```json
{
  "providers_used": ["claude", "openai", "gemini"],
  "providers_failed": [],
  "results": [
    {
      "title": "Claude answer",
      "snippet": "Answer text...",
      "url": null,
      "provider": "claude",
      "score": 0.73,
      "rationale": "ranking explanation",
      "sponsored": false
    }
  ]
}
```

The fallback mode exists so local API/provider work can continue even when the
NLIP packages are unavailable.
