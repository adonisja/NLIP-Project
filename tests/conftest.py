"""Shared pytest setup.

`angel_filter.server` builds the orchestrator at import time
(`ORCHESTRATOR = _build_orchestrator()`), which raises RuntimeError if no
provider is configured. Any test that imports the server module therefore
needs at least one provider present in the environment *before collection*.

CI runs with no API keys and no .env, so we register the one provider that
needs neither — Ollama — by setting OLLAMA_MODEL. This only makes
_build_orchestrator register OllamaProvider; it opens no connection at import,
and tests that touch the orchestrator mock it, so Ollama is never actually
called. Locally a developer's real .env is left untouched (setdefault).

This keeps the suite deterministic and network-free per the project's testing
rule, regardless of what is or isn't set on the machine running it.
"""

import os

# setdefault: don't clobber a real value a developer may have set locally.
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")
