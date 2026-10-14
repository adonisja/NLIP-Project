#!/bin/bash
# Start the dev server on the port the docs use. The package loads .env itself.
exec poetry run uvicorn angel_filter.server:app --reload --port "${PORT:-8005}"
