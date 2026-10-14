"""The supabase/ folder: migration files that match the app, and a safe config.

Supabase's GitHub integration applies supabase/migrations/ to the production
database on merge to main. The app also creates the same schema itself (so it
runs on any Postgres), which means there are two copies of one schema. These
tests keep them identical and check that applying the migration first, then
the app's own setup, works on a clean database.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from angel_filter.state import _MIGRATION, PostgresStore

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = sorted((ROOT / "supabase" / "migrations").glob("*_angel_state*.sql"))
DB_URL = os.getenv("ANGEL_TEST_DATABASE_URL")


def _statements(sql: str) -> list[str]:
    """SQL with comments removed, split into whitespace-normalised statements."""
    sql = re.sub(r"--[^\n]*", "", sql)
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


def test_there_is_an_angel_state_migration():
    assert MIGRATIONS, "supabase/migrations/*_angel_state*.sql is missing"


def test_migration_names_use_supabase_timestamps():
    for path in MIGRATIONS:
        assert re.fullmatch(r"\d{14}_[a-z0-9_]+\.sql", path.name), path.name


def test_migrations_match_what_the_app_creates():
    from_files = [s for path in MIGRATIONS for s in _statements(path.read_text())]
    assert from_files == _statements(_MIGRATION), (
        "state._MIGRATION and supabase/migrations/*_angel_state*.sql have drifted; "
        "append new migration statements to _MIGRATION"
    )


def test_config_keeps_the_data_api_off():
    tomllib = pytest.importorskip("tomllib")   # Python 3.11+
    config = tomllib.loads((ROOT / "supabase" / "config.toml").read_text())
    assert config["api"]["enabled"] is False


@pytest.mark.skipif(not DB_URL, reason="set ANGEL_TEST_DATABASE_URL to run Postgres tests")
@pytest.mark.asyncio
async def test_the_migration_then_the_app_work_on_a_clean_database():
    """The production order: Supabase applies the file, then the app connects."""
    import asyncpg

    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS angel_state CASCADE")
        for path in MIGRATIONS:
            await conn.execute(path.read_text())
            await conn.execute(path.read_text())   # applying twice is harmless
    finally:
        await conn.close()

    store = PostgresStore(DB_URL, fallback=False)   # runs its own setup on top
    await store.set("angel:k", {"ok": True})
    assert await store.get("angel:k") == {"ok": True}
