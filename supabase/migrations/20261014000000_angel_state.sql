-- Angel Filter's shared state (see angel_filter/state.py).
--
-- The app also runs these statements itself on first connection, so it works
-- with any Postgres, with or without this migration. Supabase's GitHub
-- integration applies this file on merge to main, so the schema is recorded
-- in the repository and visible to anyone reading it. Every statement is
-- idempotent, so running both is harmless.
--
-- state._MIGRATION must equal every *_angel_state*.sql migration combined, in
-- order; tests/test_supabase_migrations.py fails if they drift. To change the
-- schema, add a NEW migration file (never edit one that has been applied) and
-- append the same statements to state._MIGRATION.
CREATE SCHEMA IF NOT EXISTS angel_state;
CREATE TABLE IF NOT EXISTS angel_state.kv (
    key        text PRIMARY KEY,
    value      jsonb NOT NULL,
    expires_at timestamptz
);
CREATE INDEX IF NOT EXISTS kv_expires_at ON angel_state.kv (expires_at)
    WHERE expires_at IS NOT NULL;
CREATE TABLE IF NOT EXISTS angel_state.hits (
    key text        NOT NULL,
    at  timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS hits_key_at ON angel_state.hits (key, at);
ALTER TABLE angel_state.kv   ENABLE ROW LEVEL SECURITY;
ALTER TABLE angel_state.hits ENABLE ROW LEVEL SECURITY;
