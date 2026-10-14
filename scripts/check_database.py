"""Check (and optionally fix) the database connection in .env.

    poetry run python scripts/check_database.py                 # diagnose
    poetry run python scripts/check_database.py --set-password  # fix the password

Diagnosing checks the shape of ANGEL_DATABASE_URL (leftover [YOUR-PASSWORD],
quotes, unencoded special characters, the wrong port), then connects and
prints the result. The password is never printed: errors are shown with it
replaced by <pw>.

--set-password asks for the database password with hidden input, encodes it,
writes it into the ANGEL_DATABASE_URL line in .env, and tests the connection.
Copy-pasting a password into the URL by hand is the most common way this goes
wrong: a truncated paste gives "password authentication failed".

Restart the server afterwards: .env is only read at startup.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_LINE = re.compile(r"^ANGEL_DATABASE_URL=(.*)$", re.M)


def find_url(text: str) -> str | None:
    m = _LINE.search(text)
    return m.group(1).strip() if m else None


def problems(url: str) -> list[str]:
    """Mistakes visible in the URL itself, before trying to connect."""
    found = []
    if url[:1] in "'\"":
        found.append("the URL is wrapped in quotes; remove them")
    if "[" in url or "]" in url or "YOUR-PASSWORD" in url:
        found.append("the [YOUR-PASSWORD] placeholder (or its brackets) is still there")
    try:
        parts = urlsplit(url.strip("'\""))
        port = parts.port
    except ValueError:
        return found + ["the URL can't be parsed; a password with @, / or # needs encoding (use --set-password)"]
    if parts.scheme not in ("postgresql", "postgres"):
        found.append(f"it should start with postgresql://, not {parts.scheme}://")
    if not parts.password:
        found.append("there is no password in it")
    elif re.search(r"[@/#?\s]", unquote(parts.password)) and "%" not in parts.password:
        found.append("the password has special characters that aren't encoded (use --set-password)")
    host = parts.hostname or ""
    if host.startswith("db.") and host.endswith(".supabase.co"):
        found.append("this is Supabase's Direct connection, which needs IPv6; use the Session pooler string")
    if "pooler.supabase.com" in host and port not in (5432, 6543):
        found.append(f"the Supabase pooler uses port 5432 (session) or 6543 (transaction), not {port}")
    return found


def with_password(url: str, password: str) -> str:
    """The same URL with its password replaced (encoded)."""
    p = urlsplit(url)
    port = f":{p.port}" if p.port else ""
    return f"{p.scheme}://{p.username}:{quote(password, safe='')}@{p.hostname}{port}{p.path}"


def redact(message: str, url: str) -> str:
    pw = urlsplit(url).password or ""
    message = message.replace(url, "<database url>")
    for secret in {pw, unquote(pw)} - {""}:
        message = message.replace(secret, "<pw>")
    return message


async def try_connect(url: str) -> tuple[bool, str]:
    import asyncpg

    try:
        conn = await asyncpg.connect(url, timeout=10, statement_cache_size=0)
    except Exception as exc:
        hint = ""
        if "password authentication failed" in str(exc):
            hint = " (wrong password: run with --set-password, or reset it in Supabase)"
        elif isinstance(exc, (OSError, asyncio.TimeoutError)):
            hint = " (can't reach the host: check the host and port, and that the project isn't paused)"
        return False, f"{type(exc).__name__}: {redact(str(exc), url)}{hint}"
    try:
        version = await conn.fetchval("SHOW server_version")
        tables = await conn.fetchval(
            "SELECT count(*) FROM pg_tables WHERE schemaname = 'angel_state'"
        )
    finally:
        await conn.close()
    created = "created" if tables else "not created yet (the app creates them on first use)"
    return True, f"Postgres {version}; angel_state tables {created}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-password", action="store_true", help="prompt for the password and write it into .env")
    args = parser.parse_args()

    if not ENV_FILE.exists():
        print(f"No .env at {ENV_FILE}. Copy .env.example to .env first.")
        return 1
    text = ENV_FILE.read_text()
    url = find_url(text)
    if url is None:
        print("No ANGEL_DATABASE_URL line in .env. Add it (no '#', spaces or quotes) and run this again.")
        return 1

    if args.set_password:
        password = getpass.getpass("Database password (hidden): ")
        print(f"Got {len(password)} characters.")
        if not password:
            print("Empty password; nothing changed.")
            return 1
        url = with_password(url.strip("'\""), password)
        ENV_FILE.write_text(_LINE.sub(lambda _: f"ANGEL_DATABASE_URL={url}", text, count=1))
        print(".env updated.")

    issues = problems(url)
    for issue in issues:
        print(f"Problem: {issue}")
    if issues and not args.set_password:
        return 1

    ok, detail = asyncio.run(try_connect(url))
    print(("CONNECTED: " if ok else "FAILED: ") + detail)
    if ok:
        print("Restart the server so it picks up .env, then check /health for \"reachable\": true.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
