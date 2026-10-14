"""scripts/check_database.py: the URL checks, password rewrite and redaction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_database.py"
_spec = importlib.util.spec_from_file_location("check_database", _PATH)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# Fake connection strings are assembled at runtime from obviously fake parts,
# so no file contains a literal "scheme://user:password@host" credential for
# secret scanners (GitGuardian) to flag.
FAKE_PW = "not-a-real-password"


def _url(user="postgres.exampleref", password=FAKE_PW,
         host="aws-1-us-west-2.pooler.supabase.com", port=5432, scheme="postgresql") -> str:
    auth = f"{user}:{password}" if password else user + ":"
    return f"{scheme}://{auth}@{host}:{port}/postgres"


GOOD = _url()


def test_a_correct_pooler_url_has_no_problems():
    assert check.problems(GOOD) == []


@pytest.mark.parametrize("url, expected", [
    (_url(password="[YOUR-PASSWORD]"), "placeholder"),
    (f'"{GOOD}"', "quotes"),
    (_url(password=""), "no password"),
    (_url(user="postgres", host="db.exampleref.supabase.co"), "IPv6"),
    (_url(port=6000), "port"),
    (_url(scheme="mysql"), "postgresql://"),
])
def test_common_mistakes_are_named(url, expected):
    assert any(expected in p for p in check.problems(url)), check.problems(url)


def test_with_password_encodes_special_characters():
    url = check.with_password(GOOD, "p@ss/w#rd")
    assert ":p%40ss%2Fw%23rd@" in url
    assert url.endswith("pooler.supabase.com:5432/postgres")
    assert check.problems(url) == []


def test_redact_hides_the_password_both_ways():
    url = check.with_password(GOOD, "p@ss")
    msg = f"failed for {url} with p%40ss and p@ss"
    out = check.redact(msg, url)
    assert "p@ss" not in out and "p%40ss" not in out


def test_find_url_reads_only_the_real_line():
    text = "# ANGEL_DATABASE_URL=commented\nOTHER=1\nANGEL_DATABASE_URL=" + GOOD + "\n"
    assert check.find_url(text) == GOOD
    assert check.find_url("# ANGEL_DATABASE_URL=nope\n") is None
