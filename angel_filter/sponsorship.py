"""Sponsored-content detection.

The sponsored penalty is the project's thesis, but no live provider reports
ads: the AI providers don't know, and Brave and Google Places return none. So
sponsorship has to be inferred from the result itself. This module does that
after the fan-out, for every provider at once, and records *why* each result
was flagged so the reply can show its reasoning.

The rules are deliberately conservative. A false positive penalises an organic
result, which undermines the claim this project exists to make, so every rule
matches something that only appears when money changed hands:

  - links into ad networks (click redirects, sponsored-content widgets)
  - ad-click IDs and paid-traffic UTM tags in the URL
  - affiliate tracking parameters
  - explicit disclosure text ("Sponsored", "paid partnership", "#ad")

Plain "sponsored by" is NOT a rule: "festival sponsored by local businesses"
is ordinary event copy. Add rules here, each with a test for what it must not
match, rather than in individual providers.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlsplit

from angel_filter.providers.base import ProviderResult

logger = logging.getLogger(__name__)

# Hosts that only serve ads or paid placements. Matched as the host or any
# subdomain of it.
_AD_HOSTS = (
    "doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
    "amazon-adsystem.com",
    "dartsearch.net",          # clickserve.dartsearch.net search-ad redirects
    "taboola.com",
    "outbrain.com",
)

# Bing's ad clicks go through bing.com/aclk; the rest of bing.com is organic.
_AD_PATHS = (("bing.com", "/aclk"),)

# Click IDs that ad platforms append to paid clicks only.
_AD_CLICK_PARAMS = {"gclid", "gclsrc", "dclid", "wbraid", "gbraid", "msclkid"}

# utm_medium values that mean the visit was paid for.
_PAID_MEDIUMS = {
    "cpc", "ppc", "cpm", "paid", "paidsearch", "paid_search", "paid-search",
    "paidsocial", "paid_social", "paid-social", "display", "sponsored", "affiliate",
}

# Affiliate-network tracking parameters (CJ, Impact, Rakuten, Awin, generic).
_AFFILIATE_PARAMS = {
    "cjevent", "irclickid", "irgwc", "ranmid", "raneaid", "ransiteid",
    "clickref", "aff_id", "affid", "affiliate_id", "aff_sub",
}

# Amazon's associate tag is "tag=", a name too generic to trust elsewhere.
_AMAZON_HOST = re.compile(r"(^|\.)amazon\.[a-z.]+$")

# A label at the very start of a title or snippet: "Sponsored · ...",
# "Ad: ...", "[Ad] ...", "Promoted - ...". Anchored so "Adams Diner" and
# "Advertising agency" don't match.
_LABEL = re.compile(
    r"^\s*(?:[\[(]\s*(sponsored|promoted|ad)\s*[\])]"        # "[Ad] ...", "(Sponsored) ..."
    # "Ad: ...", "Sponsored · ...", "Ad - ...". A hyphen counts only with a
    # space after it, so "Ad-free lunch spots" is not a label.
    r"|(sponsored|promoted|ad)\s*(?:[·:|–—]|-\s))",
    re.IGNORECASE,
)

# Disclosure phrases anywhere in the text.
_DISCLOSURES = re.compile(
    r"""
    \bsponsored\s+(?:content|post|listing|result|placement)\b
  | \bpaid\s+(?:partnership|promotion|placement|advertisement)\b
  | (?<![\w#])\#(?:ad|sponsored|spon)\b
  | \b(?:contains|includes|uses)\s+affiliate\s+links?\b
  | \bwe\s+(?:may\s+)?(?:earn|receive)\s+(?:a\s+)?(?:small\s+)?commission\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def sponsorship_reason(result: ProviderResult) -> str | None:
    """Why `result` looks sponsored, or None if nothing suggests it is."""
    reason = _url_reason(result.url) if result.url else None
    if reason:
        return reason
    for text in (result.title, result.snippet):
        label = _LABEL.match(text or "")
        if label:
            return f'labelled "{label.group(1) or label.group(2)}"'
        m = _DISCLOSURES.search(text or "")
        if m:
            return f'disclosure in text: "{m.group(0).strip()}"'
    return None


def _url_reason(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()

    for ad_host in _AD_HOSTS:
        if host == ad_host or host.endswith("." + ad_host):
            return f"links through ad network {ad_host}"
    for ad_host, path in _AD_PATHS:
        if (host == ad_host or host.endswith("." + ad_host)) and parts.path.startswith(path):
            return f"links through {ad_host}{path} (ad click)"

    params = {k.lower(): v for k, v in parse_qs(parts.query).items()}
    click_ids = _AD_CLICK_PARAMS & params.keys()
    if click_ids:
        return f"ad-click tracking ({sorted(click_ids)[0]})"
    medium = (params.get("utm_medium") or [""])[0].strip().lower()
    if medium in _PAID_MEDIUMS:
        return f"paid-traffic tag (utm_medium={medium})"
    affiliate = _AFFILIATE_PARAMS & params.keys()
    if affiliate:
        return f"affiliate tracking ({sorted(affiliate)[0]})"
    if "tag" in params and _AMAZON_HOST.search(host):
        return "Amazon affiliate tag"
    return None


def flag_sponsored(results: list[ProviderResult]) -> int:
    """Mark results that look sponsored. Returns how many were newly flagged.

    A provider that already reported a result as sponsored is trusted and left
    alone. Detection only ever adds flags; it never clears one.
    """
    flagged = 0
    for r in results:
        if r.sponsored:
            continue
        reason = sponsorship_reason(r)
        if reason:
            r.sponsored = True
            r.sponsored_reason = reason
            flagged += 1
            logger.info("Flagged %r from %s as sponsored: %s", r.title, r.provider, reason)
    return flagged
