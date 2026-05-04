import re

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


# ---------- FETCH ----------

def fetch_html(url: str) -> str:
    """
    Fetch raw HTML from *url*.
    Returns an empty string on any network / HTTP error so callers never
    receive None.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(
            url, headers=headers, timeout=5, allow_redirects=True
        )
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


# ---------- FORM ANALYSIS ----------

def detect_login_form(soup: BeautifulSoup) -> int:
    """
    Return 1 when any <form> on the page looks like a login / credential
    harvesting form, 0 otherwise.

    Fixes vs original
    -----------------
    * Guard against soup=None moved to the top.
    * Avoids shadowing the built-in `input` name.
    * Slightly tighter logic: a form must have a password field OR a
      username-style text/email field with few total inputs.
    """
    if not soup:
        return 0

    for form in soup.find_all("form"):
        fields = form.find_all("input")

        has_password = False
        has_username = False

        for field in fields:
            input_type  = (field.get("type")        or "").lower()
            name        = (field.get("name")        or "").lower()
            placeholder = (field.get("placeholder") or "").lower()

            if input_type == "password":
                has_password = True

            if input_type in ("text", "email"):
                if any(k in name or k in placeholder
                       for k in ("user", "email", "login")):
                    has_username = True

        if has_password:
            return 1

        if has_username and len(fields) <= 5:
            return 1

    return 0


# ---------- LINK ANALYSIS ----------

def _root_domain(domain: str) -> str:
    """Strip leading 'www.' for a fair same-domain comparison."""
    return domain[4:] if domain.startswith("www.") else domain


def external_links_ratio(soup: BeautifulSoup, base_url: str) -> float:
    """
    Ratio of external hyperlinks to all valid hyperlinks on the page.

    Fixes vs original
    -----------------
    * Pre-compile the set of ignorable prefixes for clarity.
    * Handles urljoin / urlparse exceptions individually so one bad
      href doesn't abort the whole loop.
    * Returns 0.0 (float) consistently.
    """
    if not soup:
        return 0.0

    _SKIP_PREFIXES = ("#", "javascript:", "mailto:", "tel:", "data:")

    base_domain = _root_domain(urlparse(base_url).netloc.lower())

    external = 0
    valid_links = 0

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not href or any(href.lower().startswith(p) for p in _SKIP_PREFIXES):
            continue

        try:
            full_url = urljoin(base_url, href)
            link_domain = _root_domain(urlparse(full_url).netloc.lower())
        except Exception:
            continue

        if not link_domain:
            continue

        valid_links += 1
        if link_domain != base_domain:
            external += 1

    return external / valid_links if valid_links else 0.0


# ---------- REDIRECT ----------

def detect_redirect(html: str) -> int:
    """
    Return 1 if a meta-refresh redirect is present in *html*.

    Fixes vs original
    -----------------
    * Accepts a plain string (not a soup object) – consistent with how
      feature_extractor.py calls it.
    * Adds a JS-based redirect heuristic (window.location assignments).
    """
    if not html:
        return 0

    soup = BeautifulSoup(html, "html.parser")

    # HTML meta refresh
    if soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)}):
        return 1

    # Common JS redirect patterns
    _JS_REDIRECT_RE = re.compile(
        r"(window\.location|location\.href|location\.replace)\s*[=(]",
        re.IGNORECASE,
    )
    for script in soup.find_all("script"):
        content = script.get_text()
        if content and _JS_REDIRECT_RE.search(content):
            return 1

    return 0


# ---------- INPUT COUNTING ----------

def count_inputs(soup: BeautifulSoup) -> int:
    """Total number of <input> elements on the page."""
    return len(soup.find_all("input")) if soup else 0


def count_passwords(soup: BeautifulSoup) -> int:
    """Number of password-type <input> elements on the page."""
    if not soup:
        return 0
    return len(soup.find_all("input", {"type": "password"}))


# ---------- JS OBFUSCATION ----------

_SUSPICIOUS_PATTERNS = (
    "eval(",
    "unescape(",
    "atob(",
    "function(",       # normalised – original had capital-F mismatch
    "settimeout(",
    "setinterval(",
    "decodeuricomponent(",
)

# Matches base64-like tokens (≥ 50 chars, alphanumeric + +/=)
_B64_RE = re.compile(r"[A-Za-z0-9+/=]{50,}")


def detect_obfuscated_js(soup: BeautifulSoup) -> int:
    """
    Return a capped score (0–10) representing the likelihood that one or
    more <script> blocks contain obfuscated JavaScript.

    Fixes vs original
    -----------------
    * Suspicious pattern matching is now case-insensitive via .lower().
    * 'Function(' → normalised to 'function(' after lowercasing so it
      was actually matched in the original (capitalised comparison on a
      lower-cased string always failed).
    * Base64 detection uses a compiled regex instead of a generator that
      re-checks each character individually.
    * Per-script scoring is capped at 4 to prevent a single enormous
      script from dominating the total.
    """
    if not soup:
        return 0

    total_score = 0

    for script in soup.find_all("script"):
        content = (script.get_text() or "").lower().strip()
        if not content:
            continue

        script_score = 0

        # 1. Suspicious function calls
        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern in content:
                script_score += 1

        # 2. Base64-like long token
        if _B64_RE.search(content):
            script_score += 1

        # 3. Heavy percent-encoding
        if content.count("%") > 30:
            script_score += 1

        # 4. Dense semicolons (minified / obfuscated code)
        if len(content) > 300 and content.count(";") > 80:
            script_score += 1

        total_score += min(script_score, 4)   # cap per-script contribution

    return min(total_score, 10)