from urllib.parse import urlparse

from bs4 import BeautifulSoup

from FeaturePipeline.logical_helper import (
    is_ip_address, count_subdomains,
    keyword_score, brand_distance,
    compute_entropy, tld_score,
)
from FeaturePipeline.content_helper import (
    fetch_html, detect_obfuscated_js,
    detect_redirect, detect_login_form,
    count_inputs, count_passwords,
    external_links_ratio,
)
from FeaturePipeline.domain_helper import (
    get_domain_age, get_expiry,
    has_whois, has_dns,
    check_ssl, get_cert_duration,
    geo_risk, reputation_score, get_whois_data,
    get_hosting_ip, get_whois_country,
    get_ssl_issuer, get_asn,
    get_registrar,
)


# ─────────────────────────── UTILITIES ──────────────────────────────────────

def safe_execute(fn, default=None):
    """
    Call *fn* and return its result.  On any exception return *default*.

    This intentionally swallows all exceptions so that a single failing
    feature never aborts the entire extraction pipeline.
    """
    try:
        return fn()
    except Exception:
        return default


def normalize_url(url: str) -> str:
    """
    Ensure the URL has a scheme so urlparse() can split netloc correctly.

    Common dataset issues this handles
    -----------------------------------
    * Bare domain:        "example.com"        → "https://example.com"
    * Scheme-relative:   "//example.com/path"  → "https://example.com/path"
    * Already has scheme: unchanged
    * None / non-string:  returns "" so callers can skip gracefully
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if "://" not in url:
        return "https://" + url
    return url


_DOMAIN_EMPTY_FEATURES = {
    "domain_age":           -1,
    "time_to_expiry":       -1,
    "has_whois":             0,
    "has_dns_record":        0,
    "ssl_valid":             0,
    "cert_validity_days":   -1,
    "domain_reputation":     0,
    "hosting_country_risk":  0,
    "hosting_ip":       "UNKNOWN",
    "asn":              "UNKNOWN",
    "registrar":        "UNKNOWN",
    "whois_country":    "UNKNOWN",
    "ssl_issuer":       "UNKNOWN",
    "is_reachable":          0,
    "has_ssl":               0,
    "whois_available":       0,
}


# ─────────────────────────── FEATURE GROUPS ─────────────────────────────────

def extract_lexical_features(url: str) -> dict:
    """
    Fast, purely string-based features — no network calls.

    Fixes vs original
    -----------------
    * num_special_chars now excludes the scheme separator characters
      (://) which are structural, not suspicious.  Without this, every
      https:// URL gets +3 before any real special chars are counted.
      Use the path + query + fragment portion only for special-char
      counting so the metric is meaningful.
    """
    url = normalize_url(url)
    parsed = urlparse(url)
    # path onwards: everything after the netloc
    path_part = parsed.path + (parsed.query or "") + (parsed.fragment or "")

    return {
        "url_length":        len(url),
        "num_dots":          url.count("."),
        "num_slashes":       url.count("/"),
        "num_digits":        sum(c.isdigit() for c in url),
        # Count special chars only in the path/query/fragment section
        "num_special_chars": sum(not c.isalnum() and c not in (".", "/", "-", "_")
                                 for c in path_part),
        "has_https":         int(url.startswith("https://")),
        "has_ip":            int(is_ip_address(url)),
        "has_at_symbol":     int("@" in url),
        "has_dash":          int("-" in url),
        "subdomain_count":   count_subdomains(url),
    }


def advanced_lexical_features(url: str) -> dict:
    """
    Entropy and heuristic-similarity features — still no network calls.

    Fixes vs original
    -----------------
    * path_length was computed as len(url.split('/')) which counts
      components of the *entire* URL including the scheme and netloc.
      Changed to len(urlparse(url).path.split('/')) - 1 for just the
      path depth (subtract 1 for the leading empty string from split).
    """
    url = normalize_url(url)
    parsed = urlparse(url)
    path_depth = max(len(parsed.path.split("/")) - 1, 0)

    return {
        "path_entropy":        compute_entropy(parsed.path),
        "domain_entropy":      compute_entropy(parsed.netloc),
        "suspicious_keywords": keyword_score(url),
        "brand_similarity":    brand_distance(url),
        "tld_risk_score":      tld_score(url),
        "path_length":         path_depth,
    }


def extract_content_features(url: str) -> dict:
    """
    Features derived from fetching and parsing the page HTML.

    Fixes vs original
    -----------------
    * html is now guaranteed to be a string (fetch_html returns "" on
      failure), so the `if not html` guard covers both None and "".
    * BeautifulSoup is constructed once and shared with all helpers that
      accept a soup object, avoiding redundant parses.
    * detect_redirect receives the raw html string (as its signature
      expects), not the soup object.
    """
    _EMPTY = {
        "num_forms":           0,
        "num_iframes":         0,
        "num_anchors":         0,
        "num_scripts":         0,
        "has_login_form":      0,
        "external_link_ratio": 0.0,
        "has_redirect":        0,
        "input_fields":        0,
        "password_fields":     0,
        "js_obfuscation_score": 0,
    }

    html = safe_execute(lambda: fetch_html(normalize_url(url)), "")
    if not html:
        return _EMPTY

    soup = BeautifulSoup(html, "html.parser")

    return {
        "num_forms":            len(soup.find_all("form")),
        "num_iframes":          len(soup.find_all("iframe")),
        "num_anchors":          len(soup.find_all("a")),
        "num_scripts":          len(soup.find_all("script")),
        "has_login_form":       safe_execute(lambda: detect_login_form(soup),            0),
        "external_link_ratio":  safe_execute(lambda: external_links_ratio(soup, url),   0.0),
        "has_redirect":         safe_execute(lambda: detect_redirect(html),              0),  # ← raw html
        "input_fields":         safe_execute(lambda: count_inputs(soup),                 0),
        "password_fields":      safe_execute(lambda: count_passwords(soup),              0),
        "js_obfuscation_score": safe_execute(lambda: detect_obfuscated_js(soup),         0),
    }


def extract_domain_features(url: str) -> dict:
    """
    WHOIS / DNS / SSL / IP metadata features.

    Fixes vs original
    -----------------
    * A single get_whois_data() call is made upfront and reused by every
      downstream helper.  The original did the same but could still make
      duplicate calls if w was None inside a helper.  Now w is always
      passed explicitly.
    * get_hosting_ip() result is validated before being passed to
      get_asn() — avoids a useless RDAP lookup on None.
    * ssl_valid and has_ssl were computing the same thing via two
      separate check_ssl() calls (two TLS handshakes).  Now one call
      result is reused.
    * All metadata fields default to None; normalize_features() will
      convert them to "UNKNOWN" before model ingestion.
    """
    url = normalize_url(url)
    domain = urlparse(url).netloc

    # Guard: if we still have no domain after normalisation, the row is
    # unusable — return safe defaults immediately, no network calls made.
    if not domain:
        return dict(_DOMAIN_EMPTY_FEATURES)

    # One WHOIS call shared across all helpers
    w  = safe_execute(lambda: get_whois_data(domain), None)
    ip = safe_execute(lambda: get_hosting_ip(domain), None)

    # One SSL check shared between ssl_valid and has_ssl
    ssl_result = safe_execute(lambda: check_ssl(domain), 0)

    return {
        # ── numerical ──────────────────────────────────────────────────
        "domain_age":           safe_execute(lambda: get_domain_age(domain, w),    -1),
        "time_to_expiry":       safe_execute(lambda: get_expiry(domain, w),        -1),
        "has_whois":            int(safe_execute(lambda: has_whois(domain, w),    False)),
        "has_dns_record":       int(safe_execute(lambda: has_dns(domain),         False)),
        "ssl_valid":            ssl_result,
        "cert_validity_days":   safe_execute(lambda: get_cert_duration(domain),   -1),
        "domain_reputation":    safe_execute(lambda: reputation_score(domain),     0),
        "hosting_country_risk": safe_execute(lambda: geo_risk(domain),             0),

        # ── metadata ───────────────────────────────────────────────────
        "hosting_ip":    ip,
        "asn":           safe_execute(lambda: get_asn(ip),               None),
        "registrar":     safe_execute(lambda: get_registrar(w, domain),  None),
        "whois_country": safe_execute(lambda: get_whois_country(w, domain), None),
        "ssl_issuer":    safe_execute(lambda: get_ssl_issuer(domain),    None),

        # ── flags ──────────────────────────────────────────────────────
        "is_reachable":      int(ip is not None),
        "has_ssl":           int(ssl_result == 1),
        "whois_available":   int(w is not None),
    }


# ─────────────────────────── NORMALISATION ──────────────────────────────────

_CATEGORICAL_FIELDS = frozenset({
    "hosting_ip", "asn", "registrar", "whois_country", "ssl_issuer",
})


def normalize_features(features: dict) -> dict:
    """
    Replace None / falsy values in categorical metadata fields with the
    sentinel string "UNKNOWN" so downstream encoders never receive NaN.

    Fixes vs original
    -----------------
    * Uses a frozenset for O(1) membership test.
    * Only replaces truly missing values (None or empty string); a
      legitimate value of 0 for a numerical field is preserved.
    """
    for field in _CATEGORICAL_FIELDS:
        if not features.get(field):
            features[field] = "UNKNOWN"
    return features


# ─────────────────────────── PUBLIC API ─────────────────────────────────────

def build_feature_vector(url: str) -> dict:
    """
    Extract all features for *url* and return a single flat dict suitable
    for model inference or storage.

    Feature groups
    --------------
    lexical          → 10 fast string features
    advanced_lexical → 6 entropy / heuristic features
    content          → 10 page-content features  (1 HTTP request)
    domain           → 18 WHOIS / DNS / SSL / IP features
    """
    url = normalize_url(url)   # normalise once; all groups receive a clean URL
    if not url:
        # Completely empty / non-string row — return all-zero safe defaults
        return normalize_features({
            **extract_lexical_features(""),
            **advanced_lexical_features(""),
            **{k: 0  for k in ["num_forms","num_iframes","num_anchors","num_scripts",
                                "has_login_form","has_redirect","input_fields",
                                "password_fields","js_obfuscation_score"]},
            **{"external_link_ratio": 0.0},
            **dict(_DOMAIN_EMPTY_FEATURES),
        })

    features = {
        **extract_lexical_features(url),
        **advanced_lexical_features(url),
        **extract_content_features(url),
        **extract_domain_features(url),
    }
    return normalize_features(features)