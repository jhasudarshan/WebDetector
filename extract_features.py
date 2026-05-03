"""
extract_features.py  –  Extract the same 61 features from a new URL
===================================================================
Columns replicated from  final_dataset_with_all_features_v3.1.csv
(everything EXCEPT: url, type, label)

Usage:
    python extract_features.py "https://example.com"
    python extract_features.py "https://example.com" --csv output.csv
    python extract_features.py --batch urls.txt --csv output.csv
"""

import argparse
import csv
import ipaddress
import math
import re
import socket
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse, urljoin, parse_qs

import requests
from bs4 import BeautifulSoup

try:
    import whois
except ImportError:
    whois = None

# ─── Constants ─────────────────────────────────────────────────────
SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rb.gy",
    "bl.ink", "clck.ru", "shorte.st", "tiny.cc", "lnkd.in",
]

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "limited time",
    "expire", "suspended", "verify now", "confirm now",
    "alert", "warning", "attention", "important",
]

SECURITY_WORDS = [
    "secure", "security", "safe", "protect", "privacy",
    "encrypt", "ssl", "verified", "trusted", "authenticate",
]

BRAND_NAMES = [
    "google", "facebook", "amazon", "paypal", "apple",
    "microsoft", "netflix", "instagram", "twitter", "linkedin",
    "chase", "wellsfargo", "bankofamerica", "citibank", "usbank",
    "dropbox", "yahoo", "outlook", "office365", "icloud",
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".work", ".date", ".racing", ".win", ".bid",
    ".stream", ".download", ".loan", ".click", ".link",
    ".trade", ".review", ".science", ".party",
]

HACKED_TERMS = [
    "wp-admin", "wp-content", "wp-includes", "wp-login",
    "administrator", "admin", "shell", "backdoor",
    "hack", "c99", "r57", "webshell",
]

SUSPICIOUS_EXTENSIONS = [".exe", ".zip", ".scr", ".bat", ".cmd", ".ps1", ".js"]


# ═══════════════════════════════════════════════════════════════════
#  URL helpers
# ═══════════════════════════════════════════════════════════════════
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split(":")[0])
        return True
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════════════════
#  1. LEXICAL / URL-STRUCTURE FEATURES  (Cols 3-23)
# ═══════════════════════════════════════════════════════════════════
def _extract_lexical(url: str) -> dict:
    """Pure string analysis – no network required."""
    parsed = urlparse(url)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full = url  # raw URL string

    # Special-char counts
    features = {
        "url_len":              len(full),
        "@":                    full.count("@"),
        "?":                    full.count("?"),
        "-":                    full.count("-"),
        "=":                    full.count("="),
        ".":                    full.count("."),
        "#":                    full.count("#"),
        "%":                    full.count("%"),
        "+":                    full.count("+"),
        "$":                    full.count("$"),
        "!":                    full.count("!"),
        "*":                    full.count("*"),
        ",":                    full.count(","),
        "//":                   full.count("//"),
        "digits":               sum(c.isdigit() for c in full),
        "letters":              sum(c.isalpha() for c in full),
    }

    # domain (netloc without port)
    domain = netloc.split(":")[0]
    features["domain"] = domain

    # abnormal_url: 1 if domain NOT found inside the URL path+query
    features["abnormal_url"] = 0 if domain and domain in full else 1

    # https flag
    features["https"] = 1 if parsed.scheme == "https" else 0

    # shortening service
    features["Shortining_Service"] = int(
        any(s in netloc.lower() for s in SHORTENING_SERVICES)
    )

    # IP address as hostname
    features["having_ip_address"] = int(_is_ip(domain))

    return features


# ═══════════════════════════════════════════════════════════════════
#  2. WEB / CONTENT FEATURES  (Cols 25-39)
# ═══════════════════════════════════════════════════════════════════
def _fetch_page(url: str, timeout: int = 8):
    """Returns (response, soup) or (None, None)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=timeout,
                            allow_redirects=True, verify=False)
        soup = BeautifulSoup(resp.text, "html.parser")
        return resp, soup
    except Exception:
        return None, None


def _extract_web(url: str) -> dict:
    """Features that require fetching the page HTML."""
    resp, soup = _fetch_page(url)

    feats = {
        "scan_date":            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "web_http_status":      0,
        "web_is_live":          0,
        "web_ext_ratio":        0.0,
        "web_unique_domains":   0,
        "web_favicon":          0,
        "web_csp":              0,
        "web_xframe":           0,
        "web_hsts":             0,
        "web_xcontent":         0,
        "web_security_score":   0,
        "web_forms_count":      0,
        "web_password_fields":  0,
        "web_hidden_inputs":    0,
        "web_has_login":        0,
        "web_ssl_valid":        0,
    }

    if resp is None or soup is None:
        return feats

    feats["web_http_status"] = resp.status_code
    feats["web_is_live"] = 1 if 200 <= resp.status_code < 400 else 0

    # --- External-resource ratio ---
    parsed_base = urlparse(url)
    base_domain = (parsed_base.netloc or "").split(":")[0].lower()
    if base_domain.startswith("www."):
        base_domain = base_domain[4:]

    all_links = []
    for tag, attr in [("a", "href"), ("script", "src"), ("link", "href"),
                      ("img", "src"), ("iframe", "src")]:
        for el in soup.find_all(tag, **{attr: True}):
            all_links.append(el[attr])

    external_count = 0
    domains_seen = set()
    for link in all_links:
        try:
            full = urljoin(url, link)
            dom = urlparse(full).netloc.split(":")[0].lower()
            if dom.startswith("www."):
                dom = dom[4:]
            if dom and dom != base_domain:
                external_count += 1
                domains_seen.add(dom)
        except Exception:
            pass

    total_links = max(len(all_links), 1)
    feats["web_ext_ratio"] = round(external_count / total_links, 4)
    feats["web_unique_domains"] = len(domains_seen)

    # Favicon from external domain?
    favicon_tags = soup.find_all("link", rel=lambda x: x and "icon" in " ".join(x).lower())
    for tag in favicon_tags:
        href = tag.get("href", "")
        try:
            fav_dom = urlparse(urljoin(url, href)).netloc.split(":")[0].lower()
            if fav_dom.startswith("www."):
                fav_dom = fav_dom[4:]
            if fav_dom and fav_dom != base_domain:
                feats["web_favicon"] = 1
                break
        except Exception:
            pass

    # Security headers
    headers = {k.lower(): v for k, v in resp.headers.items()}
    sec_score = 0
    if "content-security-policy" in headers:
        feats["web_csp"] = 1
        sec_score += 1
    if "x-frame-options" in headers:
        feats["web_xframe"] = 1
        sec_score += 1
    if "strict-transport-security" in headers:
        feats["web_hsts"] = 1
        sec_score += 1
    if "x-content-type-options" in headers:
        feats["web_xcontent"] = 1
        sec_score += 1
    feats["web_security_score"] = sec_score

    # Form / input analysis
    forms = soup.find_all("form")
    feats["web_forms_count"] = len(forms)
    feats["web_password_fields"] = len(soup.find_all("input", {"type": "password"}))
    feats["web_hidden_inputs"] = len(soup.find_all("input", {"type": "hidden"}))

    # Login form detection
    for form in forms:
        inputs = form.find_all("input")
        has_pw = any((inp.get("type") or "").lower() == "password" for inp in inputs)
        has_user = any(
            (inp.get("type") or "").lower() in ("text", "email")
            and any(k in (inp.get("name") or "").lower() + (inp.get("placeholder") or "").lower()
                    for k in ["user", "email", "login"])
            for inp in inputs
        )
        if has_pw or (has_user and len(inputs) <= 5):
            feats["web_has_login"] = 1
            break

    # SSL validity (scheme-level check; deep check is in domain features)
    feats["web_ssl_valid"] = 1 if parsed_base.scheme == "https" and resp.ok else 0

    return feats


# ═══════════════════════════════════════════════════════════════════
#  3. PHISHING-HEURISTIC FEATURES  (Cols 40-63)
# ═══════════════════════════════════════════════════════════════════
def _extract_phishing_heuristics(url: str) -> dict:
    """Advanced phishing signals derived purely from the URL string."""
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full_lower = url.lower()

    domain = netloc.split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]

    subdomains = domain.split(".")
    domain_no_tld = ".".join(subdomains[:-1]) if len(subdomains) > 1 else domain

    feats = {}

    # --- Text-based heuristics (page title not available, use URL) ---
    feats["phish_urgency_words"] = sum(1 for w in URGENCY_WORDS if w in full_lower)
    feats["phish_security_words"] = sum(1 for w in SECURITY_WORDS if w in full_lower)

    # Brand mentions in URL
    brand_hits = [b for b in BRAND_NAMES if b in full_lower]
    feats["phish_brand_mentions"] = len(brand_hits)

    # Brand hijack: brand name in netloc but NOT the official domain
    feats["phish_brand_hijack"] = 0
    for brand in brand_hits:
        if brand in netloc and not domain.startswith(brand):
            feats["phish_brand_hijack"] = 1
            break

    # Multiple subdomains (≥ 3 parts before TLD)
    feats["phish_multiple_subdomains"] = int(len(subdomains) > 3)

    # Long path
    feats["phish_long_path"] = int(len(path) > 60)

    # Many query parameters
    params = parse_qs(query)
    feats["phish_many_params"] = int(len(params) > 3)

    # Suspicious TLD
    feats["phish_suspicious_tld"] = int(
        any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
    )

    # ── Advanced phishing signals ──
    # Exact brand match in netloc
    feats["phish_adv_exact_brand_match"] = int(
        any(domain_no_tld == b for b in BRAND_NAMES)
    )

    # Brand in subdomain (e.g. paypal.evil.com)
    feats["phish_adv_brand_in_subdomain"] = int(
        any(b in ".".join(subdomains[:-2]) for b in BRAND_NAMES)
        if len(subdomains) > 2 else 0
    )

    # Brand in path
    feats["phish_adv_brand_in_path"] = int(
        any(b in path for b in BRAND_NAMES)
    )

    # Hyphen count in domain
    feats["phish_adv_hyphen_count"] = domain.count("-")

    # Number count in domain
    feats["phish_adv_number_count"] = sum(c.isdigit() for c in domain)

    # Suspicious TLD (advanced, same list)
    feats["phish_adv_suspicious_tld"] = feats["phish_suspicious_tld"]

    # Long domain
    feats["phish_adv_long_domain"] = int(len(domain) > 30)

    # Many subdomains (advanced threshold)
    feats["phish_adv_many_subdomains"] = int(len(subdomains) > 4)

    # Encoded characters in URL
    feats["phish_adv_encoded_chars"] = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    # Path keywords
    path_keywords = ["login", "signin", "verify", "update", "secure",
                     "account", "confirm", "password", "banking", "webscr"]
    feats["phish_adv_path_keywords"] = sum(1 for k in path_keywords if k in path)

    # Redirect indicators in URL
    feats["phish_adv_has_redirect"] = int(
        "redirect" in full_lower or "url=" in full_lower or
        "next=" in full_lower or "rurl=" in full_lower or
        "dest=" in full_lower
    )

    # Many params (advanced)
    feats["phish_adv_many_params"] = int(len(params) > 5)

    # ── Path-level red flags ──
    feats["path_has_hacked_terms"] = int(
        any(t in path for t in HACKED_TERMS)
    )

    feats["suspicious_extension"] = int(
        any(path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS)
    )

    feats["path_underscore_count"] = path.count("_")

    # Government / education domain
    feats["is_gov_edu"] = int(
        domain.endswith(".gov") or domain.endswith(".edu") or
        ".gov." in domain or ".edu." in domain or
        ".ac." in domain or ".mil" in domain
    )

    return feats


# ═══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR – extract ALL features
# ═══════════════════════════════════════════════════════════════════

# Column order exactly as in the CSV (minus url, type, label)
FEATURE_COLUMNS = [
    "url_len", "@", "?", "-", "=", ".", "#", "%", "+", "$", "!", "*", ",", "//",
    "digits", "letters", "domain", "abnormal_url", "https",
    "Shortining_Service", "having_ip_address",
    "scan_date",
    "web_http_status", "web_is_live", "web_ext_ratio", "web_unique_domains",
    "web_favicon", "web_csp", "web_xframe", "web_hsts", "web_xcontent",
    "web_security_score", "web_forms_count", "web_password_fields",
    "web_hidden_inputs", "web_has_login", "web_ssl_valid",
    "phish_urgency_words", "phish_security_words", "phish_brand_mentions",
    "phish_brand_hijack", "phish_multiple_subdomains", "phish_long_path",
    "phish_many_params", "phish_suspicious_tld",
    "phish_adv_exact_brand_match", "phish_adv_brand_in_subdomain",
    "phish_adv_brand_in_path", "phish_adv_hyphen_count",
    "phish_adv_number_count", "phish_adv_suspicious_tld",
    "phish_adv_long_domain", "phish_adv_many_subdomains",
    "phish_adv_encoded_chars", "phish_adv_path_keywords",
    "phish_adv_has_redirect", "phish_adv_many_params",
    "path_has_hacked_terms", "suspicious_extension",
    "path_underscore_count", "is_gov_edu",
]


def extract_all_features(url: str) -> dict:
    """
    Given a raw URL, extract all 61 features that match the CSV columns.
    Web features are fetched concurrently for speed.

    Returns an ordered dict with keys == FEATURE_COLUMNS.
    """
    url = normalize_url(url)

    results = {}

    def _lex():
        return _extract_lexical(url)

    def _web():
        return _extract_web(url)

    def _phish():
        return _extract_phishing_heuristics(url)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_lex): "lex",
            pool.submit(_web): "web",
            pool.submit(_phish): "phish",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result(timeout=15)
            except Exception as e:
                print(f"[WARN] {key} extraction failed: {e}")
                results[key] = {}

    # Merge
    merged = {}
    merged.update(results.get("lex", {}))
    merged.update(results.get("web", {}))
    merged.update(results.get("phish", {}))

    # Reorder to match CSV columns exactly
    ordered = {col: merged.get(col, 0) for col in FEATURE_COLUMNS}
    return ordered


# ═══════════════════════════════════════════════════════════════════
#  CLI  interface
# ═══════════════════════════════════════════════════════════════════
def print_features(url: str, feats: dict):
    print(f"\n{'=' * 60}")
    print(f"  URL: {url}")
    print(f"{'=' * 60}")

    sections = [
        ("URL / Lexical Features", FEATURE_COLUMNS[:21]),
        ("Web / Content Features", FEATURE_COLUMNS[21:37]),
        ("Phishing Heuristics",   FEATURE_COLUMNS[37:]),
    ]

    for title, cols in sections:
        print(f"\n  -- {title} {'-' * (50 - len(title))}")
        for col in cols:
            val = feats.get(col, "N/A")
            print(f"    {col:<35} {val}")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract phishing-detection features for a URL "
                    "(matching final_dataset_with_all_features_v3.1.csv)"
    )
    parser.add_argument("url", nargs="?", help="Single URL to analyse")
    parser.add_argument("--batch", "-b", help="Path to a text file with one URL per line")
    parser.add_argument("--csv", "-o", help="Write results to a CSV file")
    args = parser.parse_args()

    if not args.url and not args.batch:
        parser.print_help()
        sys.exit(1)

    urls = []
    if args.batch:
        with open(args.batch, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = [args.url]

    all_rows = []
    for url in urls:
        print(f"[*] Extracting features for: {url} ...")
        start = time.time()
        feats = extract_all_features(url)
        elapsed = round(time.time() - start, 2)
        print(f"    Done in {elapsed}s")
        print_features(url, feats)
        all_rows.append({"url": url, **feats})

    # Write CSV if requested
    if args.csv:
        fieldnames = ["url"] + FEATURE_COLUMNS
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[OK] Saved {len(all_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
