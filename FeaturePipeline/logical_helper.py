import ipaddress
import math
from urllib.parse import urlparse


def _strip_port(host: str) -> str:
    """Remove port number from host string, handling IPv6 brackets."""
    if host.startswith("["):          # IPv6  e.g. [::1]:8080
        bracket_end = host.find("]")
        return host[: bracket_end + 1] if bracket_end != -1 else host
    return host.split(":")[0]


def is_ip_address(url: str) -> bool:
    """Return True if the URL's host is a bare IP (v4 or v6)."""
    try:
        host = _strip_port(urlparse(url).netloc)
        # strip IPv6 brackets for ip_address()
        host = host.strip("[]")
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def count_subdomains(url: str) -> int:
    """
    Count subdomains above the registrable domain.
    e.g. sub.evil.example.com  →  2  (sub + evil)
         example.com           →  0
    IP addresses always return 0.
    """
    netloc = _strip_port(urlparse(url).netloc)

    try:
        ipaddress.ip_address(netloc.strip("[]"))
        return 0
    except ValueError:
        pass

    parts = netloc.split(".")
    return max(len(parts) - 2, 0)


def compute_entropy(text: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not text:
        return 0.0
    length = len(text)
    prob = [text.count(c) / length for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def keyword_score(url: str) -> int:
    """Count how many phishing-related keywords appear in the URL."""
    suspicious_keywords = [
        "login", "secure", "account", "update", "verify",
        "bank", "paypal", "signin", "confirm", "password",
        "ebay", "amazon", "free", "bonus", "win",
    ]
    url_lower = url.lower()
    return sum(1 for word in suspicious_keywords if word in url_lower)


def _levenshtein(a: str, b: str) -> int:
    """Standard iterative Levenshtein distance (space-optimised)."""
    if len(a) < len(b):
        a, b = b, a                  # ensure a is the longer string
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def brand_distance(url: str) -> int:
    """
    Return the minimum Levenshtein distance between the URL's netloc
    and each brand name in the watchlist.
    A low value (e.g. ≤ 2) indicates possible brand-squatting.
    """
    brands = ["google", "facebook", "amazon", "paypal", "apple", "microsoft"]
    domain = _strip_port(urlparse(url).netloc).lower()

    return min(_levenshtein(domain, brand) for brand in brands)


def tld_score(url: str) -> int:
    """Return 1 if the domain uses a TLD commonly associated with phishing."""
    risky_tlds = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club"}
    netloc = _strip_port(urlparse(url).netloc).lower()
    return int(any(netloc.endswith(tld) for tld in risky_tlds))