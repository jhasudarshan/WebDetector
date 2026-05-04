import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse
import ipaddress
import whois


# ─────────────────────────── HELPERS ────────────────────────────────────────

def extract_domain(url: str) -> str:
    """
    Return a clean, lowercase domain from a full URL.
    Strips scheme, port, and leading 'www.'.
    """
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path
    domain = domain.split(":")[0]          # remove port
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.lower()


def is_ip(domain: str) -> bool:
    """Return True for bare IPv4 or IPv6 addresses (brackets stripped)."""
    try:
        ipaddress.ip_address(domain.strip("[]"))
        return True
    except ValueError:
        return False


# ─────────────────────────── WHOIS HELPERS ──────────────────────────────────

def safe_get(w, key):
    """
    Unified getter that handles both dict-style and object-style WHOIS
    responses returned by different versions of the python-whois library.
    """
    if isinstance(w, dict):
        return w.get(key)
    return getattr(w, key, None)


def normalize_date(value):
    """
    Collapse list responses to a single date and strip timezone info so
    arithmetic against datetime.utcnow() (naive) never raises TypeError.

    Fixes vs original
    -----------------
    * Handles timezone-aware datetimes correctly via .astimezone(utc)
      before stripping tzinfo, preventing incorrect UTC offsets being
      silently ignored.
    """
    if isinstance(value, list):
        value = value[0]

    if value is None:
        return None

    if hasattr(value, "tzinfo") and value.tzinfo is not None:
        # Convert to UTC first, then make naive
        value = value.astimezone(timezone.utc).replace(tzinfo=None)

    return value


def get_whois_data(domain: str):
    """
    Single WHOIS lookup.  Returns the whois object or None on failure.

    Fixes vs original
    -----------------
    * Skips lookup immediately for IP addresses — python-whois can hang
      or raise confusing errors on bare IPs.
    """
    if is_ip(domain):
        return None
    try:
        return whois.whois(domain)
    except Exception as e:
        print(f"[WHOIS] fetch error for {domain!r}: {e}")
        return None


# ─────────────────────────── WHOIS FEATURES ─────────────────────────────────

def get_domain_age(domain: str, w=None) -> int:
    """
    Days since the domain was registered.  Returns -1 on any failure.

    Fixes vs original
    -----------------
    * Uses datetime.now(timezone.utc).replace(tzinfo=None) instead of
      deprecated datetime.utcnow() (removed in Python 3.12+).
    * Guards against creation_date being in the future (malformed WHOIS).
    """
    if is_ip(domain):
        return -1
    try:
        if w is None:
            w = get_whois_data(domain)
        if not w:
            return -1

        creation_date = normalize_date(safe_get(w, "creation_date"))
        if creation_date:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            return max((now - creation_date).days, 0)
    except Exception as e:
        print(f"[WHOIS] age error for {domain!r}: {e}")
    return -1


def get_expiry(domain: str, w=None) -> int:
    """
    Days until the domain expires.  Returns -1 on any failure.

    Fixes vs original
    -----------------
    * Same datetime.utcnow() → datetime.now(utc) fix.
    * Returns 0 (not -1) when expiry is in the past — the domain has
      already expired, which is a meaningful signal.
    """
    if is_ip(domain):
        return -1
    try:
        if w is None:
            w = get_whois_data(domain)
        if not w:
            return -1

        expiry_date = normalize_date(safe_get(w, "expiration_date"))
        if expiry_date:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            return max((expiry_date - now).days, 0)
    except Exception as e:
        print(f"[WHOIS] expiry error for {domain!r}: {e}")
    return -1


def has_whois(domain: str, w=None) -> bool:
    """Return True when a valid WHOIS creation_date is found."""
    if is_ip(domain):
        return False
    try:
        if w is None:
            w = get_whois_data(domain)
        if not w:
            return False
        return safe_get(w, "creation_date") is not None
    except Exception:
        return False


# ─────────────────────────── DNS ────────────────────────────────────────────

def has_dns(domain: str) -> bool:
    """
    Return True when the domain resolves via DNS.

    Fixes vs original
    -----------------
    * IPs are still handled (always return True) but via is_ip() for
      consistency instead of a bare try/except swallowing everything.
    * Re-raises unexpected non-gaierror exceptions so they don't silently
      vanish — they indicate bugs, not missing DNS records.
    """
    if is_ip(domain):
        return True
    try:
        socket.getaddrinfo(domain, None)
        return True
    except socket.gaierror:
        return False


# ─────────────────────────── SSL ────────────────────────────────────────────

def _get_cert(domain: str, timeout: int = 5):
    """
    Internal helper: open a TLS connection and return the peer certificate
    dict, or None on any error.  Centralises the repeated
    create_connection / wrap_socket boilerplate.
    """
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                return ssock.getpeercert()
    except Exception:
        return None


def check_ssl(domain: str) -> int:
    """Return 1 if a valid TLS certificate is present, 0 otherwise."""
    if not domain or is_ip(domain):
        return 0
    cert = _get_cert(domain)
    return 1 if cert else 0


def get_cert_duration(domain: str) -> int:
    """
    Return the certificate's total validity window in days, or -1 on failure.

    Fixes vs original
    -----------------
    * Reuses _get_cert() to avoid duplicating the socket logic.
    * Parses notBefore / notAfter with a timezone-aware format string
      (%Z handles 'GMT') — the original already did this correctly but
      now uses the shared helper.
    """
    if not domain or is_ip(domain):
        return -1
    cert = _get_cert(domain)
    if not cert:
        return -1
    try:
        fmt = "%b %d %H:%M:%S %Y %Z"
        not_before = datetime.strptime(cert["notBefore"], fmt)
        not_after  = datetime.strptime(cert["notAfter"],  fmt)
        return max((not_after - not_before).days, 0)
    except Exception as e:
        print(f"[SSL] cert duration error for {domain!r}: {e}")
        return -1


def get_ssl_issuer(domain: str):
    """
    Return the certificate issuer's organisation name, or None on failure.

    Fixes vs original
    -----------------
    * Reuses _get_cert(); avoids a third independent TLS handshake.
    * Handles both tuple-of-tuples and flat-dict issuer formats.
    """
    if not domain or is_ip(domain):
        return None
    cert = _get_cert(domain)
    if not cert:
        return None
    try:
        # cert['issuer'] is a tuple of ((key, value), ...) pairs
        issuer = {k: v for rdn in cert["issuer"] for k, v in rdn}
        return issuer.get("organizationName")
    except Exception:
        return None


# ─────────────────────────── HEURISTICS ─────────────────────────────────────

# Use frozensets for O(1) lookup
_SUSPICIOUS_TLDS   = frozenset({".tk", ".ml", ".ga", ".cf", ".gq"})
_HIGH_RISK_REGIONS = frozenset({"ru", "cn", "kp"})


def reputation_score(domain: str) -> int:
    """Return 1 if the domain uses a TLD linked to free/abusive registrars."""
    return int(any(domain.endswith(tld) for tld in _SUSPICIOUS_TLDS))


def geo_risk(domain: str) -> int:
    """Return 1 if the domain is under a high-risk country ccTLD."""
    return int(any(domain.endswith("." + r) for r in _HIGH_RISK_REGIONS))


# ─────────────────────────── METADATA ───────────────────────────────────────

def get_hosting_ip(domain: str):
    """
    Resolve domain → IPv4 string, or None on failure.

    Fixes vs original
    -----------------
    * Uses getaddrinfo instead of gethostbyname so it works for
      IPv6-only hosts too, and returns the first result's address.
    """
    try:
        results = socket.getaddrinfo(domain, None)
        if results:
            return results[0][4][0]   # (family, type, proto, canonname, sockaddr)
    except Exception:
        pass
    return None


def get_asn(ip: str):
    """
    Return the ASN string for an IP address using ipwhois RDAP lookup.

    Fixes vs original
    -----------------
    * Validates that ip is a real address before calling IPWhois, which
      raises cryptic errors on None or empty strings.
    """
    if not ip:
        return None
    try:
        ipaddress.ip_address(ip)   # validates format
    except ValueError:
        return None
    try:
        from ipwhois import IPWhois
        obj = IPWhois(ip)
        res = obj.lookup_rdap(depth=1)
        return res.get("asn")
    except Exception:
        return None


def get_registrar(w=None, domain: str = None):
    """Return the registrar name from WHOIS data, or None."""
    try:
        if w is None and domain:
            w = get_whois_data(domain)
        return safe_get(w, "registrar") if w else None
    except Exception:
        return None


def get_whois_country(w=None, domain: str = None):
    """
    Return the registrant country from WHOIS data, or None.

    Fixes vs original
    -----------------
    * 'country' is not always present; falls back to checking
      'registrant_country' which some TLD registries use.
    """
    try:
        if w is None and domain:
            w = get_whois_data(domain)
        if not w:
            return None
        return safe_get(w, "country") or safe_get(w, "registrant_country")
    except Exception:
        return None