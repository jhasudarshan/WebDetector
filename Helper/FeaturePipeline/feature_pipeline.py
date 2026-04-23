from urllib.parse import urlparse

from bs4 import BeautifulSoup
from FeaturePipeline.logical_helper import (
    is_ip_address, count_subdomains,
    keyword_score, brand_distance,
    compute_entropy, tld_score
)

from FeaturePipeline.content_helper import(
    fetch_html, detect_obfuscated_js,
    detect_redirect, detect_login_form,
    count_inputs, count_passwords,
    external_links_ratio
)

from FeaturePipeline.domain_helper import (
    get_domain_age, get_expiry,
    has_whois, has_dns,
    check_ssl, get_cert_duration,
    geo_risk, reputation_score, get_whois_data
)


def extract_lexical_features(url):
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_https": int(url.startswith("https://")),
        "has_ip": int(is_ip_address(url)),
        "has_at_symbol": int('@' in url),
        "has_dash": int('-' in url),
        "subdomain_count": count_subdomains(url),
    }



def advanced_lexical_features(url):
    return {
        "entropy": compute_entropy(url),
        "suspicious_keywords": keyword_score(url),
        "brand_similarity": brand_distance(url),
        "tld_risk_score": tld_score(url),
        "path_length": len(url.split('/')),
    }


def extract_content_features(url):
    html = fetch_html(url)

    if not html:
        return {
            "num_forms": 0,
            "num_iframes": 0,
            "num_anchors": 0,
            "num_scripts": 0,
            "has_login_form": 0,
            "external_link_ratio": 0,
            "has_redirect": 0,
            "input_fields": 0,
            "password_fields": 0,
            "js_obfuscation_score": 0,
        }

    soup = BeautifulSoup(html, "html.parser")

    return {
        "num_forms": len(soup.find_all('form')),
        "num_iframes": len(soup.find_all('iframe')),
        "num_anchors": len(soup.find_all('a')),
        "num_scripts": len(soup.find_all('script')),
        "has_login_form": detect_login_form(soup),
        "external_link_ratio": external_links_ratio(soup, url),
        "has_redirect": detect_redirect(html),
        "input_fields": count_inputs(soup),
        "password_fields": count_passwords(soup),
        "js_obfuscation_score": detect_obfuscated_js(soup),
    }

def extract_domain_features(url):
    domain = urlparse(url).netloc
    w = get_whois_data(domain)

    return {
        "domain_age": get_domain_age(domain, w),
        "time_to_expiry": get_expiry(domain, w),
        "has_whois": int(has_whois(domain, w)),
        "has_dns_record": int(has_dns(domain)),
        "ssl_valid": check_ssl(domain),
        "cert_validity_days": get_cert_duration(domain),
        "domain_reputation": reputation_score(domain),
        "hosting_country_risk": geo_risk(domain),
    }


def build_feature_vector(url):
    lexical = extract_lexical_features(url)
    adv_lexical = advanced_lexical_features(url)

    content = extract_content_features(url)
    domain = extract_domain_features(url)

    return {
        **lexical,
        **adv_lexical,
        **content,
        **domain
    }
