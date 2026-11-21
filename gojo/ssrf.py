"""
SSRF Protection: Block private, internal, loopback, and reserved addresses for external article fetches.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
from urllib.parse import urlparse

_DECIMAL_OCTET_RE = re.compile(r"^(?:0|[1-9][0-9]{0,2})$")
_NUMERIC_LABEL_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|0[0-7]+|[0-9]+)$")


def _unwrap_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) to their IPv4 address."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_public_ip(ip_str: str) -> bool:
    """
    Return True if `ip_str` parses as a globally routable public IPv4/IPv6 address.
    Unwraps IPv4-mapped IPv6 and rejects private, loopback, link-local, CGNAT,
    multicast, reserved, unspecified, and non-global addresses.
    """
    if not ip_str or not isinstance(ip_str, str):
        return False

    cleaned = ip_str.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]

    # Reject zone-scoped IPv6 addresses (e.g. fe80::1%eth0)
    if "%" in cleaned:
        return False

    try:
        ip = _unwrap_ip(ipaddress.ip_address(cleaned))
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address):
        # Also reject 6to4 / Teredo embeddings that wrap non-public IPv4
        if ip.sixtofour is not None and not is_public_ip(str(ip.sixtofour)):
            return False
        if ip.teredo is not None:
            server, client = ip.teredo
            if not is_public_ip(str(server)) or not is_public_ip(str(client)):
                return False

    if (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    # Explicit IPv4 CGNAT (100.64.0.0/10) and 0.0.0.0/8 defense-in-depth
    if isinstance(ip, ipaddress.IPv4Address):
        octets = ip.packed
        if octets[0] == 0:
            return False
        if octets[0] == 100 and 64 <= octets[1] <= 127:
            return False

    return True


def _is_nonstandard_ipv4_literal(host: str) -> bool:
    """
    Detect decimal, octal, hex, or shorthand IPv4 host literals
    (e.g. 2130706433, 0x7f000001, 0177.0.0.1, 0x7f.0.0.1, 127.1).
    Standard dotted-quad decimal IPv4 (e.g. 93.184.216.34) returns False
    so it is handled by `is_public_ip`.
    """
    parts = host.split(".")
    if not (1 <= len(parts) <= 4):
        return False
    if not all(_NUMERIC_LABEL_RE.match(p) for p in parts):
        return False

    # Standard 4-part decimal without leading zeros is a canonical IPv4 literal
    if len(parts) == 4 and all(_DECIMAL_OCTET_RE.match(p) for p in parts):
        return False

    # Otherwise it is decimal integer, hex, octal, or shorthand IPv4 -> reject
    return True


def _is_blocked_ipv4(host: str) -> bool:
    """Check literal IPv4 (including non-standard forms) against internal/private ranges."""
    if _is_nonstandard_ipv4_literal(host):
        return True
    if not re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host):
        return False
    return not is_public_ip(host)


def is_public_http_url_for_fetch(url_string: str) -> bool:
    """Validate that the target URL is safe for outbound HTTP requests."""
    try:
        if not url_string or not isinstance(url_string, str):
            return False
        u = urlparse(url_string.strip())
        if u.username or u.password:
            return False
        if u.scheme not in ("http", "https"):
            return False
        raw_host = (u.hostname or "").strip().lower()
        if not raw_host:
            return False

        # Reject IPv6 zone IDs or whitespace/control chars
        if "%" in raw_host or any(ord(c) <= 32 for c in raw_host):
            return False

        host = raw_host.rstrip(".")
        if not host:
            return False

        if (
            host in ("localhost", "0.0.0.0", "[::1]", "::1")
            or host.endswith(".localhost")
            or host.endswith(".local")
            or host.endswith(".internal")
        ):
            return False

        if _is_blocked_ipv4(host):
            return False

        try:
            ipaddress.ip_address(host)
        except ValueError:
            # Not a literal IP address; hostname passed static checks.
            return True

        return is_public_ip(host)
    except Exception:
        return False


async def validate_public_url_with_dns(url: str) -> tuple[bool, str]:
    """
    Validate a URL with static SSRF checks followed by async DNS resolution.
    Rejects if any resolved A/AAAA address is non-public or if DNS resolution fails.
    Returns `(True, "")` on success or `(False, reason)` on rejection.
    """
    if not is_public_http_url_for_fetch(url):
        return False, "URL failed static SSRF validation"

    u = urlparse(url.strip())
    hostname = (u.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return False, "URL has no hostname"

    # If hostname is already a literal IP, static check already verified it with is_public_ip
    try:
        ipaddress.ip_address(hostname)
        if is_public_ip(hostname):
            return True, ""
        return False, f"IP address '{hostname}' is not public"
    except ValueError:
        pass

    try:
        loop = asyncio.get_running_loop()
        addrinfos = await loop.getaddrinfo(hostname, None)
    except Exception as exc:
        return False, f"DNS resolution failed for '{hostname}': {exc}"

    if not addrinfos:
        return False, f"DNS resolution returned no addresses for '{hostname}'"

    for info in addrinfos:
        sockaddr = info[4]
        if not sockaddr:
            return False, f"Invalid DNS record for '{hostname}'"
        ip_str = str(sockaddr[0])
        if not is_public_ip(ip_str):
            return False, f"Resolved IP '{ip_str}' for '{hostname}' is not public"

    return True, ""
