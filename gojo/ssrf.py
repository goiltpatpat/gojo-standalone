"""
SSRF Protection: Block private, internal, loopback, and reserved addresses for external article fetches.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse


def _is_blocked_ipv4(host: str) -> bool:
    """Check literal IPv4 against internal/private ranges."""
    if not re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host):
        return False
    try:
        parts = [int(x) for x in host.split(".")]
    except ValueError:
        return True
    if any(n > 255 for n in parts):
        return True
    if parts[0] == 10:  # 10.0.0.0/8
        return True
    if parts[0] == 127:  # 127.0.0.0/8 loopback
        return True
    if parts[0] == 0:  # 0.0.0.0/8
        return True
    if parts[0] == 192 and parts[1] == 168:  # 192.168.0.0/16
        return True
    if parts[0] == 169 and parts[1] == 254:  # 169.254.0.0/16 link-local
        return True
    if parts[0] == 100 and 64 <= parts[1] <= 127:  # Carrier Grade NAT
        return True
    if parts[0] == 172 and 16 <= parts[1] <= 31:  # 172.16.0.0/12
        return True
    return False


def is_public_http_url_for_fetch(url_string: str) -> bool:
    """Validate that the target URL is safe for outbound HTTP requests."""
    try:
        u = urlparse(url_string.strip())
        if u.username or u.password:
            return False
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        if not host:
            return False
        if (
            host in ("localhost", "0.0.0.0", "[::1]")
            or host.endswith(".localhost")
            or host.endswith(".local")
            or host.endswith(".internal")
        ):
            return False

        if _is_blocked_ipv4(host):
            return False

        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_loopback
                or ip.is_private
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False
        except ValueError:
            pass

        return True
    except Exception:
        return False
