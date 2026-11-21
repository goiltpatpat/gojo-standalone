import asyncio
import socket
from unittest.mock import AsyncMock, patch

import pytest

from gojo.ssrf import (
    is_public_http_url_for_fetch,
    is_public_ip,
    validate_public_url_with_dns,
)


def test_ssrf_blocks_private_and_loopback() -> None:
    assert is_public_http_url_for_fetch("http://127.0.0.1/admin") is False
    assert is_public_http_url_for_fetch("http://localhost:8080/metrics") is False
    assert is_public_http_url_for_fetch("http://10.0.0.1/status") is False
    assert is_public_http_url_for_fetch("http://192.168.1.1/secret") is False
    assert is_public_http_url_for_fetch("http://172.16.0.5/api") is False
    assert is_public_http_url_for_fetch("http://169.254.169.254/latest/meta-data/") is False
    assert is_public_http_url_for_fetch("http://0.0.0.0/") is False


def test_ssrf_blocks_credentials_and_schemes() -> None:
    assert is_public_http_url_for_fetch("ftp://example.com/file") is False
    assert is_public_http_url_for_fetch("file:///etc/passwd") is False
    assert is_public_http_url_for_fetch("http://user:pass@example.com") is False


def test_ssrf_allows_public_urls() -> None:
    assert is_public_http_url_for_fetch("https://news.ycombinator.com") is True
    assert is_public_http_url_for_fetch("https://github.com/torvalds/linux") is True
    assert is_public_http_url_for_fetch("http://example.com/blog/article-1") is True
    assert is_public_http_url_for_fetch("http://93.184.216.34/page") is True


def test_ssrf_blocks_decimal_octal_hex_and_shorthand_ipv4() -> None:
    assert is_public_http_url_for_fetch("http://2130706433/") is False
    assert is_public_http_url_for_fetch("http://0x7f000001/") is False
    assert is_public_http_url_for_fetch("http://0177.0.0.1/") is False
    assert is_public_http_url_for_fetch("http://0x7f.0.0.1/") is False
    assert is_public_http_url_for_fetch("http://127.1/") is False
    assert is_public_http_url_for_fetch("http://127.0.1/") is False
    assert is_public_http_url_for_fetch("http://017700000001/") is False
    assert is_public_http_url_for_fetch("http://010.0.0.1/") is False


def test_ssrf_blocks_ipv4_mapped_ipv6_and_special_hosts() -> None:
    assert is_public_http_url_for_fetch("http://[::ffff:127.0.0.1]/") is False
    assert is_public_http_url_for_fetch("http://[::ffff:169.254.169.254]/latest/") is False
    assert is_public_http_url_for_fetch("http://[::ffff:10.0.0.1]/") is False
    assert is_public_http_url_for_fetch("http://[::ffff:7f00:1]/") is False
    assert is_public_http_url_for_fetch("http://[::1]/") is False
    assert is_public_http_url_for_fetch("http://[fe80::1]/") is False
    assert is_public_http_url_for_fetch("http://localhost./") is False
    assert is_public_http_url_for_fetch("http://service.internal./") is False
    assert is_public_http_url_for_fetch("http://printer.local./") is False


def test_is_public_ip_helper() -> None:
    assert is_public_ip("93.184.216.34") is True
    assert is_public_ip("8.8.8.8") is True
    assert is_public_ip("2606:2800:220:1:248:1893:25c8:1946") is True
    assert is_public_ip("::ffff:93.184.216.34") is True

    for blocked in (
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fe80::1",
        "fe80::1%eth0",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "",
        "not-an-ip",
    ):
        assert is_public_ip(blocked) is False, f"Expected {blocked} to be blocked"


@pytest.mark.asyncio
async def test_validate_public_url_with_dns_allows_public_resolution() -> None:
    loop = asyncio.get_running_loop()
    fake_records = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
    ]
    with patch.object(loop, "getaddrinfo", new=AsyncMock(return_value=fake_records)) as mock_dns:
        ok, reason = await validate_public_url_with_dns("https://example.com/article")
        assert ok is True
        assert reason == ""
        mock_dns.assert_awaited_once_with("example.com", None)


@pytest.mark.asyncio
async def test_validate_public_url_with_dns_rejects_private_resolution() -> None:
    loop = asyncio.get_running_loop()
    # One public and one loopback IP -> must reject
    fake_records = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
    ]
    with patch.object(loop, "getaddrinfo", new=AsyncMock(return_value=fake_records)):
        ok, reason = await validate_public_url_with_dns("https://rebind.example.com/")
        assert ok is False
        assert "127.0.0.1" in reason

    # IPv4-mapped metadata IPv6 resolution -> must reject
    mapped_records = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:169.254.169.254", 0, 0, 0)),
    ]
    with patch.object(loop, "getaddrinfo", new=AsyncMock(return_value=mapped_records)):
        ok, reason = await validate_public_url_with_dns("https://meta.example.com/")
        assert ok is False
        assert "169.254.169.254" in reason


@pytest.mark.asyncio
async def test_validate_public_url_with_dns_rejects_dns_failure_and_static_failure() -> None:
    loop = asyncio.get_running_loop()
    with patch.object(
        loop, "getaddrinfo", new=AsyncMock(side_effect=socket.gaierror("NXDOMAIN"))
    ) as mock_dns:
        ok, reason = await validate_public_url_with_dns("https://nonexistent.invalid/")
        assert ok is False
        assert "DNS resolution failed" in reason
        mock_dns.assert_awaited_once()

    # Static SSRF failure short-circuits before DNS
    with patch.object(loop, "getaddrinfo", new=AsyncMock()) as mock_dns:
        ok, reason = await validate_public_url_with_dns("http://2130706433/")
        assert ok is False
        assert "static SSRF" in reason
        mock_dns.assert_not_awaited()
