from gojo.ssrf import is_public_http_url_for_fetch


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
