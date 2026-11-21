from gojo.fingerprint import digest_fingerprint


def test_fingerprint_deterministic() -> None:
    fp1 = digest_fingerprint(
        feed_provider="hn",
        external_id="12345",
        locale="en",
        prompt_version="v5",
        summary_type="quick",
    )
    fp2 = digest_fingerprint(
        feed_provider="hn",
        external_id="12345",
        locale="en",
        prompt_version="v5",
        summary_type="quick",
    )
    assert fp1 == fp2
    assert len(fp1) == 64


def test_fingerprint_case_normalized() -> None:
    fp_upper = digest_fingerprint(
        feed_provider="hn",
        external_id="12345",
        locale="EN ",
        prompt_version="v5",
        summary_type="QUICK",
    )
    fp_lower = digest_fingerprint(
        feed_provider="hn",
        external_id="12345",
        locale="en",
        prompt_version="v5",
        summary_type="quick",
    )
    assert fp_upper == fp_lower


def test_fingerprint_summary_type_distinct() -> None:
    fp_quick = digest_fingerprint(
        feed_provider="hn", external_id="12345", locale="en", prompt_version="v5", summary_type="quick"
    )
    fp_tech = digest_fingerprint(
        feed_provider="hn", external_id="12345", locale="en", prompt_version="v5", summary_type="technical"
    )
    fp_thai = digest_fingerprint(
        feed_provider="hn", external_id="12345", locale="th", prompt_version="v5", summary_type="thai_concise"
    )
    assert fp_quick != fp_tech
    assert fp_quick != fp_thai
