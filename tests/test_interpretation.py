from gojo.interpretation import (
    extract_interpretation_fields,
    extract_summary_extra_from_markdown,
)


def test_extract_summary_extra_success() -> None:
    markdown = """
## Summary
This is the main summary.

### Why it matters
It represents a major breakthrough in distributed consensus protocols.

### Key takeaway
Latency is reduced by 40% with zero loss in safety.

### Gojo note
Ensure your node configuration has adequate memory before migrating.
"""
    extra = extract_summary_extra_from_markdown(markdown)
    assert "why_matters" in extra
    assert "breakthrough" in extra["why_matters"]
    assert "key_takeaway" in extra
    assert "Latency is reduced" in extra["key_takeaway"]
    assert "note" in extra
    assert "Ensure your node" in extra["note"]
    assert extract_interpretation_fields(markdown) == extra


def test_extract_interpretation_fields_thai_headings() -> None:
    markdown = """
## สรุป
บทความนี้อธิบายการปรับปรุงตัวจัดตารางงานของเคอร์เนลลินุกซ์

### ความเห็นชุมชน
วิศวกรเห็นตรงกันว่า latency ลดลงจริง แต่ [1.2] @skeptic ตั้งข้อสังเกตเรื่องภาระซีพียู

### ทำไมเรื่องนี้ถึงสำคัญ
ช่วยลดเวลาตอบสนองของระบบคลาวด์ที่มีโหลดสูงได้อย่างมีนัยสำคัญ

### ประเด็นสำคัญ
ลด tail latency ลง 35% โดยไม่ต้องแก้โค้ดแอปพลิเคชัน

### ข้อสังเกต
ควรทดสอบกับเวิร์กโหลดจริงก่อนอัปเกรดเคอร์เนลบนโปรดักชัน
"""
    extra = extract_interpretation_fields(markdown)
    assert "ช่วยลดเวลาตอบสนอง" in extra["why_matters"]
    assert "35%" in extra["key_takeaway"]
    assert "โปรดักชัน" in extra["note"]
    assert "@skeptic" in extra["community_consensus"]


def test_extract_interpretation_fields_thai_alternate_note_heading() -> None:
    markdown = """
### ทำไมถึงสำคัญ:
ประหยัดหน่วยความจำลงครึ่งหนึ่ง

### ข้อสรุปสำคัญ:
เหมาะกับบริการ edge ที่มีแรมจำกัด

### หมายเหตุ Gojo:
ตรวจสอบความเข้ากันได้ของปลั๊กอินเดิมก่อนย้ายระบบ
"""
    extra = extract_interpretation_fields(markdown)
    assert "ประหยัดหน่วยความจำ" in extra["why_matters"]
    assert "edge" in extra["key_takeaway"]
    assert "ปลั๊กอินเดิม" in extra["note"]


def test_extract_summary_extra_missing_sections() -> None:
    markdown = "Just a raw summary without headers."
    extra = extract_summary_extra_from_markdown(markdown)
    assert extra == {}
