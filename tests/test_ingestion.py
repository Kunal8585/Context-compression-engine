"""Multi-file upload ingestion, including PDFs and the things that go wrong.

The property that matters most here is that **one bad file never loses the
batch**. A judge dragging in a folder will include something unparsable - a
scan, a binary, a `.DS_Store` - and the correct response is nine results and
one explanation, not a 500.

The PDFs below are generated in-process rather than checked in as binaries, so
what is being tested is visible in the test.
"""

from __future__ import annotations

import pytest

from engine.ingestion import (
    MAX_FILE_BYTES,
    IngestedFile,
    combine,
    combined_name,
    extract_file,
)


# ---------------------------------------------------------------------------
# Minimal real PDFs, built by hand so the fixture is readable
# ---------------------------------------------------------------------------
def _pdf(pages: list[str]) -> bytes:
    """A valid multi-page PDF with a text layer, using the base-14 Helvetica."""
    objects: list[bytes] = []

    def obj(body: str) -> int:
        objects.append(body.encode("latin-1"))
        return len(objects)

    # 1: catalog, 2: pages, 3: font, then one content stream + one page each.
    obj("<< /Type /Catalog /Pages 2 0 R >>")
    obj("")  # placeholder for /Pages, filled once page ids are known
    obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    for text in pages:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        content_id = obj(
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )
        page_ids.append(
            obj(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n"
        "%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


@pytest.fixture(scope="module")
def single_page_pdf() -> bytes:
    return _pdf(["Incident INC-4417 root cause was PAYMENT_POOL_SIZE set to 8"])


@pytest.fixture(scope="module")
def multi_page_pdf() -> bytes:
    return _pdf([
        "Page one covers the checkout timeout at 8000ms",
        "Page two covers the rollback across 3 regions",
        "Page three lists the follow-up actions",
    ])


# ---------------------------------------------------------------------------
# Text and code files
# ---------------------------------------------------------------------------
def test_a_text_file_is_extracted_with_its_name_preserved():
    result = extract_file("notes.md", b"# Incident\n\nPool exhausted at 10:41.")
    assert result.status == "done"
    assert result.name == "notes.md"
    assert "Pool exhausted" in result.text
    assert result.characters == len(result.text)


def test_a_directory_component_is_stripped_from_the_name():
    """A browser-supplied path must never become a path we touch."""
    result = extract_file("../../etc/passwd.txt", b"content")
    assert result.name == "passwd.txt"


def test_an_unsupported_extension_is_skipped_with_a_readable_reason():
    result = extract_file("photo.heic", b"\x00\x01binary")
    assert result.status == "skipped"
    assert ".heic" in result.reason


def test_a_file_with_no_extension_is_skipped():
    result = extract_file("Makefile", b"all:\n\techo hi")
    assert result.status == "skipped"
    assert "no extension" in result.reason


def test_an_empty_file_is_skipped_not_crashed():
    assert extract_file("empty.txt", b"").status == "skipped"


def test_a_binary_file_wearing_a_text_extension_is_skipped():
    """Decoding this would produce thousands of replacement chars to score."""
    result = extract_file("sneaky.txt", b"\x00\x01\x02\x03" * 500)
    assert result.status == "skipped"
    assert "binary" in result.reason


def test_invalid_utf8_costs_a_character_not_the_file():
    result = extract_file("mixed.log", b"ERROR at 10:41 \xff\xfe done")
    assert result.status == "done"
    assert "ERROR at 10:41" in result.text


def test_an_oversized_file_is_skipped_before_it_is_parsed():
    result = extract_file("huge.txt", b"x" * (MAX_FILE_BYTES + 1))
    assert result.status == "skipped"
    assert "per-file limit" in result.reason


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
def test_a_pdf_with_a_text_layer_is_extracted(single_page_pdf):
    result = extract_file("incident.pdf", single_page_pdf)
    assert result.status == "done", result.reason
    assert "INC-4417" in result.text
    assert "PAYMENT_POOL_SIZE" in result.text
    assert result.pages == 1


def test_every_page_of_a_multi_page_pdf_is_extracted(multi_page_pdf):
    result = extract_file("report.pdf", multi_page_pdf)
    assert result.status == "done", result.reason
    assert result.pages == 3
    for expected in ("8000ms", "3 regions", "follow-up actions"):
        assert expected in result.text


def test_a_corrupt_pdf_is_skipped_with_both_extractors_named():
    result = extract_file("broken.pdf", b"this is definitely not a pdf")
    assert result.status == "skipped"
    assert "pdfplumber" in result.reason and "PyPDF2" in result.reason


def test_a_pdf_with_no_text_layer_says_so(monkeypatch):
    """A scan is the realistic failure, and the reason has to be actionable."""
    import engine.ingestion as ingestion

    monkeypatch.setattr(ingestion, "_pdfplumber_text", lambda d: ("", 4, "no text layer found"))
    monkeypatch.setattr(ingestion, "_pypdf2_text", lambda d: ("", 4, "no text layer found"))
    result = extract_file("scan.pdf", b"%PDF-1.4 pretend")
    assert result.status == "skipped"
    assert "scanned or image-only" in result.reason


def test_pypdf2_is_used_when_pdfplumber_finds_nothing(monkeypatch):
    import engine.ingestion as ingestion

    monkeypatch.setattr(ingestion, "_pdfplumber_text", lambda d: ("", None, "boom"))
    monkeypatch.setattr(ingestion, "_pypdf2_text", lambda d: ("recovered text", 2, ""))
    result = extract_file("odd.pdf", b"%PDF-1.4 pretend")
    assert result.status == "done"
    assert result.text == "recovered text"
    assert result.pages == 2


# ---------------------------------------------------------------------------
# Combining - order, provenance, and surviving a bad file
# ---------------------------------------------------------------------------
def test_files_combine_in_upload_order_with_exact_provenance_ranges():
    files = [
        IngestedFile("a.txt", "alpha"),
        IngestedFile("b.txt", "beta"),
        IngestedFile("c.txt", "gamma"),
    ]
    text, sources, statuses = combine(files)

    assert text == "alpha\n\nbeta\n\ngamma"
    assert [s["name"] for s in sources] == ["a.txt", "b.txt", "c.txt"]
    # Every recorded range must actually contain that file's text.
    for source, original in zip(sources, files):
        assert text[source["start"]:source["end"]] == original.text
    assert len(statuses) == 3


def test_combining_is_deterministic():
    files = [IngestedFile("a.txt", "one"), IngestedFile("b.log", "two")]
    assert combine(files)[0] == combine(files)[0]


def test_a_skipped_file_is_excluded_from_the_text_but_kept_in_the_statuses():
    files = [
        IngestedFile("good.txt", "kept content"),
        IngestedFile("bad.pdf", "", "skipped", "no text layer"),
        IngestedFile("also_good.log", "more content"),
    ]
    text, sources, statuses = combine(files)

    assert "kept content" in text and "more content" in text
    assert [s["name"] for s in sources] == ["good.txt", "also_good.log"]
    assert [s["status"] for s in statuses] == ["done", "skipped", "done"]
    skipped = next(s for s in statuses if s["status"] == "skipped")
    assert skipped["reason"] == "no text layer"


def test_provenance_survives_a_skipped_file_in_the_middle():
    """The offsets must reflect what was actually concatenated, not what was uploaded."""
    files = [
        IngestedFile("first.txt", "AAAA"),
        IngestedFile("skipped.pdf", "", "skipped", "corrupt"),
        IngestedFile("third.txt", "BBBB"),
    ]
    text, sources, _ = combine(files)
    for source in sources:
        assert text[source["start"]:source["end"]] in {"AAAA", "BBBB"}


def test_all_files_skipped_produces_empty_text_and_full_statuses():
    files = [
        IngestedFile("a.pdf", "", "skipped", "corrupt"),
        IngestedFile("b.heic", "", "skipped", "unsupported"),
    ]
    text, sources, statuses = combine(files)
    assert text == "" and sources == []
    assert len(statuses) == 2


# ---------------------------------------------------------------------------
# Chunker routing for a batch
# ---------------------------------------------------------------------------
def test_a_single_upload_keeps_its_own_name_so_it_gets_the_right_chunker():
    assert combined_name([{"name": "auth_service.py", "status": "done"}]) == "auth_service.py"


def test_a_uniform_batch_keeps_the_shared_extension():
    statuses = [
        {"name": "a.py", "status": "done"},
        {"name": "b.py", "status": "done"},
    ]
    assert combined_name(statuses) == "uploads.py"


def test_a_mixed_batch_falls_back_to_text_chunking():
    statuses = [
        {"name": "a.py", "status": "done"},
        {"name": "b.log", "status": "done"},
    ]
    assert combined_name(statuses) == "uploads.txt"


def test_skipped_files_do_not_influence_chunker_routing():
    statuses = [
        {"name": "real.log", "status": "done"},
        {"name": "broken.pdf", "status": "skipped"},
    ]
    assert combined_name(statuses) == "real.log"
