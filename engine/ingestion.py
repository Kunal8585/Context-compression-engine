"""Multi-file ingestion for the upload path: many local files in, one context out.

Stage 2 onwards is unchanged by this module. Chunking, redundancy, density,
selection and reconstruction see exactly what they always saw - a single string
and a filename. All this layer does is turn N uploaded files into that string
deterministically, and remember which byte range came from which file so the
existing ``spans[]`` output can carry provenance per chunk.

Three properties the upload path has to have:

**One bad file cannot lose the batch.** A scanned PDF with no text layer, a
corrupt file, an unsupported extension - each is reported as ``skipped`` with a
readable reason and the other files still compress. An upload of ten files
where the third is a photo of a receipt must return nine results and one
explanation, never a 500.

**Order is deterministic and documented.** Files concatenate in upload order,
separated by a blank line. The same set of files uploaded in the same order
always produces the same context, so a compression ratio is reproducible.

**Limits are enforced before the work, not during it.** Count, per-file size and
total size are all checked up front so an oversized batch is rejected in
milliseconds rather than discovered after two minutes of PDF parsing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)

#: Upload caps. Deliberately modest: this is a demo path, and the compression
#: story does not need a 100 MB corpus to be legible.
MAX_FILES = 10
MAX_TOTAL_BYTES = 10_000_000
MAX_FILE_BYTES = 5_000_000

#: What the chunker can meaningfully route. The extension drives chunker
#: selection downstream (.py -> tree-sitter, .log -> record chunking), so an
#: unknown one is skipped rather than guessed at.
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".sh", ".bash", ".zsh", ".sql", ".java", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    ".css", ".scss", ".html", ".xml",
}
PDF_SUFFIXES = {".pdf"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


@dataclass
class IngestedFile:
    """One upload's outcome. ``status`` is what the UI shows per file."""

    name: str
    text: str
    status: str = "done"          # done | skipped
    reason: str | None = None
    characters: int = 0
    pages: int | None = None

    def to_dict(self) -> dict:
        payload: dict = {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "characters": self.characters,
        }
        if self.pages is not None:
            payload["pages"] = self.pages
        return payload


def _skip(name: str, reason: str) -> IngestedFile:
    return IngestedFile(name=name, text="", status="skipped", reason=reason)


def extract_file(name: str, data: bytes) -> IngestedFile:
    """Decode one upload. Never raises - a bad file becomes a skip with a reason."""
    # Path().name strips any directory component a browser might send, so an
    # upload named "../../etc/passwd" is just "passwd" and never touches a path.
    safe_name = Path(name or "upload.txt").name or "upload.txt"
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        return _skip(
            safe_name,
            f"unsupported file type {suffix or '(no extension)'}; "
            f"supported: .pdf and common text/code extensions",
        )
    if not data:
        return _skip(safe_name, "file is empty")
    if len(data) > MAX_FILE_BYTES:
        return _skip(
            safe_name,
            f"file is {len(data):,} bytes; the per-file limit is {MAX_FILE_BYTES:,}",
        )

    if suffix in PDF_SUFFIXES:
        return _extract_pdf(safe_name, data)
    return _extract_text(safe_name, data)


def _extract_text(name: str, data: bytes) -> IngestedFile:
    """UTF-8 with replacement: a stray byte should cost one character, not a file."""
    if b"\x00" in data[:8192]:
        # A NUL in the first pages means this is a binary file wearing a text
        # extension. Decoding it produces thousands of replacement characters
        # that would be chunked and density-scored as if they were content.
        return _skip(name, "looks like a binary file, not text")
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return _skip(name, "file contains no text")
    return IngestedFile(name=name, text=text, characters=len(text))


def _extract_pdf(name: str, data: bytes) -> IngestedFile:
    """pdfplumber first, PyPDF2 as fallback, a clear skip if neither finds text.

    Two extractors because they fail differently: pdfplumber has much better
    layout handling but chokes on some malformed producers, while PyPDF2 is
    cruder and more tolerant. A PDF that defeats both is almost always a scan
    with no text layer - a real thing a user will upload, and one they need to
    be told about rather than have silently contribute nothing.
    """
    attempts: list[str] = []

    text, pages, error = _pdfplumber_text(data)
    if text:
        return IngestedFile(name=name, text=text, characters=len(text), pages=pages)
    attempts.append(f"pdfplumber: {error}")

    text, pages, error = _pypdf2_text(data)
    if text:
        log.info("%s: pdfplumber found nothing, PyPDF2 fallback succeeded", name)
        return IngestedFile(name=name, text=text, characters=len(text), pages=pages)
    attempts.append(f"PyPDF2: {error}")

    return _skip(
        name,
        "no extractable text (a scanned or image-only PDF has no text layer) - "
        + "; ".join(attempts),
    )


def _pdfplumber_text(data: bytes) -> tuple[str, int | None, str]:
    try:
        import pdfplumber
    except ImportError:
        return "", None, "not installed"
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            pages = []
            for index, page in enumerate(pdf.pages):
                # Per page, so one unreadable page in a 40-page document costs
                # that page rather than the whole file.
                try:
                    pages.append((page.extract_text() or "").strip())
                except Exception as exc:  # noqa: BLE001
                    log.debug("page %d unreadable: %s", index + 1, exc)
            count = len(pdf.pages)
        text = "\n\n".join(page for page in pages if page).strip()
        return text, count, "" if text else "no text layer found"
    except Exception as exc:  # noqa: BLE001
        return "", None, f"{type(exc).__name__}: {exc}"


def _pypdf2_text(data: bytes) -> tuple[str, int | None, str]:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return "", None, "not installed"
    try:
        reader = PdfReader(BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages):
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception as exc:  # noqa: BLE001
                log.debug("page %d unreadable: %s", index + 1, exc)
        text = "\n\n".join(page for page in pages if page).strip()
        return text, len(reader.pages), "" if text else "no text layer found"
    except Exception as exc:  # noqa: BLE001
        return "", None, f"{type(exc).__name__}: {exc}"


#: Inserted between files. Two newlines so the text chunker sees a paragraph
#: boundary and never merges the tail of one file into the head of the next.
SEPARATOR = "\n\n"


def combine(files: list[IngestedFile]) -> tuple[str, list[dict], list[dict]]:
    """Join successful uploads in upload order, keeping each one's byte range.

    Returns ``(text, source_files, statuses)`` where ``source_files`` maps a
    character range in the combined text back to the file it came from. The
    pipeline uses that to tag each chunk with ``source_file``, which is what
    puts a provenance badge on every span in the diff view.
    """
    parts: list[str] = []
    sources: list[dict] = []
    statuses: list[dict] = []
    cursor = 0

    for item in files:
        statuses.append(item.to_dict())
        if item.status != "done" or not item.text:
            continue
        if parts:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
        start = cursor
        parts.append(item.text)
        cursor += len(item.text)
        sources.append({"name": item.name, "start": start, "end": cursor})

    return "".join(parts), sources, statuses


def combined_name(statuses: list[dict]) -> str:
    """Filename to hand the chunker for a multi-file upload.

    The extension decides which chunker runs, so a single-file upload keeps its
    own name and gets the right one (.py -> tree-sitter, .log -> records). A
    batch that is all one type gets that type. Genuinely mixed batches have no
    single right answer and fall back to text chunking, which is the safe
    default for heterogeneous input.
    """
    done = [f for f in statuses if f.get("status") == "done"]
    if not done:
        return "uploads.txt"
    if len(done) == 1:
        return str(done[0]["name"])
    suffixes = {Path(str(f["name"])).suffix.lower() for f in done}
    if len(suffixes) == 1:
        return f"uploads{suffixes.pop()}"
    return "uploads.txt"
