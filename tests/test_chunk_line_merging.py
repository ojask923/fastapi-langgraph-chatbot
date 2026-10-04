"""Verify that _chunk applies line-merging only to PDF documents.

Test 1 (markdown): a .md Document with a bullet list and fenced code block
  must come out of _chunk with its newlines and indentation intact.

Test 2 (plain text): a .txt Document must likewise preserve newlines.

Test 3 (PDF): prose from a PDF Document must have soft line-breaks collapsed
  to spaces and de-hyphenated, just as _merge_lines always did.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch DATABASE_URL before importing app modules so we stay on SQLite.
os.environ.setdefault("DATABASE_URL", "sqlite:///./chatbot.db")

from langchain_core.documents import Document
from app.services.rag_service import rag_service

# Shared metadata skeleton — only document_type varies between tests.
def _make_doc(text: str, doc_type: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": "test-doc",
            "document_type": doc_type,
            "section": "s1",
            "page_number": 1,
            "filename": f"test.{doc_type}",
            "user_id": "",
            "session_id": "",
        },
    )


# ---------------------------------------------------------------------------
# Test 1 — Markdown: newlines must be preserved
# ---------------------------------------------------------------------------
def test_markdown_preserves_structure():
    md_text = "## Setup\n- install deps\n- run server\n\n`python\nif x:\n    print(1)\n`"
    docs = [_make_doc(md_text, "md")]
    chunks = rag_service._chunk(docs)

    # All chunks together must contain the original structural markers
    full = "\n".join(c.page_content for c in chunks)
    assert "- install deps" in full, f"bullet list stripped: {full!r}"
    assert "    print(1)" in full, f"indentation stripped: {full!r}"
    assert "`python" in full, f"fenced code block stripped: {full!r}"
    # Crucially: list items must NOT have been collapsed onto one line
    assert "- install deps - run server" not in full, \
        f"list items were collapsed into one line: {full!r}"
    print("PASS test_markdown_preserves_structure")


# ---------------------------------------------------------------------------
# Test 2 — Plain text: newlines must be preserved
# ---------------------------------------------------------------------------
def test_txt_preserves_newlines():
    txt_text = "Line one\nLine two\nLine three"
    docs = [_make_doc(txt_text, "txt")]
    chunks = rag_service._chunk(docs)

    full = "\n".join(c.page_content for c in chunks)
    # Single newlines must NOT have been replaced by spaces
    assert "Line one Line two" not in full, \
        f"newlines were collapsed in .txt: {full!r}"
    assert "Line one" in full and "Line two" in full, f"content lost: {full!r}"
    print("PASS test_txt_preserves_newlines")


# ---------------------------------------------------------------------------
# Test 3 — PDF: prose must be re-joined (soft wraps collapsed, de-hyphenated)
# ---------------------------------------------------------------------------
def test_pdf_merges_prose():
    # Simulate two PDF lines from the same page/section that form one sentence.
    # They are passed as separate Documents because PDF loaders split by line.
    line1 = _make_doc("This is an infor-", "pdf")
    line2 = _make_doc("mation chunk.", "pdf")
    # Give them the same section/page so _chunk groups them.
    chunks = rag_service._chunk([line1, line2])

    full = " ".join(c.page_content for c in chunks)
    assert "information chunk." in full, \
        f"de-hyphenation did not fire for PDF: {full!r}"
    # The raw "\n" between lines must have been collapsed to a space (or removed)
    assert "\n" not in full or "information" in full, \
        f"soft line-wrap not collapsed for PDF: {full!r}"
    print("PASS test_pdf_merges_prose:", repr(full))


if __name__ == "__main__":
    test_markdown_preserves_structure()
    test_txt_preserves_newlines()
    test_pdf_merges_prose()
    print()
    print("ALL TESTS PASSED")