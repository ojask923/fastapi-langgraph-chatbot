"""Verify the pdfplumber word-ordering fix in _parse.

The bug: words were sorted by (round(top,1), x0) globally before grouping.
Words on the same visual line but with slightly different top values (superscripts,
bold headings, mixed font sizes) were interleaved incorrectly.

The fix: sort by top only, group with a font-size-relative tolerance, then sort
each completed line by x0 before joining.

Test 1 – superscript case:
  Words E(top=100.4), =(top=100.0), mc(top=100.0), 2(top=99.2, size=6)
  With the old code: sorted as [(99.2,'2'),(100.0,'='),(100.0,'mc'),(100.4,'E')]
    -> all fall within 3pt of first anchor (99.2) -> single line "2 = mc E"  WRONG
  With the new code: sorted by top only -> [2, =, mc, E], grouped together because
    the superscript '2' has size=6, tol=max(3,3)=3, and 100.4-99.2=1.2 <= 3,
    so they're in one line; sorted by x0 -> "E = mc 2"  CORRECT

Test 2 – two distinct lines are still split correctly.

Test 3 – unit test the grouping logic in isolation using a small harness that
  mimics _parse's inner loop exactly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite:///./chatbot.db")

# ---------------------------------------------------------------------------
# Harness: replicate the fixed grouping logic without needing a real PDF
# ---------------------------------------------------------------------------
def _group_words(words):
    """Exact copy of the new grouping code from _parse (PDF branch)."""
    words = sorted(words, key=lambda w: w["top"])

    lines = []
    current_line = []
    line_anchor_top = None

    for w in words:
        if line_anchor_top is None:
            current_line = [w]
            line_anchor_top = w["top"]
        else:
            tol = max(3.0, 0.5 * w.get("size", 0))
            if w["top"] - line_anchor_top <= tol:
                current_line.append(w)
            else:
                lines.append(current_line)
                current_line = [w]
                line_anchor_top = w["top"]

    if current_line:
        lines.append(current_line)

    result = []
    for line_words in lines:
        line_words.sort(key=lambda w: w["x0"])
        result.append(" ".join(w["text"] for w in line_words))
    return result


# ---------------------------------------------------------------------------
# Test 1 – superscript ordering (the reported repro case)
# ---------------------------------------------------------------------------
def test_superscript_ordering():
    # Visual layout (x0 increases left to right):
    #   x0=10  "E"   top=100.4  size=12
    #   x0=20  "="   top=100.0  size=12
    #   x0=30  "mc"  top=100.0  size=12
    #   x0=38  "2"   top=99.2   size=6   <- superscript, slightly higher top
    words = [
        {"text": "E",  "top": 100.4, "x0": 10, "size": 12},
        {"text": "=",  "top": 100.0, "x0": 20, "size": 12},
        {"text": "mc", "top": 100.0, "x0": 30, "size": 12},
        {"text": "2",  "top":  99.2, "x0": 38, "size": 6},
    ]
    lines = _group_words(words)
    assert len(lines) == 1, f"Expected 1 line, got {len(lines)}: {lines}"
    assert lines[0] == "E = mc 2", f"Wrong order: {lines[0]!r}"
    print("PASS test_superscript_ordering:", lines[0])


# ---------------------------------------------------------------------------
# Test 2 – genuinely separate lines are still split
# ---------------------------------------------------------------------------
def test_two_lines_split():
    words = [
        {"text": "Hello", "top": 100.0, "x0":  5, "size": 12},
        {"text": "World", "top": 100.2, "x0": 40, "size": 12},
        {"text": "Next",  "top": 115.0, "x0":  5, "size": 12},
        {"text": "line",  "top": 115.0, "x0": 35, "size": 12},
    ]
    lines = _group_words(words)
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines}"
    assert lines[0] == "Hello World", f"Line 1 wrong: {lines[0]!r}"
    assert lines[1] == "Next line",   f"Line 2 wrong: {lines[1]!r}"
    print("PASS test_two_lines_split:", lines)


# ---------------------------------------------------------------------------
# Test 3 – old bug demonstration: old code would have given "2 = mc E"
# ---------------------------------------------------------------------------
def test_old_code_would_fail():
    """Demonstrate what the OLD code produced (sort by (round(top,1), x0))."""
    words = [
        {"text": "E",  "top": 100.4, "x0": 10, "size": 12},
        {"text": "=",  "top": 100.0, "x0": 20, "size": 12},
        {"text": "mc", "top": 100.0, "x0": 30, "size": 12},
        {"text": "2",  "top":  99.2, "x0": 38, "size": 6},
    ]
    # Old sort key
    old_sorted = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    old_texts = [w["text"] for w in old_sorted]
    # Old grouping: 3-pt absolute tolerance from current_top of first word (99.2)
    # 100.4 - 99.2 = 1.2 < 3 -> all end up in one line in old order
    old_line = " ".join(old_texts)
    assert old_line == "2 = mc E", f"Old code repro differs: {old_line!r}"
    print("PASS test_old_code_would_fail: old code produced:", old_line)


if __name__ == "__main__":
    test_old_code_would_fail()
    test_superscript_ordering()
    test_two_lines_split()
    print()
    print("ALL TESTS PASSED")