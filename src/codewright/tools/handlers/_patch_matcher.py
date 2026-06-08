from __future__ import annotations

import unicodedata


def find_hunk(
    file_lines: list[str],
    anchors: list[str],
    old_lines: list[str],
    is_end_of_file: bool = False,
) -> int | None:

    start = 0
    for anchor in anchors:
        idx = _find_anchor(file_lines, anchor, start)
        if idx is None:
            return None
        start = idx + 1

    return seek_sequence(file_lines, old_lines, start, is_end_of_file)


def seek_sequence(
    file_lines: list[str],
    pattern: list[str],
    start: int = 0,
    is_end_of_file: bool = False,
) -> int | None:
    return _match_within(file_lines, pattern, start, is_end_of_file)


def _find_anchor(lines: list[str], anchor: str, start: int) -> int | None:
    needle = anchor.strip()
    if not needle:
        return start
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return None


def _match_within(
    file_lines: list[str],
    pattern: list[str],
    start: int,
    is_end_of_file: bool,
) -> int | None:
    if not pattern:
        return start
    if len(pattern) > len(file_lines) - start:

        return None


    end_anchor = len(file_lines) - len(pattern)
    candidates: list[int] = []
    if is_end_of_file and end_anchor >= start:
        candidates.append(end_anchor)
    candidates.extend(range(start, end_anchor + 1))

    seen: set[int] = set()
    ordered: list[int] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for level in (_match_exact, _match_rstrip, _match_strip, _match_normalized):
        for i in ordered:
            if level(file_lines, pattern, i):
                return i
    return None


def _match_exact(file_lines: list[str], pattern: list[str], i: int) -> bool:
    return all(file_lines[i + k] == p for k, p in enumerate(pattern))


def _match_rstrip(file_lines: list[str], pattern: list[str], i: int) -> bool:
    return all(file_lines[i + k].rstrip() == p.rstrip() for k, p in enumerate(pattern))


def _match_strip(file_lines: list[str], pattern: list[str], i: int) -> bool:
    return all(file_lines[i + k].strip() == p.strip() for k, p in enumerate(pattern))


def _match_normalized(file_lines: list[str], pattern: list[str], i: int) -> bool:
    return all(
        unicode_normalize(file_lines[i + k]) == unicode_normalize(p)
        for k, p in enumerate(pattern)
    )



_DASH_CODEPOINTS = "‐‑‒–—―−"
_SINGLE_QUOTE_CODEPOINTS = "‘’‚‛"
_DOUBLE_QUOTE_CODEPOINTS = "“”„‟"
_FANCY_SPACE_CODEPOINTS = (
    "        "
    "    　"
)


def unicode_normalize(s: str) -> str:

    out: list[str] = []
    for ch in s.strip():
        if ch in _DASH_CODEPOINTS:
            out.append("-")
        elif ch in _SINGLE_QUOTE_CODEPOINTS:
            out.append("'")
        elif ch in _DOUBLE_QUOTE_CODEPOINTS:
            out.append('"')
        elif ch in _FANCY_SPACE_CODEPOINTS:
            out.append(" ")
        else:
            out.append(ch)
    return unicodedata.normalize("NFKC", "".join(out))
