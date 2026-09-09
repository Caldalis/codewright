"""Mid-output truncation behavior."""

from __future__ import annotations

import pytest

from codewright.tools.truncate import truncate_middle


class TestTruncateMiddle:
    def test_short_text_unchanged(self):
        assert truncate_middle("hello", max_chars=100) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 5000
        result = truncate_middle(text, max_chars=1000)
        assert len(result) < len(text)
        assert "chars truncated" in result
        # Head/tail preserved
        assert result.startswith("a")
        assert result.endswith("a")

    def test_emoji_boundary_safe(self):
        emoji = "🦀"  # 1 code point, surrogate pair when encoded UTF-16
        text = emoji * 5000
        result = truncate_middle(text, max_chars=500)
        # Result must remain a valid Python string (test by re-encoding)
        result.encode("utf-8")
        assert "chars truncated" in result

    def test_cjk_boundary_safe(self):
        text = "中" * 5000
        result = truncate_middle(text, max_chars=500)
        result.encode("utf-8")
        assert "chars truncated" in result
        # Head + tail both contain the CJK char (no mojibake)
        assert "中" in result.split("…")[0]

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            truncate_middle("x", max_chars=0)
