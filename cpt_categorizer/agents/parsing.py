import re


class ParsingAgent:
    """Lightweight text cleanup before classification."""

    _WHITESPACE_RE = re.compile(r"\s+")

    def parse(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        cleaned = raw_text.strip()
        cleaned = self._WHITESPACE_RE.sub(" ", cleaned)
        return cleaned
