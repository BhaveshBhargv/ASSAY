"""
splitter.py — recursive character text splitter (no external dependency).

Splits long text (e.g. parsed PDF pages) into overlapping chunks, preferring
natural boundaries (paragraph → line → sentence → word). Curated corpus passages
are short and typically pass through as a single chunk.
"""
from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = " ".join(text.split()) if "\n" not in text else text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        pieces = self._split_recursive(text, 0)
        return self._merge(pieces)

    # ------------------------------------------------------------------ #
    def _split_recursive(self, text: str, sep_idx: int) -> list[str]:
        if len(text) <= self.chunk_size or sep_idx >= len(_SEPARATORS):
            return [text]
        sep = _SEPARATORS[sep_idx]
        parts = text.split(sep)
        out: list[str] = []
        for p in parts:
            piece = p + sep
            if len(piece) > self.chunk_size:
                out.extend(self._split_recursive(piece, sep_idx + 1))
            else:
                out.append(piece)
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        cur = ""
        for p in pieces:
            if len(cur) + len(p) <= self.chunk_size:
                cur += p
            else:
                if cur.strip():
                    chunks.append(cur.strip())
                # carry overlap tail into the next chunk
                tail = cur[-self.chunk_overlap:] if self.chunk_overlap else ""
                cur = tail + p
        if cur.strip():
            chunks.append(cur.strip())
        return chunks
