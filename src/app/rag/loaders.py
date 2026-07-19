"""
loaders.py — document loaders.

Two sources feed one corpus:
  * curated YAML  — authored, cited, paraphrased guideline passages (primary)
  * PDF folder    — real NICE/NHS/WHO PDFs the user drops in (parsed + chunked)

Both yield GuidelineChunk objects with provenance so retrieval stays citable.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import yaml

from .models import GuidelineChunk
from .splitter import RecursiveTextSplitter

log = logging.getLogger(__name__)


def _chunk_id(source: str, text: str) -> str:
    h = hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source.lower()}-{h}"


def _as_tuple(v) -> tuple[str, ...]:
    if not v:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def load_curated_yaml(path: Path) -> list[GuidelineChunk]:
    """Load the authored corpus: a YAML list of passage entries."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    entries = data.get("passages", [])
    chunks: list[GuidelineChunk] = []
    for e in entries:
        text = e["text"].strip()
        chunks.append(GuidelineChunk(
            chunk_id=e.get("id") or _chunk_id(e.get("source", "NICE"), text),
            text=text,
            source=e.get("source", ""),
            code=str(e.get("code", "")),
            title=e.get("title", ""),
            biomarkers=_as_tuple(e.get("biomarkers")),
            categories=_as_tuple(e.get("categories")),
            severities=_as_tuple(e.get("severities")) or ("all",),
            url=e.get("url", ""),
        ))
    log.info("curated corpus: %d passages from %s", len(chunks), path.name)
    return chunks


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF (PDF parser)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    return "\n".join(pages)


def load_pdf_dir(
    pdf_dir: Path, splitter: RecursiveTextSplitter | None = None
) -> list[GuidelineChunk]:
    """Parse + chunk every PDF in a directory. Metadata is inferred from the
    filename convention '<SOURCE>_<CODE>_<title words>.pdf' (all optional)."""
    if not pdf_dir.exists():
        return []
    splitter = splitter or RecursiveTextSplitter()
    chunks: list[GuidelineChunk] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        source, code, title = _meta_from_filename(pdf.stem)
        text = parse_pdf(pdf)
        for i, piece in enumerate(splitter.split(text)):
            chunks.append(GuidelineChunk(
                chunk_id=f"{pdf.stem}-{i}",
                text=piece,
                source=source, code=code, title=title,
                severities=("all",),
                url=str(pdf.name),
            ))
    if chunks:
        log.info("parsed %d PDF chunk(s) from %s", len(chunks), pdf_dir)
    return chunks


def _meta_from_filename(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    source = parts[0].upper() if parts and parts[0].upper() in {"NICE", "NHS", "WHO"} else ""
    code = parts[1] if len(parts) > 1 else ""
    title = " ".join(parts[2:]) if len(parts) > 2 else stem
    return source, code, title


def load_all(curated_path: Path, pdf_dir: Path | None = None) -> list[GuidelineChunk]:
    chunks = load_curated_yaml(curated_path)
    if pdf_dir is not None:
        chunks += load_pdf_dir(pdf_dir)
    return chunks
