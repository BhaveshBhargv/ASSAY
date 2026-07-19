"""
models.py — RAG domain value objects.

A GuidelineChunk is one retrievable passage of evidence-based guidance with full
provenance (source + code + URL) so every downstream recommendation can be cited.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GuidelineChunk:
    chunk_id: str
    text: str
    source: str                       # NICE | NHS | WHO
    code: str = ""                    # e.g. NG28, CG181, "Vitamin D"
    title: str = ""
    biomarkers: tuple[str, ...] = ()  # canonical codes this passage relates to
    categories: tuple[str, ...] = ()  # diet | physical_activity | alcohol | ...
    severities: tuple[str, ...] = ()  # normal | borderline | serious | all
    url: str = ""

    def citation(self) -> str:
        bits = [b for b in (self.source, self.code) if b]
        return " ".join(bits) if bits else self.source

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunk: GuidelineChunk
    score: float

    def to_dict(self) -> dict:
        return {"score": round(float(self.score), 4), **self.chunk.to_dict()}


@dataclass
class RetrievalQuery:
    text: str
    biomarkers: tuple[str, ...] = ()
    severity: str = ""
    metadata: dict = field(default_factory=dict)
