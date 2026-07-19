"""
report_pdf.py — render a downloadable PDF of the assessment + recommendations.

Uses fpdf2 with core fonts (no bundled font files). Text is sanitised to the
latin-1 range core fonts support. Works with or without the LLM recommendations,
so a rules-only assessment is still exportable.

Cursor positioning is explicit (`new_x`/`new_y` on every write) rather than relying
on fpdf2's version-dependent defaults, and each block starts at the left margin —
this avoids the "not enough horizontal space" errors that come from a drifting X.

    build_pdf(demographics, rule_result, fused, evidence, report, audit, disclaimer) -> bytes
"""
from __future__ import annotations

from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.recommend.report import ADVICE_SECTIONS, SECTION_TITLES

INK = (16, 32, 46)
TEAL = (14, 124, 134)
MUTED = (107, 123, 134)
SEV_RGB = {"normal": (31, 157, 116), "borderline": (200, 135, 27), "serious": (194, 74, 87)}

# advance to the left margin on the next line (the safe default for paragraphs)
_NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
# stay on the same line, moving to the right of the cell (for table columns)
_SAME = dict(new_x=XPos.RIGHT, new_y=YPos.TOP)

_REPLACERS = {
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "-", "≥": ">=", "≤": "<=",
    "→": "->", "µ": "u", "°": " deg", "…": "...", "▲": "!",
}


def _san(text) -> str:
    s = str(text)
    for bad, good in _REPLACERS.items():
        s = s.replace(bad, good)
    return s.encode("latin-1", "replace").decode("latin-1")


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


class _PDF(FPDF):
    def header(self):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*MUTED)
        half = self.epw / 2
        self.cell(half, 6, _san("Personalised Lifestyle Recommendations"), align="L", **_SAME)
        self.cell(half, 6, _san(f"Generated {date.today().isoformat()}"), align="R", **_NL)
        self.set_draw_color(219, 227, 232)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-14)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*MUTED)
        self.multi_cell(self.epw, 3.5, _san(
            "Not a medical diagnosis. This report provides general, evidence-based "
            "lifestyle information and does not replace advice from a qualified "
            "healthcare professional."), **_NL)

    # helpers -------------------------------------------------------------- #
    def para(self, text, h=5, size=10, style="", color=INK):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.multi_cell(self.epw, h, _san(text), **_NL)

    def eyebrow(self, label):
        self.ln(3)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*TEAL)
        self.cell(self.epw, 5, _san(label.upper()), **_NL)
        self.ln(1)
        self.set_text_color(*INK)


def build_pdf(demographics, rule_result, fused, evidence, report, audit, disclaimer) -> bytes:
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()

    # --- verdict ----------------------------------------------------------- #
    sev = fused.severity
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*SEV_RGB.get(sev, INK))
    pdf.cell(pdf.epw, 11, _san(f"Overall risk pattern: {sev.title()}"), **_NL)

    age, sx = demographics.get("age", "-"), demographics.get("sex", "-")
    line = f"Patient: age {age}, {sx}.   Rules: {fused.rule_severity}   |   Model: {fused.rf_severity or 'n/a'}"
    if fused.escalated_by_rf:
        line += "   (model escalated on pattern)"
    pdf.para(line, h=6, size=10, color=MUTED)
    if fused.urgent_referral:
        pdf.para("Some results warrant prompt clinical attention - please contact your GP "
                 "or an appropriate service.", h=5, size=10, style="B", color=SEV_RGB["serious"])

    # --- severity table ---------------------------------------------------- #
    pdf.eyebrow("Severity table")
    widths = (58, 34, 40, 36)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(238, 242, 244)
    pdf.set_text_color(*INK)
    for title, w, last in zip(("Biomarker", "Value", "Reference", "Status"), widths,
                              (False, False, False, True)):
        pdf.cell(w, 7, _san(title), fill=True, **(_NL if last else _SAME))

    rows = sorted(rule_result.biomarkers,
                  key=lambda b: ({"serious": 0, "borderline": 1, "normal": 2}
                                 .get(b.severity.value, 3), b.name))
    for b in rows:
        rng = b.reference_range or {}
        lo, hi = rng.get("low"), rng.get("high")
        ref = f"{_num(lo)}-{_num(hi)}" if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else "-"
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        pdf.cell(widths[0], 6, _san(b.name), **_SAME)
        pdf.cell(widths[1], 6, _san(f"{_num(b.value)} {b.unit}".strip()), **_SAME)
        pdf.cell(widths[2], 6, _san(ref), **_SAME)
        pdf.set_text_color(*SEV_RGB.get(b.severity.value, MUTED))
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(widths[3], 6, _san(b.status.value), **_NL)

    # --- recommendations --------------------------------------------------- #
    if report is not None:
        pdf.eyebrow("What your results suggest")
        pdf.para(report.explanation, h=5, size=10)
        for sec in ADVICE_SECTIONS:
            items = report.section(sec)
            if not items:
                continue
            pdf.eyebrow(SECTION_TITLES[sec])
            for it in items:
                pdf.para(f"- {it.advice}", h=5, size=10, style="B")
                cites = ", ".join(it.evidence)
                pdf.para(f"   Why: {it.rationale}  [Evidence: {cites}]", h=4.5, size=9, color=MUTED)
    else:
        pdf.eyebrow("Recommendations")
        pdf.para("Recommendations were not generated for this report (the language model "
                 "was not run).", h=5, size=10, style="I", color=MUTED)

    # --- clinician signpost ------------------------------------------------ #
    if fused.signpost:
        pdf.eyebrow("Discuss with your clinician")
        names = ", ".join(f"{s['name']} ({s['status']})" for s in fused.signpost)
        pdf.para("Outside usual range, not addressed by lifestyle advice: " + names, h=5, size=9)

    # --- evidence ---------------------------------------------------------- #
    if evidence:
        pdf.eyebrow("Evidence sources")
        for e in evidence:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*TEAL)
            pdf.cell(14, 5, _san(e.id), **_SAME)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*INK)
            title = f" - {e.title}" if e.title else ""
            pdf.multi_cell(pdf.w - pdf.r_margin - pdf.get_x(), 5, _san(f"{e.citation}{title}"), **_NL)
    if audit is not None:
        pdf.ln(2)
        pdf.para(f"Evidence grounding: {round(audit.groundedness * 100)}% of advice items "
                 f"cite a retrieved guideline.", h=4, size=8, style="I", color=MUTED)

    return bytes(pdf.output())
