#!/usr/bin/env python
"""Build the formatted PDF and DOCX from the Markdown source.

The scientific Markdown is never edited.  A temporary body is generated with
the title block removed and figure captions attached to their figures; a Typst
wrapper supplies the cover, author-information page, contents and running
headers.  DOCX post-processing supplies the same front matter, a Word TOC field,
headers, footers and page numbering.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
SOURCE = PAPER / "chicago_speed_cameras_research.md"
PDF = PAPER / "chicago_speed_cameras_research.pdf"
DOCX = PAPER / "chicago_speed_cameras_research.docx"
TEMPLATE = PAPER / "templates" / "working_paper.typ"
REPO = "https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-"
TOC_ENTRIES = [
    "Abstract",
    "1. Introduction",
    "2. Related work",
    "3. Data and study setting",
    "4. Causal identification strategy",
    "5. Geospatial matching and treatment definition",
    "6. Statistical models",
    "7. Causal results",
    "8. Robustness and sensitivity analysis",
    "9. Crash forecasting using LightGBM",
    "10. Prediction evaluation and uncertainty",
    "11. Discussion",
    "12. Limitations",
    "13. Conclusion",
    "References",
    "Appendix A — Reproducibility",
]

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
CHARCOAL = RGBColor(0x2B, 0x2F, 0x33)
MUTED = RGBColor(0x5D, 0x67, 0x72)


def prepare_body() -> tuple[Path, Path]:
    text = SOURCE.read_text()
    text = text[text.index("## Abstract"):]
    # Pandoc's default Markdown reader does not retain \[...\] math without
    # the double-backslash extension. Convert the source delimiters to the
    # portable $...$ / $$...$$ forms without changing the equations.
    text = re.sub(r"\\\[(.*?)\\\]", lambda match: f"$${match.group(1)}$$", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", lambda match: f"${match.group(1)}$", text, flags=re.S)
    pattern = re.compile(
        r"!\[[^\]]*\]\((figures/[^)]+)\)\n\n\*((?:Figure \d+\..*?))\*\n",
        re.S,
    )

    def combine_pdf(match: re.Match) -> str:
        caption = re.sub(r"^Figure \d+\.\s*", "", match.group(2))
        return f"![{caption}]({match.group(1)})\n"

    def combine_docx(match: re.Match) -> str:
        return f"![{match.group(2)}]({match.group(1)})\n"

    pdf_text, pdf_count = pattern.subn(combine_pdf, text)
    docx_text, docx_count = pattern.subn(combine_docx, text)
    if pdf_count != 11 or docx_count != 11:
        raise RuntimeError(f"expected 11 figure captions; PDF {pdf_count}, DOCX {docx_count}")
    BUILD.mkdir(parents=True, exist_ok=True)
    pdf_body = BUILD / "body_pdf.md"
    docx_body = BUILD / "body_docx.md"
    pdf_body.write_text(pdf_text)
    docx_body.write_text(docx_text)
    return pdf_body, docx_body


def run(command: list[str], cwd: Path = PAPER) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_pdf(body: Path) -> None:
    body_typ = BUILD / "body.typ"
    run([
        "/opt/homebrew/bin/pandoc", str(body), "-t", "typst",
        "--resource-path", str(PAPER), "-o", str(body_typ),
    ], cwd=PAPER)
    typ = body_typ.read_text().replace('image("figures/', 'image("../figures/')
    # Pandoc emits the source image size; make every research figure fill the
    # text measure so small labels remain legible in print and on screen.
    typ = re.sub(r'image\("(\.\./figures/[^"]+)", alt:', r'image("\1", width: 6.2in, alt:', typ)
    body_typ.write_text(typ)
    wrapper = BUILD / "paper.typ"
    wrapper.write_text(TEMPLATE.read_text())
    run([
        "/opt/homebrew/bin/typst", "compile", str(wrapper), str(PDF),
        "--root", str(PAPER),
    ], cwd=PAPER)


def _style(document: Document, name: str, *, size: float, bold: bool = False, color=CHARCOAL, font="Georgia") -> None:
    style = document.styles[name]
    style.font.name = font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = color
    style.font.italic = False


def make_reference() -> None:
    base = BUILD / "reference_base.docx"
    with base.open("wb") as handle:
        subprocess.run(
            ["/opt/homebrew/bin/pandoc", "--print-default-data-file", "reference.docx"],
            stdout=handle, check=True,
        )
    document = Document(base)
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.92)
    section.bottom_margin = Inches(0.92)
    section.left_margin = Inches(1.02)
    section.right_margin = Inches(1.02)
    _style(document, "Normal", size=10.3)
    _style(document, "Body Text", size=10.3)
    _style(document, "First Paragraph", size=10.3)
    _style(document, "Heading 1", size=18, bold=True, color=NAVY, font="Avenir Next")
    _style(document, "Heading 2", size=14.5, bold=True, color=NAVY, font="Avenir Next")
    _style(document, "Heading 3", size=11.2, bold=True, color=CHARCOAL, font="Avenir Next")
    _style(document, "Title", size=26, bold=True, color=NAVY)
    _style(document, "Subtitle", size=14, color=CHARCOAL)
    _style(document, "Caption", size=9, color=MUTED)
    document.save(BUILD / "reference.docx")


def _set_cell_shading(paragraph, color: str) -> None:
    pass


def _base_run(paragraph, text: str, *, size: float, bold: bool = False, color=CHARCOAL, font="Avenir Next", spacing: bool = False):
    run = paragraph.add_run(text)
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.all_caps = spacing
    return run


def _insert_paragraph_before(document: Document, first, paragraph) -> None:
    first._p.addprevious(paragraph._p)


def _move_to_start(document: Document, paragraph) -> None:
    body = document.element.body
    body.insert(0, paragraph._p)


def _page_break(document: Document):
    p = document.add_paragraph()
    p.add_run().add_break()
    # A page break is a run break; use explicit page break for Word.
    run = p.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)
    return p


def _field(paragraph, instruction: str, placeholder: str = "") -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    run._r.append(begin)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    run._r.append(instr)
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    run._r.append(separate)
    if placeholder:
        paragraph.add_run(placeholder)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    paragraph.add_run()._r.append(end)


def _add_page_break_before(document: Document, target) -> None:
    p = document.add_paragraph()
    run = p.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)
    target._p.addprevious(p._p)


def add_front_matter(document: Document) -> None:
    # Cover
    cover = document.add_paragraph()
    cover.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cover.paragraph_format.space_before = Pt(24)
    _base_run(cover, "RESEARCH WORKING PAPER", size=8.5, color=MUTED)
    cover.add_run().add_break()
    _base_run(cover, "________________________________________________________________", size=10, color=NAVY)
    for _ in range(5):
        cover.add_run().add_break()
    _base_run(cover, "Do Speed Cameras", size=30, bold=True, color=NAVY, font="Georgia")
    cover.add_run().add_break()
    _base_run(cover, "Make Chicago Safer?", size=30, bold=True, color=NAVY, font="Georgia")
    for _ in range(2):
        cover.add_run().add_break()
    _base_run(cover, "A Geospatial Causal Inference and Machine Learning Study", size=15, color=CHARCOAL, font="Georgia")
    for _ in range(4):
        cover.add_run().add_break()
    _base_run(cover, "Saketh Donepudi", size=12.5, bold=True, color=CHARCOAL, font="Georgia")
    cover.add_run().add_break()
    _base_run(cover, "Amrutha Kakumani", size=12.5, bold=True, color=CHARCOAL, font="Georgia")
    for _ in range(3):
        cover.add_run().add_break()
    _base_run(cover, "October 2026  ·  Chicago, Illinois — Study Location", size=8.8, color=MUTED)
    cover.add_run().add_break()
    _base_run(cover, "Independent research working paper · Not peer reviewed", size=8.8, color=MUTED)
    cover.add_run().add_break()
    cover.add_run().add_picture(str(PAPER / "figures" / "cover_street_grid.png"), width=Inches(3.2))
    _move_to_start(document, cover)

    # Author page
    author = document.add_paragraph()
    author.paragraph_format.page_break_before = True
    author.paragraph_format.space_before = Pt(20)
    _base_run(author, "AUTHORS & PUBLICATION INFORMATION", size=8.5, color=MUTED)
    author.add_run().add_break()
    _base_run(author, "______________________________", size=9, color=NAVY)
    for _ in range(4):
        author.add_run().add_break()
    _base_run(author, "Saketh Donepudi", size=18, bold=True, color=NAVY, font="Georgia")
    author.add_run().add_break()
    _base_run(author, "Co-author", size=10, color=MUTED)
    for _ in range(2):
        author.add_run().add_break()
    _base_run(author, "Amrutha Kakumani", size=18, bold=True, color=NAVY, font="Georgia")
    author.add_run().add_break()
    _base_run(author, "Co-author", size=10, color=MUTED)
    for _ in range(4):
        author.add_run().add_break()
    _base_run(author, "Affiliations, verified individual contributions, and correspondence details are withheld pending written approval from both co-authors. No university affiliation or contribution is asserted.", size=9.2, color=MUTED)
    author.add_run().add_break()
    author.add_run().add_break()
    _base_run(author, "“This manuscript is an independent research working paper and has not undergone formal peer review.”", size=9.6, color=CHARCOAL, font="Georgia")
    author.add_run().add_break()
    author.add_run().add_break()
    _base_run(author, "Repository", size=8.8, color=MUTED)
    author.add_run().add_break()
    _base_run(author, REPO, size=8.8, color=MUTED)
    _move_to_start(document, author)
    # Move author after cover; page_break_before keeps the author page separate.
    cover._p.addnext(author._p)


def add_toc(document: Document, toc_pages: dict[str, int] | None = None) -> None:
    """Insert a Word TOC field, optionally with a cached visible result.

    Word updates the field on open; the cached result makes the DOCX preview
    useful in LibreOffice as well.
    """
    first_body_heading = next(p for p in document.paragraphs if p.style.name == "Heading 2")
    toc_heading = document.add_paragraph("Contents")
    toc_heading.style = document.styles["Heading 1"]
    toc_heading.paragraph_format.page_break_before = True
    toc_heading.paragraph_format.space_before = Pt(12)
    toc_heading.paragraph_format.space_after = Pt(8)
    toc = document.add_paragraph()
    toc.paragraph_format.tab_stops.add_tab_stop(
        Inches(6.2), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
    )
    run = toc.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin"); begin.set(qn("w:dirty"), "true"); run._r.append(begin)
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = 'TOC \\o "1-2" \\h \\z \\u'; run._r.append(instr)
    separate = OxmlElement("w:fldChar"); separate.set(qn("w:fldCharType"), "separate"); run._r.append(separate)
    if toc_pages:
        for index, title in enumerate(TOC_ENTRIES):
            page = toc_pages.get(title)
            if page is None:
                continue
            if index:
                toc.add_run().add_break()
            line = toc.add_run(title)
            line.font.name = "Georgia"
            line.font.size = Pt(9.4)
            tab = OxmlElement("w:tab")
            toc.add_run()._r.append(tab)
            number = toc.add_run(str(page))
            number.font.name = "Georgia"
            number.font.size = Pt(9.4)
    else:
        toc.add_run("Right-click and choose Update Field to build the table of contents.")
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    toc.add_run()._r.append(end)
    first_body_heading._p.addprevious(toc_heading._p)
    first_body_heading._p.addprevious(toc._p)


def _add_page_number_field(paragraph) -> None:
    _field(paragraph, "PAGE", "1")


def add_body_section(document: Document) -> None:
    """Separate unnumbered front matter from the numbered main matter."""
    first_body_heading = next(p for p in document.paragraphs if p.style.name == "Heading 2")
    sectPr = deepcopy(document.sections[-1]._sectPr)
    section_type = OxmlElement("w:type")
    section_type.set(qn("w:val"), "nextPage")
    sectPr.insert(0, section_type)
    paragraph = OxmlElement("w:p")
    paragraph_properties = OxmlElement("w:pPr")
    paragraph_properties.append(sectPr)
    paragraph.append(paragraph_properties)
    first_body_heading._p.addprevious(paragraph)


def add_headers_footers(document: Document) -> None:
    # Section 1 is the unnumbered cover/author/TOC; section 2 is the body.
    front, body = document.sections
    for section, numbered in ((front, False), (body, True)):
        section.different_first_page_header_footer = False
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False
        for paragraph in section.header.paragraphs:
            paragraph.text = ""
        for paragraph in section.footer.paragraphs:
            paragraph.text = ""
        if numbered:
            p = section.header.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _base_run(p, "Do Speed Cameras Make Chicago Safer?  ·  Donepudi & Kakumani", size=7.8, color=MUTED)
            fp = section.footer.paragraphs[0]
            fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_page_number_field(fp)
    sectPr = body._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:start"), "1")
    settings = document.settings.element
    if settings.find(qn("w:updateFields")) is None:
        update_fields = OxmlElement("w:updateFields")
        update_fields.set(qn("w:val"), "true")
        settings.append(update_fields)


def build_docx(body: Path, toc_pages: dict[str, int] | None = None) -> None:
    make_reference()
    run([
        "/opt/homebrew/bin/pandoc", str(body), "-o", str(BUILD / "body.docx"),
        "--reference-doc", str(BUILD / "reference.docx"),
        "--resource-path", str(PAPER),
    ], cwd=PAPER)
    document = Document(BUILD / "body.docx")
    add_front_matter(document)
    add_toc(document, toc_pages=toc_pages)
    add_body_section(document)
    add_headers_footers(document)
    # Keep images with their captions and table labels with their tables.
    for paragraph in document.paragraphs:
        if paragraph._p.xpath(".//w:drawing"):
            paragraph.paragraph_format.keep_with_next = True
        if paragraph.text.strip().startswith("Table "):
            paragraph.paragraph_format.keep_with_next = True
    document.save(DOCX)


def _render_docx_pages(source: Path) -> dict[str, int]:
    """Render DOCX through LibreOffice and map section headings to body pages."""
    try:
        preview_docx = BUILD / "docx_preview.docx"
        shutil.copy(source, preview_docx)
        subprocess.run(
            ["/opt/homebrew/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(BUILD), str(preview_docx)],
            check=True, capture_output=True,
        )
        from pypdf import PdfReader

        reader = PdfReader(BUILD / "docx_preview.pdf")
        pages: dict[str, int] = {}
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for heading in TOC_ENTRIES:
                if heading in text and heading not in pages:
                    pages[heading] = page_number - 3
        return pages
    except Exception:
        return {}


def main() -> int:
    pdf_body, docx_body = prepare_body()
    build_pdf(pdf_body)
    build_docx(docx_body)
    toc_pages = _render_docx_pages(DOCX)
    if toc_pages:
        build_docx(docx_body, toc_pages=toc_pages)
        # Re-render once so the final DOCX is internally consistent.
        _render_docx_pages(DOCX)
    print(f"wrote {PDF} and {DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
