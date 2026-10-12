import os
import re

from src.loader import load_pdf_sections, normalize_text

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".md"} | IMAGE_TYPES

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_DOCX_HEADING_STYLE = re.compile(r"^Heading (\d)$")


class _Sections:
    """Collects text into sections, keeping a heading path like "Policies · Refunds"."""

    def __init__(self):
        self.path: list[tuple[int, str]] = []
        self.sections = [{"heading": None, "parts": []}]

    def heading(self, level: int, text: str) -> None:
        self.path = [(lvl, t) for lvl, t in self.path if lvl < level] + [(level, text)]
        self.sections.append({"heading": " · ".join(t for _, t in self.path), "parts": []})

    def add(self, text: str) -> None:
        if text.strip():
            self.sections[-1]["parts"].append(text)

    def result(self) -> list[dict]:
        out = []
        for section in self.sections:
            text = normalize_text("\n\n".join(section["parts"]))
            if text:
                out.append({"heading": section["heading"], "text": text})
        return out


def _image_text(image_bytes: bytes) -> str:
    from src.image_reader import read_image, to_png

    result = read_image(to_png(image_bytes))
    content = " ".join(result["content"].split())
    if result["kind"] == "decorative" or not content:
        return ""
    if result["kind"] == "text":
        return f"[Image text: {content}]"
    return f"[Image: {content}]"


def _load_image(path: str, read_images: bool) -> list[dict]:
    if not read_images:
        return []
    from src.image_reader import read_image, to_png

    with open(path, "rb") as f:
        result = read_image(to_png(f.read()))
    content = result["content"].strip()
    if result["kind"] == "decorative" or not content:
        return []
    # A photographed or scanned page is the document itself, so keep its words as plain text.
    text = content if result["kind"] == "text" else f"[Image: {content}]"
    return [{"heading": None, "text": text}]


def _load_text(path: str) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    sections = _Sections()
    markdown = path.lower().endswith(".md")
    for line in lines:
        match = _MD_HEADING.match(line) if markdown else None
        if match:
            sections.heading(len(match.group(1)), match.group(2))
        else:
            sections.add(line + "\n")
    # Lines were added one by one; rejoin each section's lines before normalizing.
    for section in sections.sections:
        section["parts"] = ["".join(section["parts"])]
    return sections.result()


def _load_docx(path: str, read_images: bool) -> list[dict]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    sections = _Sections()
    blip = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

    # Walk the body in order so headings, paragraphs, tables, and images stay in sequence.
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, document)
            style = paragraph.style.name if paragraph.style is not None else ""
            level = 0 if style == "Title" else None  # the title heads every section path
            match = _DOCX_HEADING_STYLE.match(style)
            if match:
                level = int(match.group(1))
            if level is not None and paragraph.text.strip():
                sections.heading(level, paragraph.text.strip())
            else:
                sections.add(paragraph.text)
            if read_images:
                for node in element.iter(blip):
                    part = document.part.related_parts.get(node.get(embed))
                    if part is not None:
                        sections.add(_image_text(part.blob))
        elif tag == "tbl":
            sections.add(_table_text(Table(element, document)))
    return sections.result()


def _table_text(table) -> str:
    # Write each row with its column names ("Product: Chairs; Price: $40") so a row still
    # makes sense on its own after chunking splits the table apart.
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if len(rows) < 2:
        return "\n\n".join("; ".join(row) for row in rows)
    header, body = rows[0], rows[1:]
    return "\n\n".join(
        "; ".join(f"{name}: {value}" if name else value for name, value in zip(header, row))
        for row in body
    )


def load_document(path: str, read_images: bool = True) -> list[dict]:
    """Read a supported file into sections: [{"heading": str | None, "text": str}]."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_sections(path, read_images=read_images)
    if ext == ".docx":
        return _load_docx(path, read_images)
    if ext in (".txt", ".md"):
        return _load_text(path)
    if ext in IMAGE_TYPES:
        return _load_image(path, read_images)
    raise ValueError(
        f"Unsupported file type '{ext}' for {path}. Supported: {', '.join(sorted(SUPPORTED_TYPES))}"
    )
