import re
from collections import Counter

import pymupdf
from pypdf import PdfReader

_WATERMARK = re.compile(r"ocean\s*of\s*pdf\s*\.\s*com", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SOFT_WRAP = re.compile(r"(?<!\n)\n(?!\n)")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")


def normalize_text(raw: str) -> str:
    text = _WATERMARK.sub(" ", raw)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _PARAGRAPH_BREAK.sub("\n\n", text)
    text = _SOFT_WRAP.sub(" ", text)
    text = _REPEATED_SPACE.sub(" ", text)
    return text.strip()


def load_pdf(filepath: str, normalize: bool = True) -> str:
    reader = PdfReader(filepath)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    raw = "\n".join(text_parts)
    return normalize_text(raw) if normalize else raw


HEADING_SIZE_RATIO = 1.25  # text this much larger than the body font counts as a heading
MAX_HEADING_CHARS = 80
MIN_IMAGE_SIDE = 16  # skip spacer pixels and tiny bullets
REPEATED_PICTURE_PAGES = 3  # a picture reused on this many pages is a template ornament
_SENTENCE_END = (".", "!", "?", "”", '"', "…")


def _looks_like_heading(text: str) -> bool:
    # Short and not a sentence: enlarged opening lines ("The snake pit is crowded tonight.")
    # are styled like headings but end like prose.
    return bool(text) and len(text) <= MAX_HEADING_CHARS and not text.endswith(_SENTENCE_END)


def _body_font_size(doc) -> float:
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _image_png(doc, xref: int) -> bytes:
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:  # CMYK and similar: convert so the PNG encoder accepts it
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix.tobytes("png")


def _image_page_counts(doc) -> Counter:
    counts: Counter = Counter()
    for page in doc:
        counts.update({info["xref"] for info in page.get_image_info(xrefs=True)})
    return counts


def _page_items(
    doc, page, body_size: float, read_images: bool, image_pages: Counter, keep_images: bool = False
) -> list[dict]:
    """Text blocks and images on one page, top to bottom, each marked heading or body.

    With keep_images, content images (pictures and scanned text, not headings or
    ornaments) are saved, and their captions carry the saved image's ID.
    """
    items = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        spans = [span for line in block["lines"] for span in line["spans"]]
        text = "\n".join("".join(s["text"] for s in line["spans"]) for line in block["lines"])
        if not text.strip():
            continue
        clean = _WATERMARK.sub(" ", text).strip()
        is_heading = (
            _looks_like_heading(clean)
            and max(s["size"] for s in spans) >= body_size * HEADING_SIZE_RATIO
        )
        items.append({"y": block["bbox"][1], "x": block["bbox"][0],
                      "heading": is_heading, "text": " ".join(clean.split()) if is_heading else text})

    if read_images:
        from src.image_index import caption, save_image
        from src.image_reader import read_image

        for info in page.get_image_info(xrefs=True):
            if info["xref"] == 0 or min(info["width"], info["height"]) < MIN_IMAGE_SIDE:
                continue
            png = _image_png(doc, info["xref"])
            result = read_image(png)
            content = " ".join(result["content"].split())
            if result["kind"] == "decorative" or not content:
                continue
            # Repeated text images are kept (they are headings like a narrator's name);
            # a repeated picture is decoration such as a chapter ornament or logo.
            if result["kind"] == "description" and image_pages[info["xref"]] >= REPEATED_PICTURE_PAGES:
                continue
            if result["kind"] == "text" and _looks_like_heading(content):
                item = {"heading": True, "text": content}
            else:
                image_id = save_image(png) if keep_images else None
                item = {"heading": False, "text": caption(image_id, result["kind"], content)}
                if image_id:
                    item["image"] = {"id": image_id, "page": page.number + 1, "caption": content}
            items.append({"y": info["bbox"][1], "x": info["bbox"][0], **item})

    return sorted(items, key=lambda item: (item["y"], item["x"]))


def load_pdf_sections(
    filepath: str, read_images: bool = True, images: list[dict] | None = None
) -> list[dict]:
    """Split a PDF into sections at its headings: [{"heading": str | None, "text": str}].

    Headings come from large text or, with read_images, from images whose content is
    short text (decorative chapter titles, names). All heading pieces on one page are
    joined, so "Chapter One" above an epigraph and "Daria" below it become one heading.
    If an `images` list is given, content images are saved and appended to it as
    {"id", "page", "caption", "heading"}.
    """
    doc = pymupdf.open(filepath)
    body_size = _body_font_size(doc)
    image_pages = _image_page_counts(doc) if read_images else Counter()
    sections = [{"heading": None, "parts": []}]

    for page in doc:
        items = _page_items(doc, page, body_size, read_images, image_pages, images is not None)
        headings = [item for item in items if item["heading"]]
        first = items.index(headings[0]) if headings else len(items)
        before, after = items[:first], [item for item in items[first:] if not item["heading"]]
        sections[-1]["parts"].extend(item["text"] for item in before)
        if headings:
            sections.append({"heading": " · ".join(item["text"] for item in headings), "parts": []})
        sections[-1]["parts"].extend(item["text"] for item in after)
        if images is not None:
            for item in items:
                if "image" in item:
                    in_new = headings and item in after
                    heading = sections[-1]["heading"] if in_new or not headings else sections[-2]["heading"]
                    images.append({**item["image"], "heading": heading})

    result = []
    for section in sections:
        text = normalize_text("\n".join(section["parts"]))
        if text:
            result.append({"heading": section["heading"], "text": text})
    return result


if __name__ == "__main__":
    raw = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf", normalize=False)
    clean = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    print(f"raw characters      : {len(raw)}")
    print(f"normalized characters: {len(clean)}")
    print(f"raw newlines        : {raw.count(chr(10))}")
    print(f"normalized newlines : {clean.count(chr(10))}")
    print(f"watermarks removed  : {len(_WATERMARK.findall(raw))}")
    print("\n--- sample ---")
    print(clean[20000:20600])
