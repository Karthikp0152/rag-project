import re

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
