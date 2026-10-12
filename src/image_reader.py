import hashlib
import json
import os

from google.genai import types
from pydantic import BaseModel

from src.generator import GENERATOR_MODEL, get_client

CACHE_PATH = os.path.join(".cache", "image_text.json")

READ_PROMPT = (
    "This image comes from a page of a document. Classify it and extract its content.\n"
    '- "text": the image mainly shows words (a heading, a name, a title, a scanned '
    "paragraph). Put the exact words in content, preserving their order. Do not add "
    "words that are not visible.\n"
    '- "description": the image shows a figure, chart, diagram, table, or photo. Put a '
    "short factual description of what it shows in content, including any labels or "
    "numbers that are visible.\n"
    '- "decorative": an ornament, divider, logo, or flourish with no information. '
    "Leave content empty."
)


class ImageContent(BaseModel):
    kind: str
    content: str


_cache: dict[str, dict] | None = None


def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, indent=2, ensure_ascii=False)


def read_image(png_bytes: bytes) -> dict:
    """Return {"kind": "text" | "description" | "decorative", "content": str}.

    Results are cached by image hash, so repeated ornaments and re-runs cost nothing.
    """
    cache = _load_cache()
    key = hashlib.sha256(GENERATOR_MODEL.encode() + READ_PROMPT.encode() + png_bytes).hexdigest()[:20]
    if key in cache:
        return cache[key]

    response = get_client().models.generate_content(
        model=GENERATOR_MODEL,
        contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), READ_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ImageContent,
        ),
    )
    parsed = response.parsed
    if parsed is None or parsed.kind not in ("text", "description", "decorative"):
        result = {"kind": "decorative", "content": ""}
    else:
        result = {"kind": parsed.kind, "content": parsed.content.strip()}

    cache[key] = result
    _save_cache()
    return result
