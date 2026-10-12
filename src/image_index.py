import hashlib
import io
import os
import re

import numpy as np
from PIL import Image

IMAGE_DIR = os.path.join(".cache", "images")
CLIP_CACHE_DIR = os.path.join(".cache", "clip")
CLIP_MODEL = "clip-ViT-B-32"
MAX_SIDE = 768  # images sent to the answer model are shrunk to this, which keeps their token cost low

# Image captions in passages carry an ID that links the text back to the saved image.
_IMAGE_REF = re.compile(r"\[Image(?: text)? ([0-9a-f]{12}):")

_clip_model = None


def save_image(png_bytes: bytes) -> str:
    """Store an image once under its content fingerprint and return that ID."""
    image_id = hashlib.sha256(png_bytes).hexdigest()[:12]
    path = image_path(image_id)
    if not os.path.exists(path):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(png_bytes)
    return image_id


def image_path(image_id: str) -> str:
    return os.path.join(IMAGE_DIR, f"{image_id}.png")


def caption(image_id: str | None, kind: str, content: str) -> str:
    label = "Image text" if kind == "text" else "Image"
    return f"[{label} {image_id}: {content}]" if image_id else f"[{label}: {content}]"


def linked_image_ids(passages: list[str]) -> list[str]:
    """IDs of images whose captions appear in the passages, in passage order."""
    ids = []
    for passage in passages:
        for image_id in _IMAGE_REF.findall(passage):
            if image_id not in ids:
                ids.append(image_id)
    return ids


def answer_png(image_id: str) -> bytes:
    """The stored image, shrunk so its longest side is at most MAX_SIDE pixels."""
    with Image.open(image_path(image_id)) as img:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()


def _get_clip():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer

        _clip_model = SentenceTransformer(CLIP_MODEL)
    return _clip_model


def _image_embedding(image_id: str) -> np.ndarray:
    path = os.path.join(CLIP_CACHE_DIR, f"{image_id}.npy")
    if os.path.exists(path):
        return np.load(path)
    with Image.open(image_path(image_id)) as img:
        embedding = _get_clip().encode(img.convert("RGB"), normalize_embeddings=True)
    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
    np.save(path, embedding)
    return embedding


class ImageIndex:
    """CLIP search over a library's images: a text question finds images by what they show."""

    def __init__(self, records: list[dict]):
        import faiss

        self.records = records
        embeddings = np.vstack([_image_embedding(r["id"]) for r in records]).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question: str, top_k: int = 3) -> list[tuple[dict, float]]:
        query = _get_clip().encode([question], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query, min(top_k, len(self.records)))
        return [(self.records[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
