import hashlib
import os

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CACHE_DIR = os.path.join(".cache", "embeddings")

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_max_seq_length() -> int:
    return get_model().max_seq_length


def embed_chunks(chunks: list[str], batch_size: int = 64, show_progress_bar: bool = False):
    return get_model().encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )


def embed_corpus(chunks: list[str], cache_key: str | None = None):
    if cache_key is None:
        return embed_chunks(chunks)

    digest = hashlib.sha256("\x00".join(chunks).encode("utf-8")).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_model = MODEL_NAME.replace("/", "_")
    path = os.path.join(CACHE_DIR, f"{cache_key}-{safe_model}-{digest}.npy")

    if os.path.exists(path):
        return np.load(path)

    embeddings = np.asarray(embed_chunks(chunks, show_progress_bar=True), dtype="float32")
    np.save(path, embeddings)
    return embeddings


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks[:3])
    print(f"Model              : {MODEL_NAME}")
    print(f"Max sequence length: {get_max_seq_length()} tokens")
    print(f"Chunks embedded    : {len(embeddings)}")
    print(f"Embedding shape    : {embeddings[0].shape}")
