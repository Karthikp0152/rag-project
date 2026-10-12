import re

import numpy as np

from src.embeddings import get_model

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?”"])\s+')


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.3,
    max_chunk_size: int = 900,
    min_chunk_size: int = 400,
    overlap_sentences: int = 1,
) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    embeddings = get_model().encode(
        sentences,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    chunks: list[str] = []
    current = [sentences[0]]
    centroid = np.array(embeddings[0], dtype="float32")
    length = len(sentences[0])

    for i in range(1, len(sentences)):
        norm = float(np.linalg.norm(centroid))
        similarity = float(np.dot(centroid / norm, embeddings[i])) if norm else 0.0

        sentence_length = len(sentences[i]) + 1
        exceeds_max = length + sentence_length > max_chunk_size
        topic_shifted = similarity < similarity_threshold and length >= min_chunk_size

        if exceeds_max or topic_shifted:
            chunks.append(" ".join(current))
            carry = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = carry + [sentences[i]]
            length = sum(len(s) + 1 for s in current) - 1
            start = i - len(carry)
            centroid = np.array(embeddings[start : i + 1], dtype="float32").sum(axis=0)
        else:
            current.append(sentences[i])
            length += sentence_length
            centroid = centroid + embeddings[i]

    if current:
        chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    from src.loader import load_pdf

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = semantic_chunk(text)
    lengths = sorted(len(c) for c in chunks)
    print(f"Total semantic chunks : {len(chunks)}")
    print(f"Average chunk length  : {sum(lengths) / len(lengths):.1f} characters")
    print(f"Shortest / Longest    : {lengths[0]} / {lengths[-1]} characters")
    over = sum(1 for length in lengths if length > 1000)
    print(f"Chunks over 1000 chars: {over}")
    mid = len(chunks) // 2
    print(f"\n---Sample chunk (index {mid})---")
    print(chunks[mid])
