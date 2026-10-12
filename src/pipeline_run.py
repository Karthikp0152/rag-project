import hashlib
import json
import os
import re

import numpy as np

from src.bm_25_search import BM25Search
from src.chunker import chunk_text
from src.embeddings import embed_chunks, embed_corpus
from src.generator import generate_answer
from src.hybrid_search import HybridSearch
from src.reranker import rerank
from src.semantic_chunker import semantic_chunk
from src.vectorstore import VectorStore

PDF_PATH = "data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf"
TOP_K = 20
CANDIDATE_K = 100

LIBRARY_CACHE_DIR = os.path.join(".cache", "library")
# Bump when loading or chunking logic changes, so cached chunks are rebuilt.
LIBRARY_CACHE_VERSION = 2

_pipeline_cache: dict[str, tuple] = {}


def chunk_sections(sections: list[dict], use_semantic_chunking: bool) -> list[str]:
    # Chunk each section on its own so no chunk spans two sections, and prefix every
    # chunk with its heading so passages far from the heading keep their context
    # (e.g. which character "I" is in a first-person chapter).
    chunker = semantic_chunk if use_semantic_chunking else chunk_text
    chunks = []
    for section in sections:
        prefix = f"[{section['heading']}] " if section["heading"] else ""
        chunks.extend(prefix + chunk for chunk in chunker(section["text"]))
    return chunks


def build_pipeline(use_semantic_chunking: bool, read_images: bool = False):
    key = ("semantic" if use_semantic_chunking else "fixed") + ("-img" if read_images else "")
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    from src.loader import load_pdf, load_pdf_sections

    if read_images:
        chunks = chunk_sections(load_pdf_sections(PDF_PATH), use_semantic_chunking)
    else:
        text = load_pdf(PDF_PATH)
        chunks = semantic_chunk(text) if use_semantic_chunking else chunk_text(text)
    embeddings = embed_corpus(chunks, cache_key=key)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    hybrid = HybridSearch(store, BM25Search(chunks))

    _pipeline_cache[key] = (store, hybrid)
    return store, hybrid


def _file_chunks(path: str, use_semantic_chunking: bool, read_images: bool) -> dict:
    """Return {"chunks": [[heading, chunk_text], ...], "images": [image records]} for one file.

    Cached on disk by file content and settings, so a file is read and chunked once;
    after that, loading it takes seconds instead of re-running the chunker.
    """
    from src.documents import load_document

    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    mode = ("semantic" if use_semantic_chunking else "fixed") + ("-img" if read_images else "")
    cache_path = os.path.join(LIBRARY_CACHE_DIR, f"{digest}-{mode}-v{LIBRARY_CACHE_VERSION}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    chunker = semantic_chunk if use_semantic_chunking else chunk_text
    images: list[dict] = []
    sections = load_document(path, read_images=read_images, images=images)
    result = {
        "chunks": [
            [section["heading"], chunk] for section in sections for chunk in chunker(section["text"])
        ],
        "images": images,
    }
    os.makedirs(LIBRARY_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result


def build_library(paths: list[str], use_semantic_chunking: bool = True, read_images: bool = True):
    """Index several files together. Every chunk is tagged [file · heading].

    Returns (store, hybrid, images), where images lists the content images found, each
    {"id", "source", "page", "heading", "caption"}.
    """
    mode = "semantic" if use_semantic_chunking else "fixed"
    all_chunks: list[str] = []
    all_embeddings = []
    all_images: list[dict] = []
    for path in paths:
        source = os.path.basename(path)
        loaded = _file_chunks(path, use_semantic_chunking, read_images)
        all_images.extend({**image, "source": source} for image in loaded["images"])
        chunks = []
        for heading, chunk in loaded["chunks"]:
            label = " · ".join(part for part in (source, heading) if part)
            chunks.append(f"[{label}] {chunk}")
        if not chunks:
            print(f"warning: no text found in {path}")
            continue
        # Embeddings are cached by chunk content, so unchanged files are not re-embedded.
        all_embeddings.append(embed_corpus(chunks, cache_key=f"lib-{mode}"))
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError("None of the given files contained any text.")

    embeddings = np.vstack(all_embeddings)
    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, all_chunks)
    return store, HybridSearch(store, BM25Search(all_chunks)), all_images


# Questions that explicitly ask about a visual ("what is in the image?") are often too vague
# for CLIP to score highly, but clearly want the picture itself.
_ASKS_FOR_VISUAL = re.compile(
    r"\b(?:images?|pictures?|photos?|photographs?|diagrams?|charts?|graphs?|plots?|maps?"
    r"|floor ?plans?|illustrations?|screenshots?|drawings?|figures?(?!\s+out))\b",
    re.IGNORECASE,
)


def asks_for_visual(question: str) -> bool:
    return bool(_ASKS_FOR_VISUAL.search(question))


def select_images(
    question: str,
    passages: list[str],
    images: list[dict],
    image_index,
    max_images: int = 3,
    min_clip_score: float = 0.25,
) -> list[dict]:
    """Pick images to show the answer model, at most max_images.

    Normally an image is sent only if CLIP rates it a strong visual match for the
    question: its caption is already in the passages, so the picture itself only pays
    for its extra tokens when it really fits. If the question explicitly asks about an
    image, chart, diagram, etc., the best matches are sent regardless of score.
    Images whose captions were retrieved come first.
    """
    from src.image_index import linked_image_ids

    explicit = asks_for_visual(question)
    matches = [
        image for image, score in image_index.search(question, top_k=len(images))
        if explicit or score >= min_clip_score
    ]
    linked = linked_image_ids(passages)
    matches.sort(key=lambda image: linked.index(image["id"]) if image["id"] in linked else len(linked))
    return matches[:max_images]


def retrieve(
    question: str,
    hybrid: HybridSearch,
    use_reranking: bool,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
):
    query_embedding = embed_chunks([question])[0]
    if use_reranking:
        candidates = hybrid.search(
            question, query_embedding, top_k=candidate_k, candidate_k=candidate_k
        )
        return rerank(question, candidates, top_k=top_k)
    return hybrid.search(question, query_embedding, top_k=top_k, candidate_k=candidate_k)


def run_query(
    question: str,
    store,
    hybrid: HybridSearch,
    use_reranking: bool,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    section_headings: bool = False,
):
    selected = retrieve(question, hybrid, use_reranking, top_k=top_k, candidate_k=candidate_k)
    retrieved_contexts = [r["chunk"] for r in selected]
    generated_answer = generate_answer(
        question, retrieved_contexts, section_headings=section_headings, subject="a novel"
    )
    return retrieved_contexts, generated_answer
