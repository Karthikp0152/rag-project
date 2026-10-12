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
        question, retrieved_contexts, section_headings=section_headings
    )
    return retrieved_contexts, generated_answer
