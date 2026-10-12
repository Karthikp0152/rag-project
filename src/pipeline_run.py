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


def build_pipeline(use_semantic_chunking: bool):
    key = "semantic" if use_semantic_chunking else "fixed"
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    from src.loader import load_pdf

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
):
    selected = retrieve(question, hybrid, use_reranking, top_k=top_k, candidate_k=candidate_k)
    retrieved_contexts = [r["chunk"] for r in selected]
    generated_answer = generate_answer(question, retrieved_contexts)
    return retrieved_contexts, generated_answer
