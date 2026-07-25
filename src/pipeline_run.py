from src.loader import load_pdf
from src.chunker import chunk_text
from src.semantic_chunker import semantic_chunk
from src.embeddings import embed_chunks
from src.vectorstore import VectorStore
from src.bm_25_search import BM25Search
from src.hybrid_search import HybridSearch
from src.reranker import rerank
from src.generator import generate_answer

PDF_PATH = "data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf"


def build_pipeline(use_semantic_chunking: bool):
    """
    Builds the retrieval components (chunks, vector store, BM25) for
    either the fixed-size or semantic chunking strategy.
    """
    text = load_pdf(PDF_PATH)

    if use_semantic_chunking:
        chunks = semantic_chunk(text)
    else:
        chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings[0].shape[0])
    store.add(embeddings, chunks)

    bm25 = BM25Search(chunks)
    hybrid = HybridSearch(store, bm25)

    return store, hybrid


def run_query(question: str, store, hybrid, use_reranking: bool):
    """
    Runs one question through retrieval (+ optional reranking) and generation.
    Returns the retrieved contexts and the generated answer.
    """
    query_embedding = embed_chunks([question])[0]

    if use_reranking:
        candidates = hybrid.search(question, query_embedding, top_k=10, candidate_k=25)
        reranked = rerank(question, candidates, top_k=3)
        retrieved_contexts = [r["chunk"] for r in reranked]
    else:
        results = store.search(query_embedding, top_k=3)
        retrieved_contexts = [r["chunk"] for r in results]

    generated_answer = generate_answer(question, retrieved_contexts)

    return retrieved_contexts, generated_answer