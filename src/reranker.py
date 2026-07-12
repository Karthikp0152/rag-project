from sentence_transformers import CrossEncoder

_reranker_model = None

def get_reranker() -> CrossEncoder:
    """
    Loads (and caches) the cross-encoder reranking model.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Re-scores candidate chunks using a cross-encoder, which reads the
    query and each chunk together for a more precise relevance judgment.
    """
    model = get_reranker()

    pairs = [(query, c["chunk"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks
    from src.vectorstore import VectorStore
    from src.bm_25_search import BM25Search
    from src.hybrid_search import HybridSearch, reciprocal_rank_fusion

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings[0].shape[0])
    store.add(embeddings, chunks)

    bm25 = BM25Search(chunks)
    hybrid = HybridSearch(store, bm25)

    query = "Did Penn cheat on Daria?"
    query_embedding = embed_chunks([query])[0]

    # get a wider shortlist from hybrid search before reranking
    candidates = hybrid.search(query, query_embedding, top_k=10, candidate_k=25)

    print("--- BEFORE reranking (RRF order) ---")
    for i, c in enumerate(candidates[:3]):
        print(f"\n{i+1}. (RRF score: {c['rrf_score']:.5f})")
        print(c["chunk"][:200])

    reranked = rerank(query, candidates, top_k=3)

    print("\n\n--- AFTER reranking (cross-encoder order) ---")
    for i, r in enumerate(reranked):
        print(f"\n{i+1}. (rerank score: {r['rerank_score']:.4f}, was RRF score: {r['rrf_score']:.5f})")
        print(r["chunk"][:200])