from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_model = None


def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, candidates: list[dict], top_k: int = 3, batch_size: int = 32) -> list[dict]:
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["chunk"]) for c in candidates]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scored = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks
    from src.vectorstore import VectorStore
    from src.bm_25_search import BM25Search
    from src.hybrid_search import HybridSearch

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    hybrid = HybridSearch(store, BM25Search(chunks))

    query = "Did Penn cheat on Daria?"
    candidates = hybrid.search(query, embed_chunks([query])[0], top_k=50, candidate_k=50)

    print("--- BEFORE reranking (RRF order) ---")
    for i, c in enumerate(candidates[:3], start=1):
        print(f"\n{i}. (RRF {c['rrf_score']:.5f})")
        print(c["chunk"][:200])

    print("\n\n--- AFTER reranking (cross-encoder order) ---")
    for i, r in enumerate(rerank(query, candidates, top_k=3), start=1):
        print(f"\n{i}. (rerank {r['rerank_score']:.4f}, was RRF {r['rrf_score']:.5f})")
        print(r["chunk"][:200])
