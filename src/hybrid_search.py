from src.bm_25_search import BM25Search
from src.vectorstore import VectorStore


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k: int = 60,
    top_k: int = 3,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    scores: dict[int, float] = {}
    chunk_lookup: dict[int, str] = {}
    ranks: dict[int, dict[str, int]] = {}

    sources = (
        ("semantic", semantic_weight, semantic_results),
        ("bm25", bm25_weight, bm25_results),
    )
    for name, weight, results in sources:
        for rank, r in enumerate(results, start=1):
            idx = r["index"]
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank)
            chunk_lookup[idx] = r["chunk"]
            ranks.setdefault(idx, {})[name] = rank

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        {
            "chunk": chunk_lookup[idx],
            "index": idx,
            "rrf_score": score,
            "semantic_rank": ranks[idx].get("semantic"),
            "bm25_rank": ranks[idx].get("bm25"),
        }
        for idx, score in ranked[:top_k]
    ]


class HybridSearch:
    def __init__(self, vector_store: VectorStore, bm25_search: BM25Search):
        self.vector_store = vector_store
        self.bm25_search = bm25_search

    def search(
        self,
        query: str,
        query_embedding,
        top_k: int = 10,
        candidate_k: int = 50,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        semantic_results = self.vector_store.search(query_embedding, top_k=candidate_k)
        bm25_results = self.bm25_search.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            top_k=top_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    hybrid = HybridSearch(store, BM25Search(chunks))

    query = "Prichard"
    results = hybrid.search(query, embed_chunks([query])[0], top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"\n--- Result {i} (RRF {r['rrf_score']:.5f}, "
              f"dense rank {r['semantic_rank']}, bm25 rank {r['bm25_rank']}) ---")
        print(r["chunk"][:300])
