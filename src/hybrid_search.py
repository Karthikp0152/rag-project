from src.vectorstore import VectorStore
from src.bm_25_search import BM25Search


def reciprocal_rank_fusion(semantic_results, bm25_results, k=60, top_k=3):
    """
    Merges two ranked result lists (semantic + BM25) into one combined
    ranking, using Reciprocal Rank Fusion.
    """
    scores = {}          # index -> combined RRF score
    chunk_lookup = {}    # index -> chunk text

    for rank, r in enumerate(semantic_results, start=1):
        idx = r["index"]
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        chunk_lookup[idx] = r["chunk"]

    for rank, r in enumerate(bm25_results, start=1):
        idx = r["index"]
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        chunk_lookup[idx] = r["chunk"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        results.append({"chunk": chunk_lookup[idx], "rrf_score": score})
    return results


class HybridSearch:
    """
    Combines semantic (FAISS) search and BM25 keyword search using RRF.
    """

    def __init__(self, vector_store: VectorStore, bm25_search: BM25Search):
        self.vector_store = vector_store
        self.bm25_search = bm25_search

    def search(self, query: str, query_embedding, top_k: int = 3, candidate_k: int = 10):
        semantic_results = self.vector_store.search(query_embedding, top_k=candidate_k)
        bm25_results = self.bm25_search.search(query, top_k=candidate_k)

        return reciprocal_rank_fusion(semantic_results, bm25_results, top_k=top_k)


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, normalize_embeddings=True) if False else embed_chunks(chunks)

    store = VectorStore(dimension=embeddings[0].shape[0])
    store.add(embeddings, chunks)

    bm25 = BM25Search(chunks)

    hybrid = HybridSearch(store, bm25)

    query = "Prichard"
    query_embedding = embed_chunks([query])[0]
    results = hybrid.search(query, query_embedding, top_k=3)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (RRF score: {r['rrf_score']:.5f}) ---")
        print(r["chunk"][:300])