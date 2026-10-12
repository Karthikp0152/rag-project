import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[str] = []

    def add(self, embeddings, chunks: list[str]):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks must have matching lengths")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k: int = 3):
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        query_embedding = np.asarray([query_embedding], dtype="float32")
        similarities, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "similarity": float(sim),
                "index": int(idx),
            })
        return results


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store = VectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    print(f"Total chunks stored: {len(store.chunks)}")

    query = "Who threw the lemonade away?"
    results = store.search(embed_chunks([query])[0], top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"\n--Result {i} (similarity: {r['similarity']:.4f})--")
        print(r["chunk"][:300])
