import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Search:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    def search(self, query: str, top_k: int = 3):
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [
            {"chunk": self.chunks[idx], "score": float(score), "index": int(idx)}
            for idx, score in ranked[:top_k]
        ]


if __name__ == "__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    bm25_search = BM25Search(chunks)
    for i, r in enumerate(bm25_search.search("Prichard", top_k=3), start=1):
        print(f"\n---Result {i} (score: {r['score']:.4f})---")
        print(r["chunk"][:300])
