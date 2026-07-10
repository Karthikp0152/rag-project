from rank_bm25 import BM25Okapi
class BM25Search:
    #it wraps a BM25 keyword search index over a list of text chunks.
    def __init__(self,chunks:list[str]):
        self.chunks=chunks
        tokenized_chunks=[chunk.lower().split() for chunk in chunks]
        self.bm25=BM25Okapi(tokenized_chunks)
    def search(self,query:str,top_k:int=3):
        #returns top_k chunks with the highest Bm25 matching score
        tokenized_query=query.lower().split()
        scores=self.bm25.get_scores(tokenized_query)
        #pair each chunk index with its score,then sort by scire in the desceding order
        ranked =sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
        top_results=ranked[:top_k]
        results=[]
        for idx,score in top_results:
            results.append({"chunk":self.chunks[idx],"score":float(score)})
        return results
if __name__=="__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=chunk_text(text)
    bm25_search=BM25Search(chunks)
    query='Prichard'
    results=bm25_search.search(query,top_k=3)
    for i,r in enumerate(results):
        print(f"\n---Result{i+1}(score:{r['score']:.4f})---")
        print(r['chunk'][:300])