import faiss
import numpy as np
class VectorStore:
    #wrapping a faiss index and a original text chunk so that both the faiss index and the text chunk are properly mapped
    def __init__(self,dimension:int):
        self.index=faiss.IndexFlatIP(dimension)
        self.chunks=[]
    def add(self,embeddings,chunks:list[str]):
        #add embeddings and the matching text chunks to the vector store
        embeddings=np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    def search(self,query_embedding,top_k:int=3):
        #Returns a single query embedding returning the top k chunks
        query_embedding=np.array([query_embedding]).astype("float32")
        similarities,indices=self.index.search(query_embedding,top_k)
        results=[]
        for idx,sim in zip(indices[0],similarities[0]):
            results.append({"chunk":self.chunks[idx],"similarity":float(sim)})
        return results
if __name__=="__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_chunks
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=chunk_text(text)
    embeddings=embed_chunks(chunks)
    store=VectorStore(dimension=embeddings[0].shape[0])
    store.add(embeddings,chunks)
    print(f"Total chunks stored:{len(store.chunks)}")
    
    query="Who threw the lemonade away?"
    query_embedding=embed_chunks([query])[0]
    results=store.search(query_embedding,top_k=3)
    for i,r in enumerate(results):
        print(f"\n--Result {i+1}(similarity : {r['similarity']:.4f})--")
        print(r["chunk"][:300])
