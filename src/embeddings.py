from sentence_transformers import SentenceTransformer
_model = None

def get_model()-> SentenceTransformer:
    #loads and caches the embedding model
    global _model
    if _model is None:
        _model=SentenceTransformer("all-MiniLM-L6-v2")
    return _model
def embed_chunks(chunks:list[str]):
    #converts the chunks into embedding vectors
    model=get_model()
    embeddings=model.encode(chunks)
    return embeddings
if __name__=="__main__":
    from src.loader import load_pdf
    from src.chunker import chunk_text
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=chunk_text(text)
    embeddings=embed_chunks(chunks[:3])
    print(f"Number of chunks embedded:{len(embeddings)}")
    print(f"Shape of embedding:{embeddings[0].shape}")
    print(f"The first 10 numbers of the 1st embedding:{embeddings[0][:10]}")