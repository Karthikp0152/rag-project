import re
import numpy as np
from src.embeddings import get_model
#Here we are looking to chunk based on sentence similarity rather than the traditional chunking techniques
def split_into_sentences(text:str)->list[str]:
    #split the texts into the chunks using the punctuation based rules using the regular expression
    sentences=re.split(r'(?<=[.!?])\s+',text)
    sentences=[s.strip() for s in sentences if s.strip()]
    return sentences
def semantic_chunk(text:str,similarity_threshold:float=0.5,max_chunk_size:int=1000)->list[str]:
    #group the sentences into chunks based on similaririty search between the sentences and if the similarity score is high then we can continue with the chunks and just if it drops too low then we do a new chunk
    sentences=split_into_sentences(text)
    if not sentences:
        return []
    model=get_model()
    embeddings=model.encode(sentences,normalize_embeddings=True)
    chunks=[]
    current_chunk_sentences=[sentences[0]]
    current_chunk_length=len(sentences[0])
    for i in range(1,len(sentences)):
        similarity=np.dot(embeddings[i-1],embeddings[i])
        sentence_len=len(sentences[i])
        if similarity<similarity_threshold or current_chunk_length+sentence_len>max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences=[sentences[i]]
            current_chunk_length=sentence_len
        else:
            current_chunk_sentences.append(sentences[i])
            current_chunk_length+=sentence_len
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks
if __name__=="__main__":
    from src.loader import load_pdf
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=semantic_chunk(text)
    print(f"Total semantic chunks:{len(chunks)}")
    print(f"\n--First Chunk--")
    print(chunks[0])
    print(f"\n--Second Chunk--")
    print(chunks[1])
    print(chunks[1] if len(chunks)>1 else "(only one chunk)")