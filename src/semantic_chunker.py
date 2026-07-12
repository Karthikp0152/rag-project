import re
import numpy as np
from src.embeddings import get_model
#Here we are looking to chunk based on sentence similarity rather than the traditional chunking techniques
def split_into_sentences(text:str)->list[str]:
    #split the texts into the chunks using the punctuation based rules using the regular expression
    sentences=re.split(r'(?<=[.!?])\s+',text)
    sentences=[s.strip() for s in sentences if s.strip()]
    return sentences
def semantic_chunk(text: str, similarity_threshold: float = 0.3, max_chunk_size: int = 1000, min_chunk_size: int = 150) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    model = get_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_length = len(sentences[0])
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i - 1], embeddings[i])
        sentence_len = len(sentences[i])
        exceeds_max = current_chunk_length + sentence_len > max_chunk_size
        topic_shifted = similarity < similarity_threshold
        big_enough = current_chunk_length >= min_chunk_size
        if exceeds_max or (topic_shifted and big_enough):
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
            current_chunk_length = sentence_len
        else:
            current_chunk_sentences.append(sentences[i])
            current_chunk_length += sentence_len
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks
if __name__=="__main__":
    from src.loader import load_pdf
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=semantic_chunk(text)
    avg_length=sum(len(c) for c in chunks)/len(chunks)
    lengths=sorted(len(c) for c in chunks)
    print(f"Total semantic chunks:{len(chunks)}")
    print(f"Average chunk length:{avg_length:.1f} characters")
    print(f"shortest chunk:{lengths[0]} characters")
    print(f"Longest chunk:{lengths[-1]} characters")
    mid=len(chunks)//2
    print(f"\n---Sample chunk from the middle (index{mid})---")
    print(chunks[mid])
    print(f"\n---Next chunk after it (index {mid+1})---")
    print(chunks[mid+1])

   