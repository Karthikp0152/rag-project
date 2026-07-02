def chunk_text(text:str,chunk_size:int=500,chunk_overlap:int=50)->list[str]:
    #splits text into overlapping chunks of roughly chunk_size characters.
    chunks=[]
    start=0
    text_length=len(text)
    while start < text_length:
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-chunk_overlap
    return chunks
if __name__=="__main__":
    from src.loader import load_pdf
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks=chunk_text(text)
    print(f"Total characters:{len(text)}")
    print(f"No.of chunks:{len(chunks)}")
    print("\n---First Chunk---")
    print(chunks[0])
    print("\n---Second Chunk---")
    print(chunks[1] if len(chunks)>1 else "(Only one chunk)")