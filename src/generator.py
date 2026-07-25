import os
from dotenv import load_dotenv
from google import genai
load_dotenv() #it reads the .env file and loads api key in the environment
_client=None
def get_client()->genai.Client:
    #it loads the api client and caches it for reuse
    global _client
    if _client is None:
      api_key=os.environ["GEMINI_API_KEY"]
      _client=genai.Client(api_key=api_key)
    return _client
def build_prompt(question:str,chunks:list[str])->str:
    #it combines the query and the top k retrieved chunks and generates a prompt
    context="\n\n".join(chunks)
    prompt=f"""Answer the question using only the context below.If the context doesn't contain the answer,say you do not have enough information.
    Context:{context}
    Question:{question}
    """
    return prompt
def generate_answer(question:str,chunks:list[str])->str:
   #sends the question and the chunks to gemini and returns the answer.
   client=get_client()
   prompt=build_prompt(question,chunks)
   response=client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
   )
   return response.text
if __name__=="__main__":
   from src.loader import load_pdf
   from src.chunker import chunk_text
   from src.embeddings import embed_chunks
   from src.vectorstore import VectorStore
   text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
   chunks=chunk_text(text)
   embeddings=embed_chunks(chunks)
   store=VectorStore(dimension=embeddings[0].shape[0])
   store.add(embeddings,chunks)
   question="What happens between main characters?"
   query_embedding=embed_chunks([question])[0]
   results=store.search(query_embedding,top_k=3)
   retrieved_chunks=[r["chunk"]for r in results]
   answer=generate_answer(question,retrieved_chunks)
   print("Question:",question)
   print("\nAnswer:",answer)