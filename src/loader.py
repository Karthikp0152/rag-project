from pypdf import PdfReader
def load_pdf(filepath:str)->str:
    #Reads a pdf file and returns the full pdf as a single big string
    reader=PdfReader(filepath)
    text_parts=[]
    for page in reader.pages:
        text_parts.append(page.extract_text())
    return "\n".join(text_parts)
if __name__=="__main__":
    text=load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    print(f"Extracted {len(text)} characters")
    print(text[:500])