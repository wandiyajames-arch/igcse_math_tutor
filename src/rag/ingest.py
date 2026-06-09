import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
PDF_PATH = "data/past_papers/cambridge_2023.pdf" # Put your PDF here!
DB_DIR = "data/chroma_db"

def ingest_pdf():
    print(f"📄 Loading PDF from {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print("❌ Error: PDF not found. Please create the folder and add a PDF.")
        return

    # 1. Read the PDF
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Slice it into chunks
    # 500 characters per chunk, with 50 characters of overlap to keep context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Sliced PDF into {len(chunks)} chunks.")

    # 3. Convert to Vectors and Store in ChromaDB
    print("🧠 Generating embeddings and saving to database... (This takes a moment)")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_model, 
        persist_directory=DB_DIR
    )
    
    print("✅ Ingestion Complete! Your AI can now read this past paper.")

if __name__ == "__main__":
    ingest_pdf()