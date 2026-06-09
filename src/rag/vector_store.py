import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.document_loader import load_and_chunk_pdfs

DB_DIR = "data/vector_db"

def build_vector_store():
    """Reads the chunked PDFs and builds a local ChromaDB database using free Hugging Face models."""
    print("📚 Loading and chunking PDFs...")
    chunks = load_and_chunk_pdfs("data/raw")
    
    if not chunks:
        print("❌ Cannot build database without chunks. Check your raw PDFs.")
        return None

    print("🧠 Downloading/Initializing Free Hugging Face Embeddings...")
    # 'all-MiniLM-L6-v2' is a lightweight, super-fast model perfect for local machines
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"🗄️ Saving {len(chunks)} chunks to ChromaDB at {DB_DIR}...")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=DB_DIR
    )
    
    print("✅ Local Vector Database built successfully for FREE!")
    return vector_store

def get_retriever():
    """Loads the existing database for the LLM to use."""
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_DIR):
        print("⚠️ Database not found! Building it now...")
        build_vector_store()
        
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    build_vector_store()