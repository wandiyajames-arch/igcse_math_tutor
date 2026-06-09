import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(raw_data_dir="data/raw", chunk_size=1000, chunk_overlap=200):
    pdf_files = glob.glob(os.path.join(raw_data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in {raw_data_dir}.")
        return []

    print(f"📄 Found {len(pdf_files)} PDFs. Starting extraction...")
    
    all_documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        # SAFETY CHECK 1: Skip 0-byte files
        if os.path.getsize(pdf_path) == 0:
            print(f"⏭️ Skipping empty file: {filename}")
            continue
            
        try:
            print(f"   -> Extracting: {filename}")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            # SAFETY CHECK 2: Catch any other PDF errors and keep going
            print(f"❌ Error reading {filename}: {e}. Skipping this file.")
            continue
        
    if not all_documents:
        print("⚠️ No pages were successfully extracted.")
        return []
        
    print(f"✅ Extracted {len(all_documents)} total pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    print("✂️ Chunking documents...")
    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ Created {len(chunks)} searchable chunks.")
    
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()