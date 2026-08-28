import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "slc_solutions_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim matching ChromaDB

def get_embeddings():
    """
    Initializes HuggingFace sentence-transformers embedding model.
    Dimension: 384 (matches ChromaDB native dimensionality).
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (vector size: 384)...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_documents(data_path: Path):
    """
    Loads all PDF documents from the specified directory.
    """
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    documents = []
    for pdf_path in pdf_files:
        print(f"Loading document: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        documents.extend(docs)
    
    print(f"Loaded {len(documents)} page(s) across {len(pdf_files)} PDF file(s).")
    return documents

def split_documents_context_window(documents, chunk_size: int = 500, chunk_overlap: int = 150):
    """
    Context Window / Sliding Window Chunking:
    Uses RecursiveCharacterTextSplitter with an overlapping window to preserve
    surrounding sentence context across chunk boundaries.
    """
    print(f"Splitting documents using context-window chunking (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enrich metadata with chunk index and context window info
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    print(f"Generated {len(chunks)} contextual chunks.")
    return chunks

def ingest_data(data_path: Path = DATA_DIR, persist_directory: Path = CHROMA_DIR):
    """
    Full pipeline to load PDFs, chunk them with context window, and store in ChromaDB.
    """
    print("=" * 60)
    print("Starting Knowledge Base Ingestion Pipeline")
    print("=" * 60)
    
    # 1. Load Documents
    docs = load_documents(data_path)
    
    # 2. Context Window Chunking
    chunks = split_documents_context_window(docs, chunk_size=500, chunk_overlap=150)
    
    # 3. Embedding model
    embeddings = get_embeddings()
    
    # 4. Create and Persist Chroma Vector Store
    print(f"Storing vector embeddings in ChromaDB at: {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory)
    )
    
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'!")
    print("=" * 60)
    return vector_store

if __name__ == "__main__":
    ingest_data()
