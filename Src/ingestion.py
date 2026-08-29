import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

# Base paths & load env
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(SRC_DIR / ".env")

DATA_DIR = BASE_DIR / "Data"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "slcindex1")
EMBEDDING_MODEL_NAME = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_embeddings():
    """
    Initializes Pinecone Hosted Embeddings (llama-text-embed-v2, dimension=1024).
    Zero local disk usage.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment or .env file.")
        
    print(f"Initializing Pinecone Hosted Embeddings ({EMBEDDING_MODEL_NAME}, dim={EMBEDDING_DIMENSION})...")
    return PineconeEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        dimension=EMBEDDING_DIMENSION,
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

def ingest_data(data_path: Path = DATA_DIR, index_name: str = PINECONE_INDEX_NAME):
    """
    Full pipeline to load PDFs, chunk them with context window, and upsert embeddings into Pinecone index.
    """
    print("=" * 60)
    print(f"Starting Knowledge Base Ingestion to Pinecone Index: '{index_name}'")
    print("=" * 60)
    
    # 1. Load Documents
    docs = load_documents(data_path)
    
    # 2. Context Window Chunking
    chunks = split_documents_context_window(docs, chunk_size=500, chunk_overlap=150)
    
    # 3. Embedding model
    embeddings = get_embeddings()
    
    # 4. Create and Upsert into Pinecone Vector Store
    print(f"Upserting vector embeddings to Pinecone index '{index_name}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    print(f"Successfully ingested {len(chunks)} chunks into Pinecone index '{index_name}'!")
    print("=" * 60)
    return vector_store

if __name__ == "__main__":
    ingest_data()
