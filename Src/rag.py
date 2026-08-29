import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Base paths & load env
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(SRC_DIR / ".env")

DATA_DIR = BASE_DIR / "Data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "slc_solutions_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

def get_embeddings():
    """
    Initializes Hugging Face Serverless Endpoint Embeddings.
    Dimension: 384 (zero local model download, no PyTorch overhead).
    """
    if not HF_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing in environment or .env file.")
        
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        huggingfacehub_api_token=HF_TOKEN,
    )

def get_vectorstore(persist_directory: Path = CHROMA_DIR):
    """
    Connects to the persisted ChromaDB vector store.
    If not initialized yet, runs ingestion automatically.
    """
    embeddings = get_embeddings()
    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        from Src.ingestion import ingest_data
        print(f"[RAG] ChromaDB not found at {persist_directory}. Triggering automatic ingestion...")
        return ingest_data(DATA_DIR, persist_directory)
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory)
    )

def get_llm(model_name: str = DEFAULT_GROQ_MODEL, temperature: float = 0.2):
    """Initializes ChatGroq LLM."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables or .env file.")
    
    return ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=1024
    )

def format_docs(docs):
    """Formats retrieved documents into a clean context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "KB")
        page = doc.metadata.get("page", 1)
        # Adjust 0-indexed page if needed
        page_num = page + 1 if isinstance(page, int) and page == 0 else (page or 1)
        formatted.append(f"[Source {i}: {source} (Page {page_num})]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)

SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for S.L.C Solutions (Sri Lankan Construction Solutions) — an open B2B digital marketplace for Sri Lanka's construction and hardware industry.

Your goal is to assist buyers, contractors, suppliers, and hardware shop owners accurately and professionally.

Rules & Guidelines:
1. Answer the user's question directly using the retrieved context provided below.
2. If the answer is found in the context, be clear, structured, and informative.
3. If the information is NOT mentioned in the context, politely state that you don't have that specific information in the S.L.C Solutions knowledge base and provide the official contact support details:
   - Live Chat Support: 24/7 on the portal
   - Official Email: support@slcsolutions.lk / inquiries@slcsolutions.lk
   - WhatsApp / Telegram Bots: Instant alert channels
   - Dispute Resolution Hours: Mon - Sat: 8:00 AM - 6:00 PM (IST)
4. Maintain a helpful, courteous, and professional tone.
5. If relevant, explain terms like MOQ (Order Pooling), Contract Proposal Wizard, boostScore, or deal statuses.

Retrieved Knowledge Base Context:
---------------------
{context}
---------------------
"""

class RAGChatbot:
    def __init__(self, top_k: int = 4, model_name: str = DEFAULT_GROQ_MODEL):
        self.vectorstore = get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        self.llm = get_llm(model_name=model_name)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        self.output_parser = StrOutputParser()
        
    def query(self, question: str):
        """Executes RAG search and generation for a user query."""
        docs = self.retriever.invoke(question)
        context_text = format_docs(docs)
        
        messages = self.prompt_template.format_messages(
            context=context_text,
            question=question
        )
        
        response = self.llm.invoke(messages)
        
        sources = []
        for doc in docs:
            page_val = doc.metadata.get("page", 1)
            page_num = (page_val + 1) if isinstance(page_val, int) and page_val == 0 else (page_val or 1)
            content = doc.page_content
            snippet = (content[:200] + "...") if len(content) > 200 else content
            
            sources.append({
                "file": doc.metadata.get("source_file", "SLC_Solutions_KB.pdf"),
                "page": int(page_num) if str(page_num).isdigit() else 1,
                "content_snippet": snippet
            })
        
        return {
            "answer": response.content,
            "sources": sources
        }

if __name__ == "__main__":
    bot = RAGChatbot()
    test_query = "What is S.L.C Solutions and how can I order materials?"
    print(f"\nUser Question: {test_query}\n")
    result = bot.query(test_query)
    print("Response:\n" + result["answer"])
    print("\nSources Retrieved:")
    for s in result["sources"]:
        print(f"- {s['file']} (Page {s['page']})")
