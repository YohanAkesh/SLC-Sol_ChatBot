import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Base paths & load env
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(SRC_DIR / ".env")

DATA_DIR = BASE_DIR / "Data"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "slcindex1")
EMBEDDING_MODEL_NAME = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_embeddings():
    """
    Initializes Pinecone Hosted Embeddings (llama-text-embed-v2, dimension=1024).
    Zero local disk usage — 100% serverless and Vercel compatible.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment or .env file.")
        
    return PineconeEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        dimension=EMBEDDING_DIMENSION,
    )

def get_vectorstore(index_name: str = PINECONE_INDEX_NAME):
    """
    Connects to the cloud-hosted Pinecone vector index.
    """
    embeddings = get_embeddings()
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment or .env file.")
        
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
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
