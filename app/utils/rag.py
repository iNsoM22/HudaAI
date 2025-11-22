import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from scripts.db import supabase
from uuid import uuid4
from dotenv import load_dotenv

def get_embeddings():
    """Initialize Hugging Face Embeddings Model (Free & Fast)."""
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        model="sentence-transformers/all-MiniLM-L6-v2"
    )    
    return embeddings


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings():
    """Initialize Hugging Face Embeddings Model (Free & Fast)."""
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        model="sentence-transformers/all-MiniLM-L6-v2"
    )    
    return embeddings


def store_vector(user_id: str, resume_id: str, text_chunks: list[str], embeddings):
    vectors = embeddings.embed_documents(text_chunks)
    rows = [
        {
            "user_id": user_id,
            "resume_id": resume_id,
            "content": chunk,
            "embedding": vector,
            "metadata": {
                "user_id": user_id,
            }
        }
        for chunk, vector in zip(text_chunks, vectors)
    ]
    
    supabase.table("documents").insert(rows).execute()


def retrieve_vectors(user_id: str, query_vector, top_k: int = 5):
    try:
        response = supabase.rpc(
            "match_resume_chunks",
            {
                "query_embedding": query_vector,
                "match_count": top_k,
                "user_id": user_id
            }
        ).execute()

        if not response.data:
            return []

        results = []
        for r in response.data:
            results.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "metadata": r.get("metadata"),
                "similarity": r.get("similarity"),
            })
            
        return results

    except Exception as e:
        print(f"[retrieve_vectors] Error: {e}")
        return []