"""Retrieval & embedding utilities for verses.

Designed to be framework-agnostic so it can be reused both inside
Streamlit demos and future backend API services.
"""

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.scripts.db import get_supabase_client
import torch

load_dotenv()

HF_MODEL = os.getenv("EMBEDDINGS_MODEL")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
EMBED_MODE = os.getenv("EMBED_MODE_OFFLINE") == "TRUE"

EMBED_DIM = 768

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_embeddings_model():
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_KEY missing in environment.")
    
    if EMBED_MODE:
        from sentence_transformers import SentenceTransformer
    
        return SentenceTransformer(HF_MODEL,
                                device=device,
                                trust_remote_code=True,
                                cache_folder="/model/")
        
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model=HF_MODEL,
    )


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = get_embeddings_model()
    return embeddings.embed_documents(chunks)


def embed_query(query: str) -> List[float]:
    embeddings = get_embeddings_model()
    return embeddings.embed_query(query)


def semantic_search(
    query: str, 
    top_k: int = 5,
    embedding_table: str = "verse_embeddings",
    embedding_column: str = "embedding",
    chunk_table: str = "verse_chunks"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform semantic search using the updated Supabase RPC `semantic_search`.
    
    Supports dynamic table selection for scalability.
    """
    supabase = get_supabase_client()
    q_emb = embed_query(query)

    rpc_params = {
        "p_embedding_table": embedding_table,
        "p_embedding_column": embedding_column,
        "p_chunk_table": chunk_table,
        "p_query_vector": q_emb,
        "p_limit": top_k
    }

    response = supabase.rpc("semantic_search", rpc_params).execute()
    
    data = response.data or []
    results = []
    model_context = []
    

    # Map the new detailed SQL return columns to your Python dictionary
    for r in data:
        temp = {
            "chunk_key": r.get("chunk_key"),
            "surah_id": r.get("surah_id"),
            "verse_range": f"{r.get('start_verse')} - {r.get('end_verse')}",
            
            # Content Texts
            "text_english": r.get("text_english"),
            
            # Context Texts (Important for RAG)
            "context_english": r.get("context_text_english"),
            "similarity": r.get("similarity"),
        }
        model_context.append(temp)
        results.append({
            **temp,
            "chunk_id": r.get("chunk_id"),
            "text_uthmani": r.get("text_uthmani"),
            "context_uthmani": r.get("context_text_uthmani"),
            
        })

    return results, model_context
