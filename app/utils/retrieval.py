"""Retrieval & embedding utilities for verses.

Designed to be framework-agnostic so it can be reused both inside
Streamlit demos and future backend API services.
"""

from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.scripts.db import get_supabase_client
from app.config.secrets import HF_MODEL, HF_TOKEN, EMBED_MODE
import torch


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


def semantic_search_quran(
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

    response = supabase.rpc("semantic_search_quran", rpc_params).execute()
    
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


def semantic_search_hadiths(
    query: str, 
    top_k: int = 5,
    book_filter: int = None,
    match_threshold: float = 0.5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform semantic search on Hadiths using Supabase RPC.
    
    Returns hadith chunks matching the query with full context.
    
    Args:
        query: Natural language search query
        top_k: Maximum number of results to return
        book_filter: Optional book_id to filter results
        match_threshold: Minimum similarity threshold (0-1)
        
    Returns:
        Tuple of (full_results, model_context) where:
        - full_results: Complete data including Arabic text
        - model_context: Simplified context for LLM consumption
    """
    supabase = get_supabase_client()
    q_emb = embed_query(query)

    rpc_params = {
        "query_embedding": q_emb,
        "match_count": top_k,
        "match_threshold": match_threshold,
        "filter_book_id": book_filter
    }

    response = supabase.rpc("semantic_search_hadith", rpc_params).execute()
    
    data = response.data or []
    results = []
    model_context = []
    
    # Map SQL columns to Python dictionary following the schema
    for r in data:
        temp = {
            "hadith_id": r.get("hadith_id"),
            "book_name": r.get("book_name"),
            "hadith_number": r.get("hadith_number"),
            "similarity": r.get("similarity"),
            "status": r.get("status"),
            "chunk_text": r.get("chunk_text"),
            "context_english": r.get("context_text_english"),
        }
        model_context.append(temp)
        
        results.append({
            **temp,
            "context_arabic": r.get("context_text_arabic"),
        })

    return results, model_context


def get_records(
    table: str,
    filters: Dict[str, Any] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = None,
    ascending: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve records from a specified table with optional filtering and ordering.
    
    Args:
        table: Table name ('verse_chunks', 'hadiths', etc.)
        filters: Dictionary of column:value pairs for filtering
        limit: Maximum number of records to return
        offset: Number of records to skip
        order_by: Column name to sort by
        ascending: Sort direction (True for ascending, False for descending)
        
    Returns:
        List of records as dictionaries
    """
    supabase = get_supabase_client()
    
    try:
        query = supabase.table(table).select("*")
        
        # Apply filters
        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)
        
        # Apply ordering
        if order_by:
            query = query.order(order_by, desc=not ascending)
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        response = query.execute()
        return response.data or []
        
    except Exception as e:
        print(f"Error fetching records from {table}: {str(e)}")
        return []


def get_expanded_context(
    source_type: str,
    identifier: str,
    context_window: int = 3
) -> Dict[str, Any]:
    """
    Get expanded context around a specific verse or hadith.
    
    Args:
        source_type: 'quran' or 'hadith'
        identifier: chunk_key for Quran, hadith_id for Hadith
        context_window: Number of items before/after to retrieve
        
    Returns:
        Dictionary with expanded contexts and metadata
    """
    supabase = get_supabase_client()
    
    try:
        if source_type == "quran":
            # Get the original chunk to find its position
            chunk_response = supabase.table("verse_chunks").select(
                "chunk_id, surah_id, start_verse, end_verse"
            ).eq("chunk_key", identifier).execute()
            
            if not chunk_response.data:
                return {"status": "error", "message": "Chunk not found", "contexts": []}
            
            chunk = chunk_response.data[0]
            surah_id = chunk["surah_id"]
            start_verse = chunk["start_verse"]
            end_verse = chunk["end_verse"]
            
            # Calculate expanded verse range
            expanded_start = max(1, start_verse - context_window)
            expanded_end = end_verse + context_window
            
            # Fetch surrounding chunks from same surah
            expanded_response = supabase.table("verse_chunks").select(
                "chunk_key, chunk_id, surah_id, start_verse, end_verse, text_english, text_uthmani, context_text_english, context_text_uthmani"
            ).eq("surah_id", surah_id).gte("start_verse", expanded_start).lte(
                "end_verse", expanded_end
            ).order("start_verse", desc=False).execute()
            
            return {
                "status": "success",
                "source": "quran",
                "original_identifier": identifier,
                "expanded_range": f"{expanded_start}-{expanded_end}",
                "surah_id": surah_id,
                "contexts": expanded_response.data or [],
                "count": len(expanded_response.data or [])
            }
            
        elif source_type == "hadith":
            # Get the original hadith
            hadith_response = supabase.table("hadiths").select(
                "hadith_id, book_id, hadith_number, status, hadith_text_english, hadith_text_arabic"
            ).eq("hadith_id", identifier).execute()
            
            if not hadith_response.data:
                return {"status": "error", "message": "Hadith not found", "contexts": []}
            
            hadith = hadith_response.data[0]
            book_id = hadith["book_id"]
            hadith_number = hadith["hadith_number"]
            
            # Calculate expanded hadith range
            expanded_start = max(1, hadith_number - context_window)
            expanded_end = hadith_number + context_window
            
            # Fetch surrounding hadiths from same book
            expanded_response = supabase.table("hadiths").select(
                "hadith_id, book_id, hadith_number, status, hadith_text_english, hadith_text_arabic"
            ).eq("book_id", book_id).gte("hadith_number", expanded_start).lte(
                "hadith_number", expanded_end
            ).order("hadith_number", desc=False).execute()
            
            # Get book name
            book_response = supabase.table("hadith_books").select(
                "book_name"
            ).eq("book_id", book_id).execute()
            
            book_name = book_response.data[0]["book_name"] if book_response.data else "Unknown"
            
            return {
                "status": "success",
                "source": "hadith",
                "original_identifier": identifier,
                "book_name": book_name,
                "expanded_range": f"{expanded_start}-{expanded_end}",
                "book_id": book_id,
                "contexts": expanded_response.data or [],
                "count": len(expanded_response.data or [])
            }
        else:
            return {"status": "error", "message": "Invalid source_type. Use 'quran' or 'hadith'", "contexts": []}
            
    except Exception as e:
        return {"status": "error", "message": str(e), "contexts": []}
