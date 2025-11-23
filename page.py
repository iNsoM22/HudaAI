import streamlit as st
import sys
import os
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    

from app.services.agent import answer_query
from app.utils.retrieval import semantic_search

st.set_page_config(page_title="HudaAI Verse Explorer", layout="wide")
st.title("🕌 HudaAI Verse Explorer (Agentic RAG Demo)")

st.markdown(
    """
Enter a question about Quranic content. The system will perform a semantic search over embedded verse **chunks** (with Arabic & English plus context), then generate an answer grounded only in retrieved material.

If insufficient context is found, the agent will indicate uncertainty. This UI is framework-agnostic and can be migrated to an API service.
"""
)


def normalize_contexts(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map new retrieval schema to keys the agent prompt builder may expect.

    The current agent implementation (see `agent.py`) was built for older keys
    like `verse_id` and `content`. We construct those while preserving richer
    metadata for display.
    """
    normalized = []
    for r in raw:
        verse_id = r.get("chunk_key") or r.get("chunk_id") or r.get("surah_id")
        # Prefer English verse text; fall back to Arabic if missing.
        content_text = r.get("text_english") or r.get("text_uthmani") or "[No verse text]"
        normalized.append({
            "verse_id": verse_id,
            "content": content_text,
            **r,  # keep original rich fields
        })
    return normalized


def display_contexts(contexts: List[Dict[str, Any]]) -> None:
    if not contexts:
        st.info("No matching verse chunks were retrieved.")
        return
    for c in contexts:
        verse_range = c.get("verse_range") or "N/A"
        surah = c.get("surah_id") or "?"
        sim = c.get("similarity")
        header = f"Surah {surah} | Verses {verse_range}"
        if sim is not None:
            header += f" | Similarity: {sim:.4f}"
        st.markdown(f"**{header}**")
        if c.get("text_english"):
            st.write(c["text_english"].strip())
        if c.get("text_uthmani"):
            with st.expander("Arabic Text"):
                st.write(c["text_uthmani"].strip())
        # Optional extended context
        if c.get("context_english") or c.get("context_uthmani"):
            with st.expander("Extended Context"):
                if c.get("context_english"):
                    st.write(c["context_english"].strip())
                if c.get("context_uthmani"):
                    st.write(c["context_uthmani"].strip())
        st.markdown("---")


query = st.text_input(
    "Your question",
    placeholder="e.g. What verses speak about patience?",
    help="Will perform semantic retrieval then generate an answer citing verses.",
)
top_k = st.slider("Number of verse chunks", min_value=1, max_value=10, value=5)
generate = st.button("Generate Answer")

if generate:
    cleaned = (query or "").strip()
    if not cleaned:
        st.warning("Please enter a question before generating an answer.")
    elif len(cleaned) < 3:
        st.warning("Query too short. Add a few more words for better retrieval.")
    else:
        try:
            with st.spinner("Retrieving and generating answer..."):
                # raw_contexts, _ = semantic_search(cleaned, top_k=top_k)
                # contexts = normalize_contexts(raw_contexts)
                result = answer_query(cleaned, top_k=top_k)
                # Replace contexts in result with normalized for consistency in display
                # result["contexts"] = contexts
        except Exception as e:
            st.error(f"Error during generation: {e}")
        else:
            st.subheader("Answer")
            st.write(result.get("answer", "[No answer returned]").strip())
            st.subheader("Retrieved Contexts")
            display_contexts(result.get("contexts", []))

st.sidebar.header("Quick Semantic Search")
search_q = st.sidebar.text_input(
    "Search only (no generation)",
    placeholder="mercy",
    help="Returns top matching verse chunks without running the LLM.",
)
if st.sidebar.button("Search"):
    cleaned = (search_q or "").strip()
    if not cleaned:
        st.sidebar.warning("Enter a search term.")
    else:
        try:
            hits = semantic_search(cleaned, top_k=5)
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")
        else:
            st.sidebar.write(f"Found {len(hits)} chunks:")
            for h in hits:
                verse_range = h.get("verse_range") or "?"
                surah = h.get("surah_id") or "?"
                sim = h.get("similarity")
                caption = f"Surah {surah} | {verse_range}"
                if sim is not None:
                    caption += f" | sim={sim:.3f}"
                st.sidebar.caption(caption)
                preview = h.get("text_english") or h.get("text_uthmani") or "[No text]"
                preview = preview.strip()
                st.sidebar.write(preview[:180] + ("..." if len(preview) > 180 else ""))
