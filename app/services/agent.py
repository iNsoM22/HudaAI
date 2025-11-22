"""Agentic RAG service for answering user queries.

This keeps orchestration logic separate so it can later be exposed via
an API framework (FastAPI, Django, etc.)
"""

from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from app.utils.retrieval import semantic_search

load_dotenv()

GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _get_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it to use the agent.")
    return ChatGoogleGenerativeAI(model=GOOGLE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Construct a richer, structured prompt to reduce false insufficiency replies.

    Each context chunk may contain:
    - surah_id
    - verse_range (start - end)
    - text_english (primary verse text)
    - context_english (extended surrounding verses)
    - similarity (retrieval score)

    We normalize an identifier and explicitly instruct the model to answer
    using ONLY provided material, but to synthesize across chunks before
    deciding insufficiency.
    """
    if not contexts:
        return (
            "You are an Islamic assistant. No verses were retrieved for the query. "
            "Politely state that there is insufficient context.\n\n"
            f"Query: {query}\n\nAnswer:"
        )

    formatted_chunks = []
    for idx, c in enumerate(contexts, start=1):
        surah = c.get("surah_id", "?")
        verse_range = c.get("verse_range", "?")
        sim = c.get("similarity")
        identifier = c.get("chunk_key") or c.get("chunk_id") or c.get("verse_id") or f"chunk_{idx}"
        header = f"[Chunk {idx} | ID {identifier} | Surah {surah} | Verses {verse_range}" + (f" | Sim {sim:.4f}]" if isinstance(sim, (int, float)) else "]")
        main_text = (c.get("text_english") or c.get("text_uthmani") or "[No main text]").strip()
        extended = (c.get("context_english") or "").strip()
        block = header + "\nMain: " + main_text
        if extended and extended != main_text:
            block += "\nExtended: " + extended
        formatted_chunks.append(block)

    joined = "\n\n".join(formatted_chunks)

    instructions = (
        "You are an Islamic assistant. Use ONLY the verses provided below to answer. "
        "First, silently synthesize the key themes across all chunks. Do not claim insufficiency if the answer can be reasonably inferred from any combination of the provided verses. "
        "When citing, reference Surah and verse range in brackets like [Surah 2: 153 - 154]. "
        "If genuinely none of the retrieved content addresses the query, state 'I don't have enough provided context to answer.' without adding external information. "
        "Avoid hallucination; do not introduce verses not shown."
    )

    return (
        f"{instructions}\n\nQuery: {query}\n\nRetrieved Verse Chunks:\n{joined}\n\nAnswer:".
        strip()
    )


def answer_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    # Retrieval
    results, contexts = semantic_search(query, top_k=top_k)
    # LLM
    llm = _get_llm()
    prompt = build_prompt(query, contexts)
    response = llm.invoke(prompt)
    return {
        "query": query,
        "contexts": results,
        "answer": getattr(response, "content", str(response))
    }
