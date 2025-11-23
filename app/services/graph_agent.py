"""LangGraph-based agentic RAG system with proper state management.

This implementation follows industry best practices:
- TypedDict state schema for type safety
- Structured tool calling with Pydantic validation
- Conditional routing based on query analysis
- Memory checkpointing for conversation history
- LangSmith tracing integration
"""

from __future__ import annotations

from typing import Annotated, TypedDict, Sequence, Literal
from typing_extensions import TypedDict as TypedDictExt

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from app.utils.retrieval import semantic_search_hadiths, semantic_search_quran, get_records
from app.config.secrets import GOOGLE_API_KEY, GOOGLE_MODEL


# ============================================================================
# State Schema - Tracks all information through the graph
# ============================================================================

class AgentState(TypedDict):
    """Complete state for the agentic RAG workflow."""
    messages: Annotated[Sequence[BaseMessage], "conversation history"]
    query: str
    contexts: list[dict]
    retrieval_results: list[dict]
    summary: str | None
    needs_summary: bool
    needs_context_expansion: bool
    tool_outputs: dict
    final_answer: str | None


# ============================================================================
# Pydantic Models for Structured Outputs
# ============================================================================

class QueryAnalysis(BaseModel):
    """Structured analysis of user query to determine tool selection."""
    intent: Literal["factual", "summary", "comparison", "explanation"] = Field(
        description="Primary intent of the query"
    )
    needs_summary: bool = Field(
        default=False,
        description="Whether user explicitly wants a summarized response"
    )
    needs_context_expansion: bool = Field(
        default=False,
        description="Whether query requires broader verse context"
    )
    search_quran: bool = Field(
        default=True,
        description="Whether to search Quranic verses"
    )
    search_hadith: bool = Field(
        default=True,
        description="Whether to search Hadiths"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of verses/hadiths to retrieve per source"
    )
    reasoning: str = Field(
        description="Brief reasoning for the analysis decisions"
    )


class VerseMetadata(BaseModel):
    """Metadata about a specific verse."""
    surah_name: str
    revelation_type: Literal["Meccan", "Medinan", "Unknown"]
    theme: str
    historical_context: str | None = None


# ============================================================================
# LangChain Tools with Structured Outputs
# ============================================================================

@tool
def retrieve_verses(query: str, top_k: int = 5) -> dict:
    """Semantic search over Quranic verses returning relevant contexts.
    
    Args:
        query: Natural language query from the user
        top_k: Number of most relevant verses to retrieve (1-20)
        
    Returns:
        Dictionary with 'results' (full details) and 'contexts' (for LLM)
    """
    results, contexts = semantic_search_quran(query, top_k=top_k)
    return {
        "source": "quran",
        "results": results,
        "contexts": contexts,
        "count": len(results)
    }


@tool
def retrieve_hadiths(query: str, top_k: int = 5, book_filter: int | None = None, match_threshold: float = 0.5) -> dict:
    """Semantic search over Hadiths returning relevant contexts.
    
    Args:
        query: Natural language query from the user
        top_k: Number of most relevant hadiths to retrieve (1-20)
        book_filter: Optional book ID to filter by specific collection
        match_threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        Dictionary with 'results' (full details) and 'contexts' (for LLM)
    """
    results, contexts = semantic_search_hadiths(query, top_k=top_k, book_filter=book_filter, match_threshold=match_threshold)
    return {
        "source": "hadith",
        "results": results,
        "contexts": contexts,
        "count": len(results)
    }


@tool
def summarize_verses(contexts: list[dict], query: str, max_points: int = 6) -> dict:
    """Generate a concise bullet-point summary of retrieved verses.
    
    Args:
        contexts: List of verse context dictionaries
        query: Original user query for focused summary
        max_points: Maximum number of bullet points (1-15)
        
    Returns:
        Dictionary with 'summary' text and 'points_count'
    """
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    
    bullet_texts = []
    for c in contexts:
        main = c.get("text_english") or c.get("text_uthmani") or ""
        verse_range = c.get("verse_range", "?")
        surah = c.get("surah_id", "?")
        bullet_texts.append(f"Surah {surah} {verse_range}: {main[:180].strip()}")
    
    prompt = (
        f"Create a thematic summary of these Quranic verses/Hadiths with at most {max_points} focused bullet points.\n\n"
        f"User's Question: {query}\n\n"
        f"Retrieved Content:\n" + "\n".join(bullet_texts) + "\n\n"
        "Guidelines for effective summarization:\n"
        "1. Identify core Islamic themes: faith (iman), worship (ibadah), morality (akhlaq), guidance (hidayah), mercy (rahma), justice (adl)\n"
        "2. Extract key teachings and divine commands present in the verses\n"
        "3. Note any repeated concepts or complementary messages across different verses\n"
        "4. Preserve the reverent tone and precise meaning of the original texts\n"
        "5. Connect the themes directly to the user's query\n\n"
        "Format each bullet point as: Theme/Concept - Brief explanation with reference\n"
        "Example: • Patience in adversity - Allah promises ease after hardship (Surah 94)"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    summary_text = response.content
    
    return {
        "summary": summary_text,
        "points_count": len([l for l in summary_text.split("\n") if l.strip().startswith(("•", "-", "*", "1", "2", "3", "4", "5", "6"))])
    }


@tool
def expand_context(source_type: str, identifier: str, context_window: int = 3) -> dict:
    """Expand context by retrieving surrounding verses or hadiths.
    
    Args:
        source_type: 'quran' or 'hadith' to specify which source to expand
        identifier: chunk_key for Quran verses, hadith_id for Hadiths
        context_window: Number of items before/after to retrieve (default: 3)
        
    Returns:
        Dictionary with expanded context items and metadata
    """
    from app.utils.retrieval import get_expanded_context
    
    result = get_expanded_context(
        source_type=source_type,
        identifier=identifier,
        context_window=context_window
    )
    
    return result


# ============================================================================
# LangGraph Node Functions
# ============================================================================

def analyze_query(state: AgentState) -> AgentState:
    """Analyze query to determine routing and tool selection."""
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    
    structured_llm = llm.with_structured_output(QueryAnalysis)
    
    analysis_prompt = f"""Analyze this Islamic knowledge query to optimize retrieval from Quran and Hadiths.

Query: {state["query"]}

Determine the best approach by considering:

1. **Query Intent Classification**:
   - Factual: Direct questions about rulings, facts, or specific teachings (e.g., "What does Islam say about...")
   - Summary: Requests for overview or multiple perspectives (e.g., "Explain the concept of...", "What are the teachings on...")
   - Comparison: Contrasting different aspects or related concepts (e.g., "difference between...", "compare...")
   - Explanation: Deep dive into meaning, context, or interpretation (e.g., "Why...", "What is the wisdom...")

2. **Islamic Terminology Detection**:
   - Look for Arabic terms (sabr, taqwa, zakat, salah, jihad, iman, etc.)
   - Identify concepts (patience, faith, charity, prayer, fasting, paradise, hellfire, prophets)
   - Detect references to specific sources (Quran, Hadith, Bukhari, Muslim, specific Surahs)

3. **Source Selection**:
   - Search Quran only: If query explicitly mentions "Quran", "verse", "surah", "ayah", "revelation"
   - Search Hadith only: If query explicitly mentions "hadith", "sunnah", "prophet said", "bukhari", "muslim", "narrator"
   - Search both (default): For general Islamic questions, concepts, or when source isn't specified

4. **Retrieval Optimization**:
   - For specific topics: 5-8 items per source for focused depth
   - For broad concepts: 10-15 items per source for comprehensive coverage
   - For comparisons: 12-20 items per source to show multiple perspectives
   - For simple queries: 3-5 items per source for direct answers

5. **Context Requirements**:
   - Needs summary: When query asks for "overview", "main teachings", "key points"
   - Needs context expansion: When understanding requires surrounding verses/hadiths (narrative passages, conditional statements)

Provide reasoning that explains which Islamic concepts/terms were detected, which sources to search, and why the chosen approach will retrieve the most relevant content."""
    
    analysis: QueryAnalysis = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
    
    return {
        **state,
        "needs_summary": analysis.needs_summary,
        "needs_context_expansion": analysis.needs_context_expansion,
        "tool_outputs": {
            "analysis": {
                **analysis.model_dump(),
                "search_quran": analysis.search_quran,
                "search_hadith": analysis.search_hadith
            }
        }
    }


def retrieve_node(state: AgentState) -> AgentState:
    """Execute semantic search retrieval for Quran and/or Hadith."""
    analysis = state.get("tool_outputs", {}).get("analysis", {})
    top_k = analysis.get("top_k", 5)
    search_quran = analysis.get("search_quran", True)
    search_hadith = analysis.get("search_hadith", True)
    
    all_results = []
    all_contexts = []
    retrieval_info = {}
    
    # Retrieve from Quran if needed
    if search_quran:
        quran_result = retrieve_verses.invoke({
            "query": state["query"],
            "top_k": top_k
        })
        all_results.extend(quran_result["results"])
        all_contexts.extend(quran_result["contexts"])
        retrieval_info["quran"] = {
            "count": quran_result["count"],
            "source": "quran"
        }
    
    # Retrieve from Hadith if needed
    if search_hadith:
        hadith_result = retrieve_hadiths.invoke({
            "query": state["query"],
            "top_k": top_k,
            "book_filter": None,
            "match_threshold": 0.5
        })
        all_results.extend(hadith_result["results"])
        all_contexts.extend(hadith_result["contexts"])
        retrieval_info["hadith"] = {
            "count": hadith_result["count"],
            "source": "hadith"
        }
    
    return {
        **state,
        "retrieval_results": all_results,
        "contexts": all_contexts,
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "retrieval": retrieval_info,
            "total_retrieved": len(all_results)
        }
    }


def summarize_node(state: AgentState) -> AgentState:
    """Generate summary if needed."""
    if not state.get("contexts"):
        return state
    
    tool_result = summarize_verses.invoke({
        "contexts": state["contexts"],
        "query": state["query"],
        "max_points": 6
    })
    
    return {
        **state,
        "summary": tool_result["summary"],
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "summary": tool_result
        }
    }


def synthesize_answer(state: AgentState) -> AgentState:
    """Final LLM synthesis with all retrieved context."""
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    
    # Build rich prompt from state
    contexts = state.get("contexts", [])
    
    if not contexts:
        return {
            **state,
            "final_answer": "I apologize, but I couldn't retrieve any relevant verses for your query. Please try rephrasing your question."
        }
    
    # Format contexts - handle both Quran and Hadith
    formatted_chunks = []
    for idx, c in enumerate(contexts, start=1):
        sim = c.get("similarity")
        
        # Check if this is a Quran verse or Hadith
        if c.get("surah_id"):
            # Quran verse formatting
            surah = c.get("surah_id", "?")
            verse_range = c.get("verse_range", "?")
            identifier = c.get("chunk_key", f"chunk_{idx}")
            
            header = f"[Verse {idx} | ID {identifier} | Surah {surah} | Verses {verse_range}"
            if isinstance(sim, (int, float)):
                header += f" | Similarity {sim:.4f}]"
            else:
                header += "]"
            
            main_text = (c.get("text_english") or c.get("text_uthmani") or "[No text]").strip()
            extended = (c.get("context_english") or "").strip()
            
            block = f"{header}\nText: {main_text}"
            if extended and extended != main_text:
                block += f"\nContext: {extended}"
        
        elif c.get("hadith_id"):
            # Hadith formatting
            hadith_id = c.get("hadith_id", "?")
            book_name = c.get("book_name", "Unknown")
            hadith_num = c.get("hadith_number", "?")
            
            header = f"[Hadith {idx} | {book_name} #{hadith_num}"
            if isinstance(sim, (int, float)):
                header += f" | Similarity {sim:.4f}]"
            else:
                header += "]"
            
            chunk_text = (c.get("chunk_text") or "").strip()
            context_text = (c.get("context_english") or "").strip()
            
            if chunk_text and context_text and chunk_text != context_text:
                block = f"{header}\nMatched part: {chunk_text}\nFull hadith: {context_text}"
            else:
                block = f"{header}\nText: {context_text or chunk_text or '[No text]'}"
        
        else:
            # Fallback for unknown format
            identifier = c.get("chunk_key") or c.get("chunk_id") or f"item_{idx}"
            header = f"[Item {idx} | ID {identifier}"
            if isinstance(sim, (int, float)):
                header += f" | Similarity {sim:.4f}]"
            else:
                header += "]"
            
            text = c.get("text_english") or c.get("chunk_text") or c.get("context_english") or "[No text]"
            block = f"{header}\nText: {text.strip()}"
        
        formatted_chunks.append(block)
    
    joined = "\n\n".join(formatted_chunks)
    
    # Add summary if available
    summary_extra = ""
    if state.get("summary"):
        summary_extra = f"\n\nPre-Summary:\n{state['summary']}\n"
    
    system_instructions = """You are a knowledgeable Islamic assistant helping someone understand Quranic verses and Hadiths. Your role is to provide accurate, reverent, and contextually rich explanations.

CORE PRINCIPLES:
1. Prefer the verses/hadiths provided below in comparison to your knowledge. Never fabricate verses or hadiths.
2. Before answering, identify common themes, complementary teachings, and the overall message across all provided texts.
3. Citation should be made in these format: 
   - For Quran: [Surah X:Y-Z] or [Surah Name X:Y]
   - For Hadith: [Bukhari #XXX] or [Book Name #XXX]
4. Respect the elevated, precise language of Quran; maintain the narrative style of Hadiths.
5. Consider verse context, historical background (when evident), and how verses relate to each other.
6. If the provided texts don't fully address the query, clearly state your issue, and ask for more information.

RESPONSE STRUCTURE:
- Open with direct relevance to the user's question
- Weave in verses naturally with proper citations
- Explain key concepts or Arabic terms when they appear
- Connect related teachings across different verses
- Conclude with a cohesive summary when appropriate"""
    
    user_prompt = f"""User's Question: {state["query"]}{summary_extra}

Retrieved Quranic Verses and Hadiths:
{joined}

INSTRUCTIONS:
Analyze the provided texts carefully, identify the key themes and teachings most relevant to the user's question, then provide a comprehensive, well-structured answer that:
- Directly addresses their question
- Synthesizes multiple verses/hadiths from your knowledge or provided ones, to show the complete picture
- Cites sources naturally within your explanation
- Explains any important concepts or terminology

Your response:"""    
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        **state,
        "final_answer": response.content,
        "messages": state.get("messages", []) + messages + [AIMessage(content=response.content)]
    }


# ============================================================================
# Routing Functions
# ============================================================================

def should_summarize(state: AgentState) -> Literal["summarize", "synthesize"]:
    """Route to summarization if needed."""
    if state.get("needs_summary", False) and state.get("contexts"):
        return "summarize"
    return "synthesize"


def should_expand_context(state: AgentState) -> Literal["expand", "continue"]:
    """Route to context expansion if needed."""
    if state.get("needs_context_expansion", False):
        return "expand"
    return "continue"


# ============================================================================
# Graph Construction
# ============================================================================

def create_agent_graph():
    """Construct the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("synthesize", synthesize_answer)
    
    # Define edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    
    # Conditional routing after retrieval
    workflow.add_conditional_edges(
        "retrieve",
        should_summarize,
        {
            "summarize": "summarize",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("summarize", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile with checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================================================
# Public API
# ============================================================================

class GraphAgent:
    """Production-grade agentic RAG using LangGraph."""
    
    def __init__(self):
        self.graph = create_agent_graph()
    
    def answer(self, query: str, thread_id: str = "default", top_k: int = 5, include_metrics: bool = False) -> dict:
        """Process query through the agent graph.
        
        Args:
            query: User's natural language query
            thread_id: Conversation thread identifier for checkpointing
            top_k: Number of results per source (used in initial state)
            include_metrics: Whether to include execution metrics
            
        Returns:
            Complete state including answer, contexts, and tool outputs
        """
        import time
        start_time = time.time()
        
        initial_state = AgentState(
            messages=[],
            query=query,
            contexts=[],
            retrieval_results=[],
            summary=None,
            needs_summary=False,
            needs_context_expansion=False,
            tool_outputs={},
            final_answer=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Execute graph
        final_state = self.graph.invoke(initial_state, config)
        
        result = {
            "query": query,
            "answer": final_state.get("final_answer", ""),
            "contexts": final_state.get("retrieval_results", []),
            "tools_used": list(final_state.get("tool_outputs", {}).keys()),
            "tool_outputs": final_state.get("tool_outputs", {}),
            "thread_id": thread_id
        }
        
        if include_metrics:
            result["metrics"] = {
                "total_execution_time_ms": int((time.time() - start_time) * 1000),
                "context_count": len(final_state.get("retrieval_results", [])),
                "tool_count": len(final_state.get("tool_outputs", {}))
            }
        
        return result
    
    def answer_stream(self, query: str, thread_id: str = "default", top_k: int = 5):
        """Stream agent execution with incremental updates.
        
        Yields:
            Dictionary with step updates and final answer tokens
        """
        import time
        
        # Yield initial status
        yield {
            "type": "status",
            "message": "Analyzing your question...",
            "tools_selected": []
        }
        
        initial_state = AgentState(
            messages=[],
            query=query,
            contexts=[],
            retrieval_results=[],
            summary=None,
            needs_summary=False,
            needs_context_expansion=False,
            tool_outputs={},
            final_answer=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Execute graph and track progress
        try:
            # Step 1: Analysis
            yield {
                "type": "tool_start",
                "tool_name": "analyze_query",
                "message": "Understanding your question..."
            }
            
            analysis_start = time.time()
            state_after_analysis = analyze_query(initial_state)
            analysis = state_after_analysis.get("tool_outputs", {}).get("analysis", {})
            
            # Determine which tools will be used
            tools_to_use = []
            if analysis.get("search_quran", True):
                tools_to_use.append("retrieval_quran")
            if analysis.get("search_hadith", True):
                tools_to_use.append("retrieval_hadith")
            if analysis.get("needs_summary", False):
                tools_to_use.append("summarize_contexts")
            
            yield {
                "type": "status",
                "message": "Analysis complete",
                "tools_selected": tools_to_use
            }
            
            # Step 2: Retrieval
            state_after_retrieval = retrieve_node(state_after_analysis)
            retrieval_info = state_after_retrieval.get("tool_outputs", {}).get("retrieval", {})
            
            # Yield completion for each retrieval source
            if "quran" in retrieval_info:
                yield {
                    "type": "tool_start",
                    "tool_name": "retrieval_quran",
                    "message": "Searching Quran..."
                }
                yield {
                    "type": "tool_complete",
                    "tool_name": "retrieval_quran",
                    "output": {
                        "status": "success",
                        "count": retrieval_info["quran"].get("count", 0),
                        "results": [r for r in state_after_retrieval.get("retrieval_results", []) if r.get("surah_id")],
                        "contexts": [c for c in state_after_retrieval.get("contexts", []) if c.get("surah_id")]
                    },
                    "execution_time_ms": 0
                }
            
            if "hadith" in retrieval_info:
                yield {
                    "type": "tool_start",
                    "tool_name": "retrieval_hadith",
                    "message": "Searching Hadiths..."
                }
                yield {
                    "type": "tool_complete",
                    "tool_name": "retrieval_hadith",
                    "output": {
                        "status": "success",
                        "count": retrieval_info["hadith"].get("count", 0),
                        "results": [r for r in state_after_retrieval.get("retrieval_results", []) if r.get("hadith_id")],
                        "contexts": [c for c in state_after_retrieval.get("contexts", []) if c.get("hadith_id")]
                    },
                    "execution_time_ms": 0
                }
            
            # Step 3: Optional Summarization
            state_after_summary = state_after_retrieval
            if analysis.get("needs_summary", False) and state_after_retrieval.get("contexts"):
                yield {
                    "type": "tool_start",
                    "tool_name": "summarize_contexts",
                    "message": "Summarizing key themes..."
                }
                state_after_summary = summarize_node(state_after_retrieval)
                summary_output = state_after_summary.get("tool_outputs", {}).get("summary", {})
                yield {
                    "type": "tool_complete",
                    "tool_name": "summarize_contexts",
                    "output": summary_output,
                    "execution_time_ms": 0
                }
            
            # Step 4: Synthesis with streaming
            yield {
                "type": "synthesis_start",
                "message": "Generating answer..."
            }
            
            # Get LLM and stream the answer
            llm = ChatGoogleGenerativeAI(
                model=GOOGLE_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
            
            # Build prompt from current state
            contexts = state_after_summary.get("contexts", [])
            if not contexts:
                yield {
                    "type": "complete",
                    "query": query,
                    "answer": "I apologize, but I couldn't retrieve any relevant content for your query. Please try rephrasing your question.",
                    "contexts": [],
                    "tool_outputs": state_after_summary.get("tool_outputs", {}),
                    "tools_used": list(state_after_summary.get("tool_outputs", {}).keys())
                }
                return
            
            # Format contexts for prompt
            formatted_chunks = []
            for idx, c in enumerate(contexts, start=1):
                sim = c.get("similarity")
                
                if c.get("surah_id"):
                    surah = c.get("surah_id", "?")
                    verse_range = c.get("verse_range", "?")
                    identifier = c.get("chunk_key", f"chunk_{idx}")
                    header = f"[Verse {idx} | ID {identifier} | Surah {surah} | Verses {verse_range}"
                    if isinstance(sim, (int, float)):
                        header += f" | Similarity {sim:.4f}]"
                    else:
                        header += "]"
                    main_text = (c.get("text_english") or c.get("text_uthmani") or "[No text]").strip()
                    extended = (c.get("context_english") or "").strip()
                    block = f"{header}\nText: {main_text}"
                    if extended and extended != main_text:
                        block += f"\nContext: {extended}"
                elif c.get("hadith_id"):
                    hadith_id = c.get("hadith_id", "?")
                    book_name = c.get("book_name", "Unknown")
                    hadith_num = c.get("hadith_number", "?")
                    header = f"[Hadith {idx} | {book_name} #{hadith_num}"
                    if isinstance(sim, (int, float)):
                        header += f" | Similarity {sim:.4f}]"
                    else:
                        header += "]"
                    chunk_text = (c.get("chunk_text") or "").strip()
                    context_text = (c.get("context_english") or "").strip()
                    if chunk_text and context_text and chunk_text != context_text:
                        block = f"{header}\nMatched part: {chunk_text}\nFull hadith: {context_text}"
                    else:
                        block = f"{header}\nText: {context_text or chunk_text or '[No text]'}"
                else:
                    identifier = c.get("chunk_key") or c.get("chunk_id") or f"item_{idx}"
                    header = f"[Item {idx} | ID {identifier}"
                    if isinstance(sim, (int, float)):
                        header += f" | Similarity {sim:.4f}]"
                    else:
                        header += "]"
                    text = c.get("text_english") or c.get("chunk_text") or c.get("context_english") or "[No text]"
                    block = f"{header}\nText: {text.strip()}"
                
                formatted_chunks.append(block)
            
            joined = "\n\n".join(formatted_chunks)
            
            summary_extra = ""
            if state_after_summary.get("summary"):
                summary_extra = f"\n\nPre-Summary:\n{state_after_summary['summary']}\n"
            
            system_instructions = """You are a knowledgeable Islamic assistant helping someone understand Quranic verses and Hadiths. Your role is to provide accurate, reverent, and contextually rich explanations.

CORE PRINCIPLES:
1. Prefer the verses/hadiths provided below in comparison to your knowledge. Never fabricate verses or hadiths.
2. Before answering, identify common themes, complementary teachings, and the overall message across all provided texts.
3. Citation should be made in these format: 
   - For Quran: [Surah X:Y-Z] or [Surah Name X:Y]
   - For Hadith: [Bukhari #XXX] or [Book Name #XXX]
4. Respect the elevated, precise language of Quran; maintain the narrative style of Hadiths.
5. Consider verse context, historical background (when evident), and how verses relate to each other.
6. If the provided texts don't fully address the query, clearly state your issue, and ask for more information.

RESPONSE STRUCTURE:
- Open with direct relevance to the user's question
- Weave in verses naturally with proper citations
- Explain key concepts or Arabic terms when they appear
- Connect related teachings across different verses
- Conclude with a cohesive summary when appropriate"""
            
            user_prompt = f"""User's Question: {query}{summary_extra}

Retrieved Quranic Verses and Hadiths:
{joined}

INSTRUCTIONS:
Analyze the provided texts carefully, identify the key themes and teachings most relevant to the user's question, then provide a comprehensive, well-structured answer that:
- Directly addresses their question
- Synthesizes multiple verses/hadiths from your knowledge or provided ones, to show the complete picture
- Cites sources naturally within your explanation
- Explains any important concepts or terminology

Your response:"""
            
            messages = [
                SystemMessage(content=system_instructions),
                HumanMessage(content=user_prompt)
            ]
            
            # Stream the response
            full_answer = ""
            for chunk in llm.stream(messages):
                token = getattr(chunk, "content", "")
                if token:
                    full_answer += token
                    yield {
                        "type": "answer_token",
                        "token": token,
                        "full_answer": full_answer
                    }
            
            # Final complete message
            yield {
                "type": "complete",
                "query": query,
                "answer": full_answer,
                "contexts": state_after_retrieval.get("retrieval_results", []),
                "tool_outputs": state_after_summary.get("tool_outputs", {}),
                "tools_used": list(state_after_summary.get("tool_outputs", {}).keys())
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def get_state(self, thread_id: str = "default") -> dict:
        """Retrieve current state for a conversation thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config)
        return state_snapshot.values if state_snapshot else {}
    
    def visualize(self) -> str:
        """Return mermaid diagram of the graph structure."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return "Graph visualization requires additional dependencies"


# Backward-compatible function
def answer_query(query: str, top_k: int = 5) -> dict:
    """Backward-compatible API using new graph agent."""
    agent = GraphAgent()
    return agent.answer(query)
