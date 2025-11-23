"""Agentic RAG service with enhanced tool orchestration.

Provides two implementations:
1. Agent class: Enhanced traditional flow with improved tools
2. GraphAgent: Production LangGraph implementation (recommended)

Both share the same tool ecosystem but GraphAgent provides:
- State management and checkpointing
- Conditional routing based on query analysis  
- Better observability and debugging
- Conversation memory across sessions
"""

from typing import Dict, Any, List, Optional, Literal, Iterator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config.secrets import GOOGLE_API_KEY, GOOGLE_MODEL
from app.services import tools as toollib


def _get_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it to use the agent.")
    return ChatGoogleGenerativeAI(model=GOOGLE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)


def build_prompt(query: str, 
                 contexts: List[Dict[str, Any]],
                 tool_outputs: Dict[str, Any]) -> str:
    """
    Construct a structured prompt from contexts and tool outputs.
    Supports both Quran verses and Hadiths with appropriate formatting.
    """
    
    if not contexts:
        return (
            "You're a helpful Islamic knowledge assistant. Unfortunately, no relevant content was found for this query. "
            "Let the user know in a warm, conversational way that you'd need more specific context or a rephrased question to help them better.\n\n"
            f"User's question: {query}\n\nYour response:"
        )

    formatted_chunks = []
    has_quran = False
    has_hadith = False
    
    for idx, c in enumerate(contexts, start=1):
        sim = c.get("similarity")
        
        # Check if this is a Quran verse or Hadith
        if c.get("surah_id"):
            # Quran verse
            has_quran = True
            surah = c.get("surah_id", "?")
            verse_range = c.get("verse_range", "?")
            identifier = c.get("chunk_key") or c.get("chunk_id") or f"verse_{idx}"
            
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
            # Hadith
            has_hadith = True
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
            # Generic fallback
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
    if (summary_data := tool_outputs.get("summarize_contexts")):
        if summary_data.get("status") == "success":
            summary_text = summary_data.get("summary", "").strip()
            if summary_text:
                summary_extra = f"\n\nKey Themes Summary:\n{summary_text}\n"

    # Determine source type for instructions
    if has_quran and has_hadith:
        source_desc = "Quranic verses and Hadiths"
        citation_example = "[Surah 2:153] or [Bukhari #123]"
    elif has_hadith:
        source_desc = "Hadiths (sayings and actions of Prophet Muhammad ﷺ)"
        citation_example = "[Bukhari #123] or [Book name #number]"
    else:
        source_desc = "Quranic verses"
        citation_example = "[Surah 2:153-154]"

    instructions = (
        f"You're a knowledgeable Islamic assistant helping someone understand {source_desc}. "
        "Your style is warm, clear, and conversational - like a thoughtful teacher. "
        "Work only with the texts shown below. Take a moment to understand the themes, then explain them naturally. "
        f"When you reference a source, cite it casually like {citation_example} so it flows with your explanation. "
        "If the texts don't fully address what they're asking, be honest - acknowledge what you found and suggest they might need additional sources. "
        "Keep your response grounded in what's actually provided. Be helpful and authentic."
    )

    return (
        f"{instructions}\n\n"
        f"User's question: {query}{summary_extra}\n"
        f"Retrieved Content:\n{joined}\n\n"
        "Your answer:"
    )


class Agent:
    """Agent orchestrating tool execution + final LLM answer."""

    def __init__(self, llm_factory=_get_llm, tools: Optional[List[toollib.Tool]] = None):
        self._llm_factory = llm_factory
        self._available_tools = tools or toollib.default_tools(llm_factory)


    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description} for t in self._available_tools
        ]


    def _select_tools(self, query: str) -> List[toollib.Tool]:
        return toollib.pick_tools(query, self._available_tools)


    def answer(self, query: str, top_k: int = 5, include_metrics: bool = False) -> Dict[str, Any]:
        """Process query with tool execution and LLM synthesis.
        
        Args:
            query: User's natural language query
            top_k: Number of verses to retrieve (1-20)
            include_metrics: Include execution time and token metrics
            
        Returns:
            Dictionary with answer, contexts, tool outputs, and optional metrics
        """
        import time
        start_time = time.time()
        
        selected = self._select_tools(query)
        tool_payloads: Dict[str, Any] = {}
        retrieval_contexts: List[Dict[str, Any]] = []
        retrieval_results: List[Dict[str, Any]] = []
        tool_errors: List[str] = []
        expanded_queries = [query]  # Default

        # Ordered tool execution with enhanced error tracking
        for tool in selected:
            tool_start = time.time()
            try:
                if tool.name == "expand_query":
                    output = tool.run(query=query)
                    tool_payloads[tool.name] = output
                    if output.get("status") == "success":
                        expanded_queries = output.get("expanded_queries", [query])
                    else:
                        tool_errors.append(f"Query expansion failed: {output.get('error', 'Unknown error')}")
                        
                elif tool.name in ["retrieval_quran", "retrieval_hadith"]:
                    # Use expanded queries for better retrieval
                    all_results = []
                    all_contexts = []
                    seen_chunk_keys = set()
                    
                    for exp_query in expanded_queries[:3]:  # Limit to top 3
                        output = tool.run(query=exp_query, top_k=max(3, top_k // len(expanded_queries[:3])))
                        if output.get("status") == "success":
                            for res in output.get("results", []):
                                # Use appropriate unique identifier
                                chunk_key = res.get("chunk_key") or res.get("hadith_id") or res.get("chunk_id")
                                if chunk_key and chunk_key not in seen_chunk_keys:
                                    seen_chunk_keys.add(chunk_key)
                                    all_results.append(res)
                            for ctx in output.get("contexts", []):
                                chunk_key = ctx.get("chunk_key") or ctx.get("hadith_id") or ctx.get("chunk_id")
                                if chunk_key and chunk_key not in seen_chunk_keys or chunk_key is None:
                                    all_contexts.append(ctx)
                    
                    # Sort by similarity and limit
                    all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    all_contexts.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    
                    # Keep top K per source, then combine
                    source_results = all_results[:top_k]
                    source_contexts = all_contexts[:top_k]
                    
                    retrieval_results.extend(source_results)
                    retrieval_contexts.extend(source_contexts)
                    
                    tool_payloads[tool.name] = {
                        "status": "success",
                        "source": output.get("source", tool.name.replace("retrieval_", "")),
                        "results": source_results,
                        "contexts": source_contexts,
                        "count": len(source_results),
                        "queries_used": expanded_queries[:3]
                    }
                        
                elif tool.name == "summarize_contexts":
                    if retrieval_contexts:
                        output = tool.run(contexts=retrieval_contexts, query=query)
                        tool_payloads[tool.name] = output
                        if output.get("status") == "error":
                            tool_errors.append(f"Summarization failed: {output.get('error', 'Unknown error')}")
                            
            except Exception as e:
                error_msg = f"{tool.name} execution failed: {str(e)}"
                tool_errors.append(error_msg)
                tool_payloads[tool.name] = {"status": "error", "error": str(e)}
            
            if include_metrics:
                tool_payloads[tool.name]["execution_time_ms"] = int((time.time() - tool_start) * 1000)

        # Final LLM synthesis
        llm = self._llm_factory()
        prompt = build_prompt(query, retrieval_contexts, tool_payloads)
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer_text = getattr(response, "content", str(response))
        except Exception as e:
            answer_text = f"Error generating answer: {str(e)}"
            tool_errors.append(str(e))
        
        result = {
            "query": query,
            "tools_used": list(tool_payloads.keys()),
            "tool_outputs": tool_payloads,
            "contexts": retrieval_results,
            "answer": answer_text,
        }
        
        if tool_errors:
            result["errors"] = tool_errors
        
        if include_metrics:
            result["metrics"] = {
                "total_execution_time_ms": int((time.time() - start_time) * 1000),
                "context_count": len(retrieval_contexts),
                "tool_count": len(selected)
            }
        
        return result
    
    def answer_stream(self, query: str, top_k: int = 5) -> Iterator[Dict[str, Any]]:
        """Stream agent execution with incremental updates.
        
        Yields:
            Dictionary with step updates and final answer tokens
        """
        import time
        
        selected = self._select_tools(query)
        expanded_queries = [query]
        retrieval_contexts = []
        retrieval_results = []
        tool_payloads = {}
        
        # Yield initial status
        yield {
            "type": "status",
            "message": "Let me help you with that...",
            "tools_selected": [t.name for t in selected]
        }
        
        # Execute tools with streaming updates
        for tool in selected:
            friendly_messages = {
                "expand_query": "Thinking of different ways to understand your question...",
                "retrieval": "Searching through relevant verses...",
                "summarize_contexts": "Identifying key themes for you..."
            }
            yield {
                "type": "tool_start",
                "tool_name": tool.name,
                "message": friendly_messages.get(tool.name, f"Working on {tool.name}...")
            }
            
            tool_start = time.time()
            
            try:
                if tool.name == "expand_query":
                    output = tool.run(query=query)
                    tool_payloads[tool.name] = output
                    if output.get("status") == "success":
                        expanded_queries = output.get("expanded_queries", [query])
                        yield {
                            "type": "tool_complete",
                            "tool_name": tool.name,
                            "output": output,
                            "execution_time_ms": int((time.time() - tool_start) * 1000)
                        }
                        
                elif tool.name in ["retrieval_quran", "retrieval_hadith"]:
                    # Use expanded queries
                    all_results = []
                    all_contexts = []
                    seen_chunk_keys = set()
                    
                    for exp_query in expanded_queries[:3]:
                        output = tool.run(query=exp_query, top_k=max(3, top_k // len(expanded_queries[:3])))
                        if output.get("status") == "success":
                            for res in output.get("results", []):
                                chunk_key = res.get("chunk_key") or res.get("hadith_id") or res.get("chunk_id")
                                if chunk_key and chunk_key not in seen_chunk_keys:
                                    seen_chunk_keys.add(chunk_key)
                                    all_results.append(res)
                            for ctx in output.get("contexts", []):
                                chunk_key = ctx.get("chunk_key") or ctx.get("hadith_id") or ctx.get("chunk_id")
                                if chunk_key and chunk_key not in seen_chunk_keys or chunk_key is None:
                                    all_contexts.append(ctx)
                    
                    all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    all_contexts.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    
                    source_results = all_results[:top_k]
                    source_contexts = all_contexts[:top_k]
                    
                    retrieval_results.extend(source_results)
                    retrieval_contexts.extend(source_contexts)
                    
                    tool_payloads[tool.name] = {
                        "status": "success",
                        "source": output.get("source", tool.name.replace("retrieval_", "")),
                        "results": source_results,
                        "contexts": source_contexts,
                        "count": len(source_results)
                    }
                    
                    yield {
                        "type": "tool_complete",
                        "tool_name": tool.name,
                        "output": tool_payloads[tool.name],
                        "execution_time_ms": int((time.time() - tool_start) * 1000)
                    }
                    
                elif tool.name == "summarize_contexts":
                    if retrieval_contexts:
                        output = tool.run(contexts=retrieval_contexts, query=query)
                        tool_payloads[tool.name] = output
                        yield {
                            "type": "tool_complete",
                            "tool_name": tool.name,
                            "output": output,
                            "execution_time_ms": int((time.time() - tool_start) * 1000)
                        }
                        
            except Exception as e:
                yield {
                    "type": "tool_error",
                    "tool_name": tool.name,
                    "error": str(e)
                }
        
        # Stream LLM response
        yield {
            "type": "synthesis_start",
            "message": "Now let me explain what these verses mean..."
        }
        
        llm = self._llm_factory()
        prompt = build_prompt(query, retrieval_contexts, tool_payloads)
        
        try:
            # Stream tokens
            full_answer = ""
            for chunk in llm.stream([HumanMessage(content=prompt)]):
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
                "contexts": retrieval_results,
                "tool_outputs": tool_payloads,
                "tools_used": list(tool_payloads.keys())
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }


def answer_query(query: str, top_k: int = 5, use_graph: bool = False) -> Dict[str, Any]:
    """Backward-compatible functional API.
    
    Args:
        query: User's query
        top_k: Number of verses to retrieve
        use_graph: If True, uses LangGraph implementation (recommended for production)
    """
    if use_graph:
        try:
            from app.services.graph_agent import GraphAgent
            agent = GraphAgent()
            return agent.answer(query=query)
        except ImportError:
            # Fallback to traditional if graph dependencies missing
            pass
    
    agent = Agent()
    return agent.answer(query=query, top_k=top_k)
