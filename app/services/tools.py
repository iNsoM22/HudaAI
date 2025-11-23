"""Enhanced tool abstractions for the agentic RAG system.

Each tool exposes:
 - name: unique identifier
 - description: short natural language description for LLM tool selection
 - input_model: pydantic model validating inputs
 - run(): executes and returns a dict payload (serializable)
 - metadata: additional properties (category, requires_llm, etc.)

Supports both heuristic and LLM-based tool selection strategies.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Type, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from app.utils import retrieval


class ToolMetadata(BaseModel):
    """Metadata describing tool characteristics."""
    category: Literal["retrieval", "analysis", "generation", "utility"] = "utility"
    requires_llm: bool = False
    cost_estimate: Literal["low", "medium", "high"] = "low"
    latency_estimate: Literal["fast", "medium", "slow"] = "fast"


class Tool:
    """Base tool with enhanced metadata and error handling."""
    name: str = "tool"
    description: str = "Generic tool"
    input_model: Type[BaseModel] | None = None
    metadata: ToolMetadata = ToolMetadata()

    def validate(self, **kwargs) -> BaseModel | None:
        """Validate inputs against pydantic schema."""
        if self.input_model is None:
            return None
        try:
            return self.input_model(**kwargs)
        except Exception as e:
            raise ValueError(f"Tool '{self.name}' input validation failed: {e}")

    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute tool logic and return structured result."""
        raise NotImplementedError(f"Tool '{self.name}' must implement run().")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool information for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.metadata.category,
            "parameters": self.input_model.model_json_schema() if self.input_model else {}
        }


class RetrievalInput(BaseModel):
    query: str = Field(..., description="User natural language query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class HadithRetrievalInput(BaseModel):
    query: str = Field(..., description="User natural language query")
    top_k: int = Field(5, ge=1, le=20, description="Number of hadiths to retrieve")
    book_filter: Optional[int] = Field(None, description="Optional book ID to filter results")
    match_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")


class QuranRetrievalTool(Tool):
    name = "retrieval_quran"
    description = "Search Quranic verses using semantic search - finds relevant verses from the Quran based on meaning and themes"
    input_model = RetrievalInput
    metadata = ToolMetadata(category="retrieval", cost_estimate="medium", latency_estimate="medium")

    def run(self, **kwargs) -> Dict[str, Any]:
        data = self.validate(**kwargs)
        try:
            results, contexts = retrieval.semantic_search_quran(data.query, top_k=data.top_k)
            return {
                "status": "success",
                "source": "quran",
                "query": data.query,
                "results": results,
                "contexts": contexts,
                "count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": data.query,
                "source": "quran",
                "results": [],
                "contexts": []
            }


class HadithRetrievalTool(Tool):
    name = "retrieval_hadith"
    description = "Search Hadith collections using semantic search - finds relevant hadiths (sayings and actions of Prophet Muhammad ﷺ) based on meaning"
    input_model = HadithRetrievalInput
    metadata = ToolMetadata(category="retrieval", cost_estimate="medium", latency_estimate="medium")

    def run(self, **kwargs) -> Dict[str, Any]:
        data = self.validate(**kwargs)
        try:
            results, contexts = retrieval.semantic_search_hadiths(
                data.query, 
                top_k=data.top_k,
                book_filter=data.book_filter,
                match_threshold=data.match_threshold
            )
            result_data = {
                "status": "success",
                "source": "hadith",
                "query": data.query,
                "results": results,
                "contexts": contexts,
                "count": len(results)
            }
            
            print(f"[HadithRetrievalTool] Retrieved {len(results)} hadiths for query: '{data.query}'")
            return result_data
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": data.query,
                "source": "hadith",
                "results": [],
                "contexts": []
            }


class SummaryInput(BaseModel):
    contexts: List[Dict[str, Any]] = Field(..., description="List of context chunks from retrieval")
    query: str = Field(..., description="Original user query for focus")
    max_points: int = Field(6, ge=1, le=15, description="Maximum bullet points to output")


class QueryExpansionInput(BaseModel):
    query: str = Field(..., description="Original user query")
    num_variations: int = Field(3, ge=1, le=5, description="Number of query variations")


class QueryExpansionTool(Tool):
    name = "expand_query"
    description = "Expand user query into multiple semantic variations for better retrieval coverage"
    input_model = QueryExpansionInput
    metadata = ToolMetadata(category="analysis", requires_llm=True, cost_estimate="low", latency_estimate="fast")

    def __init__(self, llm_factory):
        self._llm_factory = llm_factory

    def run(self, **kwargs) -> Dict[str, Any]:
        data = self.validate(**kwargs)
        try:
            llm = self._llm_factory()
            prompt = f"""I need help expanding this question about Islamic teachings into {data.num_variations} different ways of asking the same thing. This helps find more relevant verses.

Original question: {data.query}

Create variations that:
- Rephrase using different words but keep the same meaning
- Include related concepts or synonyms someone might use
- Match how verses might actually phrase these ideas

Just list them naturally, one per line (no numbers or bullets):"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            variations_text = getattr(response, "content", str(response))
            
            # Parse variations
            variations = [v.strip() for v in variations_text.split('\n') if v.strip() and not v.strip().startswith('#')]
            variations = [v.lstrip('123456789.-*• ') for v in variations][:data.num_variations]
            
            # Include original query
            all_queries = [data.query] + variations
            
            return {
                "status": "success",
                "original_query": data.query,
                "expanded_queries": all_queries,
                "variation_count": len(variations)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "expanded_queries": [data.query]  # Fallback to original
            }


class SummarizeContextsTool(Tool):
    name = "summarize_contexts"
    description = "Generate a concise summary of retrieved Islamic texts (Quran verses or Hadiths) with key themes"
    input_model = SummaryInput
    metadata = ToolMetadata(category="analysis", requires_llm=True, cost_estimate="medium", latency_estimate="medium")

    def __init__(self, llm_factory):
        self._llm_factory = llm_factory

    def run(self, **kwargs) -> Dict[str, Any]:
        data = self.validate(**kwargs)
        try:
            llm = self._llm_factory()
            bullet_texts = []
            source_type = "mixed"
            
            for c in data.contexts:
                # Detect if this is a Quran verse or Hadith
                if c.get("surah_id"):
                    # Quran verse
                    source_type = "quran" if source_type == "mixed" or source_type == "quran" else "mixed"
                    main = c.get("text_english") or c.get("text_uthmani") or ""
                    verse_range = c.get("verse_range", "?")
                    surah = c.get("surah_id", "?")
                    bullet_texts.append(f"Quran - Surah {surah} {verse_range}: {main[:180].strip()}")
                elif c.get("hadith_id"):
                    # Hadith
                    source_type = "hadith" if source_type == "mixed" or source_type == "hadith" else "mixed"
                    main = c.get("chunk_text") or c.get("context_english") or ""
                    book = c.get("book_name", "Unknown")
                    hadith_num = c.get("hadith_number", "?")
                    bullet_texts.append(f"Hadith - {book} #{hadith_num}: {main[:180].strip()}")
                else:
                    # Generic fallback
                    main = c.get("text_english") or c.get("chunk_text") or ""
                    bullet_texts.append(main[:180].strip())

            source_label = "Quranic verses" if source_type == "quran" else "Hadiths" if source_type == "hadith" else "Islamic texts"
            prompt = (
                f"Help me understand these {source_label} by highlighting the {data.max_points} most important themes and lessons. "
                f"The person asked about: {data.query}\n\n"
                f"Here are the {source_label}:\n" + "\n".join(bullet_texts) + "\n\n"
                "Share your insights as clear bullet points (•), making each one meaningful but easy to grasp."
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            summary_text = getattr(resp, "content", str(resp))
            return {
                "status": "success",
                "summary": summary_text,
                "item_count": len(data.contexts),
                "source_type": source_type
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "summary": ""
            }


def default_tools(llm_factory) -> List[Tool]:
    """Return the default tool set for the agent with both Quran and Hadith retrieval."""
    return [
        QueryExpansionTool(llm_factory),
        QuranRetrievalTool(),
        HadithRetrievalTool(),
        SummarizeContextsTool(llm_factory),
    ]


def pick_tools_heuristic(query: str, available: List[Tool]) -> List[Tool]:
    """Heuristic-based tool selection (fast, no LLM call).

    Rules:
    - Expand query first for better coverage (always)
    - Select appropriate retrieval sources based on query content:
      * Hadith-specific keywords → Hadith only
      * Quran-specific keywords → Quran only
      * General Islamic questions → Both sources
    - Add summarization if requested
    """
    ql = query.lower()
    selected: List[Tool] = []
    
    # Start with query expansion for better results
    expansion_tool = next((t for t in available if t.name == "expand_query"), None)
    if expansion_tool:
        selected.append(expansion_tool)
    
    # Intelligent source selection
    hadith_keywords = ["hadith", "sunnah", "prophet said", "narrated", "sahih", "bukhari", "muslim", 
                       "abu dawud", "tirmidhi", "ibn majah", "prophet muhammad", "messenger said"]
    quran_keywords = ["quran", "verse", "surah", "ayah", "ayat", "revelation", "allah said", "quranic"]
    
    has_hadith_keyword = any(kw in ql for kw in hadith_keywords)
    has_quran_keyword = any(kw in ql for kw in quran_keywords)
    
    # Get retrieval tools
    quran_tool = next((t for t in available if t.name == "retrieval_quran"), None)
    hadith_tool = next((t for t in available if t.name == "retrieval_hadith"), None)
    
    # Decision logic
    if has_hadith_keyword and not has_quran_keyword:
        # Explicitly asking about Hadith
        if hadith_tool:
            selected.append(hadith_tool)
    elif has_quran_keyword and not has_hadith_keyword:
        # Explicitly asking about Quran
        if quran_tool:
            selected.append(quran_tool)
    else:
        # General question or mentions both - search both sources
        if quran_tool:
            selected.append(quran_tool)
        if hadith_tool:
            selected.append(hadith_tool)

    # Add summarization if requested
    if any(k in ql for k in ("summary", "summarize", "outline", "briefly", "key points")):
        summary_tool = next((t for t in available if t.name == "summarize_contexts"), None)
        if summary_tool:
            selected.append(summary_tool)
    
    return selected


class ToolSelectionInput(BaseModel):
    """Structured output for LLM-based tool selection."""
    selected_tools: List[str] = Field(description="List of tool names to use in order")
    reasoning: str = Field(description="Brief explanation of tool selection")
    estimated_steps: int = Field(ge=1, le=10, description="Expected number of steps")


def pick_tools_llm(query: str, available: List[Tool], llm_factory) -> List[Tool]:
    """LLM-based tool selection using structured output.
    
    More sophisticated than heuristics but requires an LLM call.
    Use for complex queries where tool selection isn't obvious.
    """
    llm = llm_factory()
    structured_llm = llm.with_structured_output(ToolSelectionInput)
    
    tools_desc = "\n".join([
        f"- {t.name}: {t.description} (category: {t.metadata.category})"
        for t in available
    ])
    
    prompt = f"""Given this user query about Islamic knowledge, select the optimal tools to use.

Query: {query}

Available Tools:
{tools_desc}

Consider:
1. Query expansion helps find better matches
2. Choose retrieval sources wisely:
   - Use retrieval_quran for questions about Quranic verses
   - Use retrieval_hadith for questions about Prophet's sayings/actions
   - Use BOTH for general Islamic questions to get comprehensive answers
3. Summarization is useful for "summary", "outline", or "key points" requests
4. Tools should be ordered by execution sequence

Select tools strategically to provide the most complete answer."""
    
    try:
        selection: ToolSelectionInput = structured_llm.invoke([HumanMessage(content=prompt)])
        selected = []
        for tool_name in selection.selected_tools:
            tool = next((t for t in available if t.name == tool_name), None)
            if tool:
                selected.append(tool)
        return selected if selected else pick_tools_heuristic(query, available)
    except Exception:
        # Fallback to heuristic if LLM selection fails
        return pick_tools_heuristic(query, available)


def pick_tools(query: str, available: List[Tool], strategy: Literal["heuristic", "llm"] = "heuristic", llm_factory=None) -> List[Tool]:
    """Main tool selection interface with configurable strategy.
    
    Args:
        query: User query
        available: List of available tools
        strategy: "heuristic" (fast, rule-based) or "llm" (intelligent, requires LLM call)
        llm_factory: Required if strategy="llm"
    """
    if strategy == "llm" and llm_factory:
        return pick_tools_llm(query, available, llm_factory)
    return pick_tools_heuristic(query, available)
