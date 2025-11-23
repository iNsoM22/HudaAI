import streamlit as st
import sys
import os
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    
from app.utils.retrieval import semantic_search_quran, semantic_search_hadiths

st.set_page_config(page_title="HudaAI - Quran & Hadith Explorer", layout="wide")
st.title("🕌 HudaAI - Islamic Knowledge Explorer")

st.markdown(
    """
Ask me anything about Islam! I'll search both the Quran and Hadith collections to help you:

1. 🔍 **Understanding** - I'll figure out what you're really asking
2. 📚 **Searching** - Find relevant content from Quran, Hadith, or both
3. 📝 **Summarizing** - Pull out key themes if you want a quick overview
4. 💬 **Explaining** - Give you a clear answer based on authentic sources

Watch my thought process unfold below!
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
    """Display retrieved contexts from both Quran and Hadith sources."""
    if not contexts:
        st.info("No matching content was retrieved.")
        return
    
    for c in contexts:
        sim = c.get("similarity")
        
        # Check if this is a Quran verse or Hadith
        if c.get("surah_id"):
            # Quran verse
            verse_range = c.get("verse_range") or "N/A"
            surah = c.get("surah_id") or "?"
            header = f"📚 Quran - Surah {surah} | Verses {verse_range}"
            if sim is not None:
                header += f" | ⭐ {sim:.4f}"
            st.markdown(f"**{header}**")
            
            if c.get("text_english"):
                st.write(c["text_english"].strip())
            if c.get("text_uthmani"):
                with st.expander("🌙 Arabic Text"):
                    st.write(c["text_uthmani"].strip())
            if c.get("context_english") or c.get("context_uthmani"):
                with st.expander("📜 Extended Context"):
                    if c.get("context_english"):
                        st.write(c["context_english"].strip())
                    if c.get("context_uthmani"):
                        st.write(c["context_uthmani"].strip())
                        
        elif c.get("hadith_id"):
            # Hadith
            book_name = c.get("book_name", "Unknown Collection")
            hadith_num = c.get("hadith_number", "?")
            header = f"📜 Hadith - {book_name} #{hadith_num}"
            if sim is not None:
                header += f" | ⭐ {sim:.4f}"
            st.markdown(f"**{header}**")
            
            # Show matched chunk if different from full context
            chunk_text = c.get("chunk_text", "").strip()
            context_text = c.get("context_english", "").strip()
            
            if chunk_text and context_text and chunk_text != context_text:
                st.info(f"🎯 **Matched part:** {chunk_text}")
                with st.expander("📜 Full Hadith"):
                    st.write(context_text)
            else:
                st.write(context_text or chunk_text or "[No text available]")
            
            if c.get("context_arabic"):
                with st.expander("🌙 Arabic Text"):
                    st.write(c["context_arabic"].strip())
        else:
            # Generic fallback
            header = "Content"
            if sim is not None:
                header += f" | Similarity: {sim:.4f}"
            st.markdown(f"**{header}**")
            text = c.get("text_english") or c.get("chunk_text") or c.get("context_english") or "[No text]"
            st.write(text.strip())
        
        st.markdown("---")


def display_execution_steps(result: Dict[str, Any], step_containers: Dict[str, Any] = None) -> None:
    """Display agent execution steps with visual indicators.
    
    Args:
        result: Result dictionary from agent
        step_containers: Optional dict of streamlit containers for live updates
    """
    st.subheader("🔄 Agent Execution Steps")
    
    tools_used = result.get("tools_used", [])
    tool_outputs = result.get("tool_outputs", {})
    metrics = result.get("metrics", {})
    errors = result.get("errors", [])
    
    # Display metrics summary
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{metrics.get('total_execution_time_ms', 0)}ms")
        with col2:
            st.metric("Tools Used", metrics.get('tool_count', len(tools_used)))
        with col3:
            st.metric("Contexts Retrieved", metrics.get('context_count', 0))
    
    # Step 0: Query Expansion (if present)
    if "expand_query" in tool_outputs:
        expansion_data = tool_outputs["expand_query"]
        status = expansion_data.get("status", "unknown")
        
        with st.expander(
            f"{'✅' if status == 'success' else '❌'} Step 0: Query Expansion",
            expanded=True
        ):
            if status == "success":
                expanded_queries = expansion_data.get("expanded_queries", [])
                st.success(f"Expanded into {len(expanded_queries)} queries for better coverage")
                
                if "execution_time_ms" in expansion_data:
                    st.caption(f"⏱️ Execution time: {expansion_data['execution_time_ms']}ms")
                
                st.markdown("**Query Variations:**")
                for idx, q in enumerate(expanded_queries, 1):
                    icon = "🎯" if idx == 1 else "🔄"
                    st.write(f"{icon} {q}")
            else:
                st.error(f"Expansion failed: {expansion_data.get('error', 'Unknown error')}")
    
    # Step 1: Tool Selection
    with st.expander("✅ Step 1: Tool Selection & Query Analysis", expanded=True):
        if tools_used:
            st.success(f"Selected {len(tools_used)} tool(s): {', '.join(tools_used)}")
            for tool_name in tools_used:
                tool_icon = "🔍" if tool_name == "retrieval" else "📝" if tool_name == "summarize_contexts" else "🔄" if tool_name == "expand_query" else "🔧"
                st.write(f"{tool_icon} **{tool_name}**")
        else:
            st.warning("No tools were selected")
    
    # Step 2: Retrieval (Quran and/or Hadith)
    for source_name in ["retrieval_quran", "retrieval_hadith"]:
        if source_name in tool_outputs:
            retrieval_data = tool_outputs[source_name]
            status = retrieval_data.get("status", "unknown")
            source_label = "Quran" if "quran" in source_name else "Hadith"
            source_icon = "📚" if "quran" in source_name else "📜"
            
            with st.expander(
                f"{'✅' if status == 'success' else '❌'} Step 2: {source_icon} {source_label} Search",
                expanded=(status != "success")
            ):
                if status == "success":
                    count = retrieval_data.get("count", 0)
                    st.success(f"Found {count} relevant {source_label.lower()} {'items' if count != 1 else 'item'}")
                    
                    if "execution_time_ms" in retrieval_data:
                        st.caption(f"⏱️ Execution time: {retrieval_data['execution_time_ms']}ms")
                    
                    # Show top similarities
                    contexts = retrieval_data.get("contexts", [])
                    if contexts:
                        similarities = [c.get("similarity", 0) for c in contexts if c.get("similarity")]
                        if similarities:
                            st.write(f"📊 Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")
                            st.write(f"📈 Average similarity: {sum(similarities)/len(similarities):.4f}")
                else:
                    st.error(f"{source_label} search failed: {retrieval_data.get('error', 'Unknown error')}")
    
    # Step 3: Optional Summarization
    if "summarize_contexts" in tool_outputs:
        summary_data = tool_outputs["summarize_contexts"]
        status = summary_data.get("status", "unknown")
        
        with st.expander(
            f"{'✅' if status == 'success' else '❌'} Step 3: Context Summarization",
            expanded=True
        ):
            if status == "success":
                summary_text = summary_data.get("summary", "")
                st.success("Generated thematic summary")
                
                if "execution_time_ms" in summary_data:
                    st.caption(f"⏱️ Execution time: {summary_data['execution_time_ms']}ms")
                
                st.markdown("**Summary:**")
                st.info(summary_text)
            else:
                st.error(f"Summarization failed: {summary_data.get('error', 'Unknown error')}")
    
    # Step 3.5: Web Search (if executed)
    if "web_search" in tool_outputs:
        web_data = tool_outputs["web_search"]
        status = web_data.get("status", "unknown")
        
        with st.expander(
            f"{'✅' if status == 'success' else '❌'} Step 3: 🌐 Web Search for Additional Context",
            expanded=True
        ):
            if status == "success":
                results = web_data.get("results", [])
                st.success(f"Found {len(results)} relevant web sources for contemporary context")
                
                st.markdown("**Web Search Results:**")
                for idx, result in enumerate(results, 1):
                    st.markdown(f"**{idx}. {result['title']}**")
                    st.caption(f"🔗 Source: {result['source']}")
                    st.write(result['snippet'])
                    st.markdown(f"[Read more]({result['link']})")
                    if idx < len(results):
                        st.markdown("---")
            else:
                st.error(f"Web search failed: {web_data.get('error', 'Unknown error')}")
    
    # Step 4: Final Synthesis
    with st.expander("✅ Step 4: LLM Answer Synthesis", expanded=False):
        if result.get("answer"):
            st.success("Generated comprehensive answer from retrieved verses")
            st.write("The LLM synthesized the answer using only the provided verse contexts.")
        else:
            st.warning("No answer was generated")
    
    # Show errors if any
    if errors:
        with st.expander("⚠️ Errors & Warnings", expanded=True):
            for error in errors:
                st.error(error)


st.markdown("---")
st.subheader("💬 Ask a Question")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What does Islam teach about patience? Or: Find hadiths about charity",
        help="Ask naturally - I'll search both Quran and Hadith! Mention 'hadith' or 'verse' to focus on one source.",
    )

with col2:
    top_k = st.slider("Verse chunks", min_value=1, max_value=15, value=5)

show_metrics = st.checkbox("Show execution metrics", value=True, help="Display timing and performance data")
use_streaming = st.checkbox("Enable streaming", value=True, help="Stream responses in real-time")
generate = st.button("🚀 Generate Answer", type="primary")

if generate:
    cleaned = (query or "").strip()
    if not cleaned:
        st.warning("I'd love to help, but I need a question first! 😊")
    elif len(cleaned) < 3:
        st.warning("That's a bit short - could you add a few more words so I can understand better?")
    else:
        from app.services.graph_agent import GraphAgent
        agent = GraphAgent()
        
        if use_streaming:
            # Streaming mode with real-time updates
            try:
                # Create containers for live updates
                status_container = st.empty()
                steps_container = st.container()
                answer_container = st.empty()
                contexts_container = st.container()
                
                tool_outputs = {}
                tools_used = []
                final_answer = ""
                final_contexts = []
                
                # Stream agent execution
                for update in agent.answer_stream(cleaned, top_k=top_k):
                    update_type = update.get("type")
                    
                    if update_type == "status":
                        with status_container:
                            st.info(f"🔄 {update.get('message')}")
                            if update.get("tools_selected"):
                                tools = update['tools_selected']
                                friendly_names = []
                                for t in tools:
                                    if t == 'expand_query': friendly_names.append('expanding your question')
                                    elif t == 'retrieval_quran': friendly_names.append('searching Quran')
                                    elif t == 'retrieval_hadith': friendly_names.append('searching Hadiths')
                                    elif t == 'summarize_contexts': friendly_names.append('summarizing themes')
                                    else: friendly_names.append(t)
                                st.write(f"**My plan:** {', '.join(friendly_names)}")
                    
                    elif update_type == "tool_start":
                        with status_container:
                            tool_name = update["tool_name"]
                            friendly_msg = {
                                "expand_query": "🔄 Thinking of different ways to phrase your question...",
                                "retrieval_quran": "📚 Searching through the Quran...",
                                "retrieval_hadith": "📜 Searching through Hadith collections...",
                                "summarize_contexts": "📝 Pulling out key themes..."
                            }.get(tool_name, f"🛠️ Working on {tool_name}...")
                            st.info(friendly_msg)
                    
                    elif update_type == "tool_complete":
                        tool_name = update.get("tool_name")
                        tool_outputs[tool_name] = update.get("output", {})
                        tools_used.append(tool_name)
                        
                        with steps_container:
                            friendly_label = {
                                "expand_query": "✅ Rephrased your question",
                                "retrieval_quran": "✅ Found Quranic verses",
                                "retrieval_hadith": "✅ Found Hadiths",
                                "summarize_contexts": "✅ Key themes identified"
                            }.get(tool_name, f"✅ {tool_name} complete")
                            
                            with st.expander(f"{friendly_label} ({update.get('execution_time_ms', 0)}ms)", expanded=False):
                                if tool_name == "expand_query" and update["output"].get("status") == "success":
                                    queries = update["output"].get("expanded_queries", [])
                                    st.write(f"**Found {len(queries)} ways to ask this:**")
                                    for idx, q in enumerate(queries, 1):
                                        icon = "🎯" if idx == 1 else "🔄"
                                        st.write(f"{icon} {q}")
                                elif tool_name in ["retrieval_quran", "retrieval_hadith"] and update["output"].get("status") == "success":
                                    count = update["output"].get("count", 0)
                                    source = "verses" if "quran" in tool_name else "hadiths"
                                    st.write(f"**✅ Found {count} relevant {source}**")
                                    # Accumulate contexts from both sources
                                    final_contexts.extend(update["output"].get("results", []))
                                elif tool_name == "summarize_contexts" and update["output"].get("status") == "success":
                                    st.write(update["output"].get("summary", ""))
                    
                    elif update_type == "synthesis_start":
                        with status_container:
                            st.info("💬 Putting it all together for you...")
                        with answer_container:
                            st.subheader("💡 Here's what I found (streaming...)")
                    
                    elif update_type == "answer_token":
                        final_answer = update.get("full_answer", "")
                        with answer_container:
                            st.subheader("💡 Answer")
                            st.markdown(final_answer)
                    
                    elif update_type == "complete":
                        final_answer = update.get("answer", "")
                        final_contexts = update.get("contexts", [])
                        
                        with status_container:
                            st.success("✅ All done! Hope this helps 😊")
                        
                        with answer_container:
                            st.subheader("💡 Your Answer")
                            if "Error" in final_answer:
                                st.error(final_answer)
                            elif any(phrase in final_answer.lower() for phrase in ["don't have enough", "need more", "couldn't find"]):
                                st.info(final_answer)
                            else:
                                st.markdown(final_answer)
                        
                        # Show contexts
                        with contexts_container:
                            with st.expander("📚 View Retrieved Content (Quran & Hadith)", expanded=False):
                                display_contexts(final_contexts)
                    
                    elif update_type == "tool_error" or update_type == "error":
                        with status_container:
                            st.error(f"❌ Error: {update.get('error')}")
                
            except Exception as e:
                st.error(f"❌ Oops, something went wrong: {e}")
                st.info("Try rephrasing your question or let me know if this keeps happening!")
                import traceback
                with st.expander("🔍 Debug Information"):
                    st.code(traceback.format_exc())
        else:
            # Non-streaming mode (original)
            try:
                # Show progress with different steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Figuring out the best way to help you...")
                progress_bar.progress(25)
                
                with st.spinner("Looking through the verses and crafting your answer..."):
                    result = agent.answer(cleaned, top_k=top_k, include_metrics=show_metrics)
                    
                progress_bar.progress(100)
                status_text.text("✅ Got it!")
                
                # Display execution steps first
                display_execution_steps(result)
                
                # Then show the final answer
                st.markdown("---")
                st.subheader("💡 Final Answer")
                answer_text = result.get("answer", "[No answer returned]").strip()
                
                if "Error generating answer" in answer_text:
                    st.error(answer_text)
                elif "don't have enough" in answer_text.lower() or "need more" in answer_text.lower():
                    st.info(answer_text)
                else:
                    st.markdown(answer_text)
                
                # Show retrieved contexts in expandable section
                with st.expander("📚 View Retrieved Content (Quran & Hadith)", expanded=False):
                    display_contexts(result.get("contexts", []))
                    
            except Exception as e:
                st.error(f"❌ Error during generation: {e}")
                import traceback
                with st.expander("🔍 Debug Information"):
                    st.code(traceback.format_exc())

st.sidebar.header("🔎 Quick Search")
st.sidebar.markdown("Want to see what verses match your search? Try this!")

search_q = st.sidebar.text_input(
    "Search query",
    placeholder="e.g., mercy, patience, guidance",
    help="Returns top matching verse chunks without running the LLM.",
)

if st.sidebar.button("🔍 Search", type="primary"):
    cleaned = (search_q or "").strip()
    if not cleaned:
        st.sidebar.warning("What would you like to search for?")
    else:
        try:
            with st.sidebar.spinner("Searching..."):
                raw_results, _ = semantic_search_quran(cleaned, top_k=5)
                
            st.sidebar.success(f"✅ Found {len(raw_results)} matching verse{'s' if len(raw_results) != 1 else ''}!")
            
            for idx, h in enumerate(raw_results, 1):
                verse_range = h.get("verse_range") or "?"
                surah = h.get("surah_id") or "?"
                sim = h.get("similarity")
                caption = f"**{idx}. Surah {surah} | {verse_range}**"
                if sim is not None:
                    caption += f" | ⭐ {sim:.3f}"
                st.sidebar.markdown(caption)
                preview = h.get("text_english") or h.get("text_uthmani") or "[No text]"
                preview = preview.strip()
                st.sidebar.write(preview[:150] + ("..." if len(preview) > 150 else ""))
                st.sidebar.markdown("---")
        except Exception as e:
            st.sidebar.error(f"❌ Search failed: {e}")

# Add info about the system
st.sidebar.markdown("---")
st.sidebar.info(
    """**About HudaAI**

I'm powered by:
• 📚 Quranic verses (complete text)
• 📜 Hadith collections (authentic narrations)
• 🔍 Smart search understanding meaning, not just keywords
• 🧠 AI explaining Islamic teachings clearly
• ✅ Answers based strictly on authentic sources

Your friendly Islamic knowledge companion! 🕌
    """
)