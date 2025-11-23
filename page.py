import streamlit as st
import sys
import os
from typing import List, Dict, Any

# --- Project path setup (keeps imports working when running from app root) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.utils.retrieval import semantic_search_quran, semantic_search_hadiths

# Page config
st.set_page_config(page_title="HudaAI - Quran & Hadith Explorer", layout="wide")

# -----------------------------
# Styles (kept minimal & Streamlit-friendly)
# -----------------------------
st.markdown(
    """
    <style>
        /* Modern card styling */
        .stExpander {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Smooth animations */
        .stMarkdown {
            animation: fadeIn 0.28s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Better button styling */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 50px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.18s ease;
        }
        .stButton > button:hover { transform: translateY(-2px); }

        /* Input field styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e8e8ee;
            padding: 12px;
            font-size: 16px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.08);
        }

        /* Metric cards */
        .stMetric { padding: 12px; border-radius: 10px; }

        /* Subtle divider spacing */
        .divider { margin: 16px 0; }
        
        /* Current step highlight */
        .current-step {
            border-left: 4px solid #667eea;
            padding-left: 12px;
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
            margin: 10px 0;
        }
        
        /* Auto-scroll container */
        .scroll-container {
            max-height: 600px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }
    </style>
    <script>
        // Auto-scroll to bottom of answer
        function scrollToAnswer() {
            const answerElement = document.querySelector('[data-testid="stMarkdownContainer"]');
            if (answerElement) {
                answerElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        }
        setInterval(scrollToAnswer, 500);
    </script>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Page header
# -----------------------------
st.title("🕌 HudaAI - Islamic Knowledge Explorer")
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: white;'>✨ Ask Anything About Islam</h3>
        <p style='margin: 10px 0 0 0; opacity: 0.95;'>
            I'll search the Quran and Hadith collections, analyze the context, and provide you with authentic Islamic knowledge backed by sources.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper utilities
# -----------------------------

def normalize_contexts(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map retrieval schema to the agent prompt builder keys while preserving metadata."""
    normalized = []
    for r in raw:
        verse_id = r.get("chunk_key") or r.get("chunk_id") or r.get("surah_id")
        content_text = r.get("text_english") or r.get("text_uthmani") or "[No verse text]"
        normalized.append({
            "verse_id": verse_id,
            "content": content_text,
            **r,
        })
    return normalized


def display_contexts(contexts: List[Dict[str, Any]]) -> None:
    """Display retrieved contexts from Quran and Hadith sources."""
    if not contexts:
        st.info("No matching content was retrieved.")
        return

    for c in contexts:
        sim = c.get("similarity")

        if c.get("surah_id"):
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
            book_name = c.get("book_name", "Unknown Collection")
            hadith_num = c.get("hadith_number", "?")
            header = f"📜 Hadith - {book_name} #{hadith_num}"
            if sim is not None:
                header += f" | ⭐ {sim:.4f}"
            st.markdown(f"**{header}**")

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
            header = "Content"
            if sim is not None:
                header += f" | Similarity: {sim:.4f}"
            st.markdown(f"**{header}**")
            text = c.get("text_english") or c.get("chunk_text") or c.get("context_english") or "[No text]"
            st.write(text.strip())

        st.markdown("---")


def display_execution_steps(result: Dict[str, Any]) -> None:
    """Display agent execution steps (post-run summary)."""
    st.subheader("🔄 Agent Execution Steps")

    tools_used = result.get("tools_used", [])
    tool_outputs = result.get("tool_outputs", {})
    metrics = result.get("metrics", {})
    errors = result.get("errors", [])

    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{metrics.get('total_execution_time_ms', 0)}ms")
        with col2:
            st.metric("Tools Used", metrics.get('tool_count', len(tools_used)))
        with col3:
            st.metric("Contexts Retrieved", metrics.get('context_count', 0))

    # Show simple tool outputs summary
    for k, v in tool_outputs.items():
        with st.expander(f"Tool: {k}"):
            st.write(v)

    if errors:
        with st.expander("⚠️ Errors & Warnings", expanded=True):
            for error in errors:
                st.error(error)


# -----------------------------
# Main UI: query form + streaming generation
# -----------------------------

# initialize processing state
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What does Islam teach about patience? Or: What do scholars say about cryptocurrency?",
        help="Ask naturally - I'll search Quran, Hadith, and the web when needed!",
        key='query_input',
        label_visibility="visible",
        disabled=st.session_state.is_processing,
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    top_k = st.slider("Results", min_value=3, max_value=15, value=5, help="Number of verses/hadiths to retrieve")

# primary action button
generate = st.button(
    "Generate Answer" if not st.session_state.is_processing else "Processing...",
    type="primary",
    disabled=st.session_state.is_processing,
    use_container_width=True,
    key='generate_button'
)

# decide whether to process
should_process = generate

# validate & run
if should_process:
    cleaned = (query or "").strip()
    if not cleaned:
        st.warning("Please enter a question first.")
    elif len(cleaned) < 3:
        st.warning("Please provide more details for better results.")
    else:
        # set processing flag immediately so UI disables appropriately
        st.session_state.is_processing = True

        from app.services.graph_agent import GraphAgent
        agent = GraphAgent()

        # prepare UI placeholders for streaming
        st.markdown("### 🔍 Agent Thinking")
        current_step_container = st.empty()
        history_container = st.container()
        
        st.markdown("---")
        st.markdown("### 💡 Answer")
        answer_container = st.empty()
        
        st.markdown("---")
        contexts_container = st.container()
        
        st.markdown("---")
        st.markdown("### 📊 Execution Metrics")
        metrics_cols = st.columns(4)
        metric_placeholders = {
            'time': metrics_cols[0].empty(),
            'tools': metrics_cols[1].empty(),
            'contexts': metrics_cols[2].empty(),
            'status': metrics_cols[3].empty()
        }
        
        steps_summary = {"tools_used": [], "tool_outputs": {}, "metrics": {}, "errors": []}
        step_history = []
        step_count = 0
        start_time = None
        current_step_info = {}

        try:
            import time
            start_time = time.time()
            
            # streaming loop
            final_answer = ""
            final_contexts = []

            for update in agent.answer_stream(cleaned, top_k=top_k):
                update_type = update.get("type")
                
                # Update metrics in real-time
                if start_time:
                    elapsed = time.time() - start_time
                    metric_placeholders['time'].metric("Time", f"{elapsed:.1f}s")
                    metric_placeholders['tools'].metric("Tools Used", len(steps_summary['tools_used']))
                    metric_placeholders['contexts'].metric("Contexts", len(final_contexts))

                # STATUS updates
                if update_type == "status":
                    step_count += 1
                    current_step_info = {
                        'number': step_count,
                        'name': 'Analysis',
                        'message': update.get('message'),
                        'status': 'running'
                    }
                    
                    if update.get("tools_selected"):
                        plan = update.get('tools_selected', [])
                        tool_names = {
                            'expand_query': 'Query Expansion',
                            'retrieval_quran': 'Quran Search',
                            'retrieval_hadith': 'Hadith Search',
                            'summarize_contexts': 'Summarization',
                            'web_search': 'Web Search'
                        }
                        plan_friendly = [tool_names.get(p, p) for p in plan]
                        current_step_info['plan'] = ' → '.join(plan_friendly)
                    
                    # Display current step
                    with current_step_container.container():
                        st.markdown(f"<div class='current-step'>", unsafe_allow_html=True)
                        st.markdown(f"**Step {step_count}: {current_step_info['name']}**")
                        st.info(current_step_info['message'])
                        if 'plan' in current_step_info:
                            st.caption(f"Planned: {current_step_info['plan']}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    metric_placeholders['status'].metric("Status", "Analyzing")

                # TOOL START
                elif update_type == "tool_start":
                    tool_name = update.get("tool_name")
                    step_count += 1
                    
                    # Archive previous step
                    if current_step_info and current_step_info.get('number') != step_count:
                        step_history.append(current_step_info.copy())
                    
                    current_step_info = {
                        'number': step_count,
                        'name': tool_name.replace('_', ' ').title(),
                        'message': 'Executing...',
                        'status': 'running'
                    }
                    
                    # Display current step
                    with current_step_container.container():
                        st.markdown(f"<div class='current-step'>", unsafe_allow_html=True)
                        st.markdown(f"**Step {step_count}: {current_step_info['name']}**")
                        st.info("⏳ Executing...")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    metric_placeholders['status'].metric("Status", f"Running: {tool_name}")

                # TOOL COMPLETE
                elif update_type == "tool_complete":
                    tool_name = update.get("tool_name")
                    output = update.get("output", {})
                    exec_time = update.get('execution_time_ms', 0)
                    steps_summary['tools_used'].append(tool_name)
                    steps_summary['tool_outputs'][tool_name] = output
                    
                    # Build completion details
                    details = []
                    if output.get('status') == 'success':
                        current_step_info['status'] = 'success'
                        current_step_info['time'] = exec_time
                        
                        if tool_name == "expand_query":
                            queries = output.get("expanded_queries", [])
                            details.append(f"Generated {len(queries)} query variations")
                            
                        elif tool_name in ["retrieval_quran", "retrieval_hadith"]:
                            count = output.get("count", 0)
                            source = "Quran verses" if "quran" in tool_name else "Hadiths"
                            details.append(f"Found {count} {source}")
                            results = output.get('results', [])
                            if results:
                                sims = [r.get('similarity', 0) for r in results if r.get('similarity')]
                                if sims:
                                    details.append(f"Similarity: {min(sims):.3f} - {max(sims):.3f}")
                            final_contexts.extend(results)
                            
                        elif tool_name == "summarize_contexts":
                            details.append("Summary generated")
                            
                        elif tool_name == "web_search":
                            results = output.get("results", [])
                            details.append(f"Found {len(results)} web sources")
                    else:
                        current_step_info['status'] = 'error'
                        details.append(f"Failed: {output.get('error', 'Unknown error')}")
                    
                    current_step_info['details'] = details
                    current_step_info['output'] = output
                    
                    # Display current completed step
                    with current_step_container.container():
                        st.markdown(f"<div class='current-step'>", unsafe_allow_html=True)
                        st.markdown(f"**Step {step_count}: {current_step_info['name']} - Completed**")
                        if current_step_info['status'] == 'success':
                            st.success(f"✓ Completed in {exec_time}ms")
                            for detail in details:
                                st.caption(detail)
                        else:
                            st.error(details[0] if details else "Failed")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Update history dropdown
                    if len(step_history) > 0:
                        with history_container:
                            with st.expander(f"📋 Step History ({len(step_history)} previous steps)", expanded=False):
                                for hist_step in step_history:
                                    status_icon = "✓" if hist_step.get('status') == 'success' else "✗"
                                    st.caption(f"{status_icon} Step {hist_step['number']}: {hist_step['name']} ({hist_step.get('time', 0)}ms)")
                                    if hist_step.get('details'):
                                        for detail in hist_step['details']:
                                            st.text(f"  • {detail}")
                    
                    metric_placeholders['tools'].metric("Tools Used", len(steps_summary['tools_used']))
                    metric_placeholders['contexts'].metric("Contexts", len(final_contexts))

                # SYNTHESIS START
                elif update_type == "synthesis_start":
                    step_count += 1
                    
                    # Archive previous step
                    if current_step_info:
                        step_history.append(current_step_info.copy())
                    
                    current_step_info = {
                        'number': step_count,
                        'name': 'Synthesizing Answer',
                        'message': 'Generating comprehensive answer...',
                        'status': 'running'
                    }
                    
                    # Display current step
                    with current_step_container.container():
                        st.markdown(f"<div class='current-step'>", unsafe_allow_html=True)
                        st.markdown(f"**Step {step_count}: Synthesizing Answer**")
                        st.info("⏳ Generating comprehensive answer...")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    metric_placeholders['status'].metric("Status", "Synthesizing")

                # ANSWER TOKEN (incremental streaming chunk)
                elif update_type == "answer_token":
                    final_answer = update.get("full_answer", final_answer)
                    answer_container.markdown(final_answer)
                    metric_placeholders['status'].metric("Status", "Streaming...")

                # COMPLETE
                elif update_type == "complete":
                    final_answer = update.get("answer", final_answer)
                    final_contexts = update.get("contexts", final_contexts)
                    
                    # Archive final step
                    if current_step_info:
                        current_step_info['status'] = 'success'
                        step_history.append(current_step_info.copy())
                    
                    # Clear current step display
                    current_step_container.empty()
                    
                    # Show completion message
                    with current_step_container.container():
                        st.success("✓ All steps completed successfully")
                    
                    # Update final metrics
                    if start_time:
                        total_time = time.time() - start_time
                        metric_placeholders['time'].metric("Total Time", f"{total_time:.1f}s")
                    metric_placeholders['tools'].metric("Tools Used", len(steps_summary['tools_used']))
                    metric_placeholders['contexts'].metric("Contexts Retrieved", len(final_contexts))
                    metric_placeholders['status'].metric("Status", "✓ Complete")

                    # final answer
                    if "Error" in final_answer:
                        answer_container.error(final_answer)
                    elif any(phrase in final_answer.lower() for phrase in ["don't have enough", "need more", "couldn't find"]):
                        answer_container.info(final_answer)
                    else:
                        answer_container.markdown(final_answer)

                    # show contexts (collapsible)
                    with contexts_container:
                        with st.expander("📚 View Retrieved Sources", expanded=False):
                            display_contexts(final_contexts)

                    # Update history dropdown with all steps
                    with history_container:
                        with st.expander(f"📋 Complete Step History ({len(step_history)} steps)", expanded=False):
                            for hist_step in step_history:
                                status_icon = "✓" if hist_step.get('status') == 'success' else "✗" if hist_step.get('status') == 'error' else "⏳"
                                st.markdown(f"**{status_icon} Step {hist_step['number']}: {hist_step['name']}**")
                                if hist_step.get('time'):
                                    st.caption(f"Duration: {hist_step['time']}ms")
                                if hist_step.get('details'):
                                    for detail in hist_step['details']:
                                        st.caption(f"• {detail}")
                                st.markdown("---")

                    # collect tools/metrics if provided
                    steps_summary['metrics'] = update.get('metrics', steps_summary['metrics'])
                    steps_summary['errors'] = update.get('errors', steps_summary.get('errors', []))

                    # reset processing flag so UI becomes interactive again
                    st.session_state.is_processing = False

                # ERRORS
                elif update_type in ("tool_error", "error"):
                    err = update.get('error', 'Unknown error')
                    
                    # Archive previous step
                    if current_step_info:
                        step_history.append(current_step_info.copy())
                    
                    step_count += 1
                    
                    # Display error in current step
                    with current_step_container.container():
                        st.markdown(f"<div class='current-step'>", unsafe_allow_html=True)
                        st.markdown(f"**Step {step_count}: Error**")
                        st.error(f"Error: {err}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    metric_placeholders['status'].metric("Status", "✗ Error")
                    steps_summary.setdefault('errors', []).append(err)
                    st.session_state.is_processing = False
                    break

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            with st.expander("Debug Information"):
                st.code(traceback.format_exc())
            st.session_state.is_processing = False
            if start_time:
                metric_placeholders['status'].metric("Status", "✗ Failed")

# -----------------------------
# Sidebar quick search + about
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: white;'>Quick Search</h3>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>Search Quran verses directly</p>
    </div>
    """,
    unsafe_allow_html=True,
)

search_q = st.sidebar.text_input(
    "Search query",
    placeholder="e.g., mercy, patience, guidance",
    help="Returns top matching verse chunks without running the LLM.",
)

if st.sidebar.button("Search", type="primary"):
    cleaned = (search_q or "").strip()
    if not cleaned:
        st.sidebar.warning("Please enter a search query")
    else:
        try:
            loader = st.sidebar.empty()
            loader.info("🔄 Searching...")
            raw_results, _ = semantic_search_quran(cleaned, top_k=5)
            st.sidebar.success(f"Found {len(raw_results)} matching verse{'s' if len(raw_results) != 1 else ''}")

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
            st.sidebar.error(f"Search failed: {e}")
        
        finally:
            loader.empty()

# About card
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white;'>
        <h3 style='margin: 0; color: white;'>About HudaAI</h3>
        <p style='margin: 10px 0 0 0; font-size: 14px; line-height: 1.6;'>
            <strong>Powered by:</strong><br/>
            • Complete Quranic verses<br/>
            • Authentic Hadith collections<br/>
            • Contemporary web context<br/>
            • Semantic search technology<br/>
            • AI-powered explanations<br/>
            <br/>
            <em>Your trusted Islamic knowledge companion</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
