# 🚀 Quick Reference: Agentic RAG System

## 📋 What Changed

### Before (Basic RAG)
```python
# Simple retrieval + LLM
retrieval → prompt → LLM → answer
```

### After (Agentic RAG)
```python
# Tool-based with routing
query_analysis → tool_selection → [retrieval, summarization] → synthesis → answer
```

## 🎯 Key Improvements

| Feature | Benefit |
|---------|---------|
| 🔧 **Tool System** | Easy to add new capabilities (metadata, translations, etc.) |
| 📊 **Metrics** | Track timing, context counts, tool usage |
| 🔄 **State Management** | Conversation memory with LangGraph |
| ✅ **Error Handling** | Graceful degradation per tool |
| 🎨 **Interactive UI** | See agent reasoning in real-time |
| 🧪 **Type Safety** | Pydantic validation everywhere |

## 💻 Usage Examples

### 1. Simple Query (Backward Compatible)
```python
from app.services.agent import answer_query

result = answer_query("What is patience in Islam?", top_k=5)
print(result["answer"])
```

### 2. With Metrics
```python
from app.services.agent import Agent

agent = Agent()
result = agent.answer("query", include_metrics=True)

print(f"Time: {result['metrics']['total_execution_time_ms']}ms")
print(f"Tools: {result['tools_used']}")
```

### 3. LangGraph (Production)
```python
from app.services.graph_agent import GraphAgent

agent = GraphAgent()
result = agent.answer("Tell me about sabr", thread_id="user_123")
# Conversation memory preserved in thread_id
```

### 4. Custom Tool
```python
from app.services import tools as toollib
from pydantic import BaseModel

class MyInput(BaseModel):
    param: str

class MyTool(toollib.Tool):
    name = "my_tool"
    description = "Does something"
    input_model = MyInput
    
    def run(self, **kwargs):
        data = self.validate(**kwargs)
        return {"result": f"processed {data.param}"}

# Use it
agent = Agent(tools=[*toollib.default_tools(llm_factory), MyTool()])
```

## 🎨 UI Features

### Run the App
```bash
streamlit run page.py
```

### What You'll See
1. **📊 Metrics Dashboard** - Time, tool count, context count
2. **✅ Step 1** - Tool selection with reasoning
3. **🔍 Step 2** - Retrieval with similarity stats
4. **📝 Step 3** - Summarization (if query has "summary")
5. **💬 Step 4** - Final synthesis confirmation
6. **💡 Answer** - With verse citations
7. **📚 Contexts** - Expandable verse details

### Try These Queries

**Basic:**
```
What does the Quran say about patience?
```

**With Summary:**
```
Give me a summary of verses about charity
```

**Complex:**
```
Provide a brief outline of key themes about prayer
```

## 🏗️ Architecture

### File Structure
```
app/
├── services/
│   ├── agent.py           # Enhanced traditional agent
│   ├── graph_agent.py     # LangGraph production agent ⭐
│   └── tools.py           # Tool system & selection
├── utils/
│   └── retrieval.py       # Semantic search
└── scripts/
    └── db.py              # Supabase client

page.py                    # Interactive UI with steps
test_agentic.py           # Verification tests
AGENTIC_SETUP.md          # Full documentation
```

### Tool Flow
```
User Query
    ↓
Query Analysis (optional in GraphAgent)
    ↓
Tool Selection (heuristic or LLM-based)
    ↓
Execute Tools in Order:
    • Retrieval (always)
    • Summarization (conditional)
    • [Future tools...]
    ↓
Aggregate Tool Outputs
    ↓
LLM Synthesis with Context
    ↓
Final Answer
```

### LangGraph Flow
```
analyze → retrieve → [conditional] → synthesize → END
                            ↓
                       summarize?
```

## 🔑 Key Classes

### `Tool` (Base)
- `name`: Unique identifier
- `description`: For tool selection
- `input_model`: Pydantic schema
- `metadata`: Category, cost, latency
- `run()`: Execute logic

### `Agent`
- `__init__(llm_factory, tools, strategy)`
- `list_tools()`: Get available tools
- `answer(query, top_k, include_metrics)`

### `GraphAgent` ⭐
- `__init__()`: Creates StateGraph
- `answer(query, thread_id)`: Execute with memory
- `get_state(thread_id)`: Inspect state
- `visualize()`: Mermaid diagram

## 📊 Response Structure

### Enhanced Agent Response
```python
{
    "query": "...",
    "answer": "...",
    "contexts": [...],           # Retrieved verses
    "tools_used": ["retrieval", "summarize_contexts"],
    "tool_outputs": {
        "retrieval": {
            "status": "success",
            "count": 5,
            "execution_time_ms": 245
        },
        "summarize_contexts": {
            "status": "success",
            "summary": "...",
            "execution_time_ms": 892
        }
    },
    "metrics": {                 # If include_metrics=True
        "total_execution_time_ms": 1850,
        "context_count": 5,
        "tool_count": 2
    },
    "errors": []                 # If any errors occurred
}
```

## 🎯 Best Practices

### 1. Use GraphAgent for Production
```python
# Better state management and memory
from app.services.graph_agent import GraphAgent
agent = GraphAgent()
```

### 2. Always Include Metrics in Dev
```python
result = agent.answer(query, include_metrics=True)
# Debug timing and performance
```

### 3. Handle Errors Gracefully
```python
result = agent.answer(query)
if "errors" in result:
    print("Issues occurred:", result["errors"])
```

### 4. Use Thread IDs for Users
```python
agent = GraphAgent()
result = agent.answer(query, thread_id=f"user_{user_id}")
# Maintains conversation context
```

## 🧪 Testing

```bash
# Run verification tests
python test_agentic.py

# Run specific component
python -c "from app.services.agent import Agent; print(Agent().list_tools())"
```

## 📝 Customization

### Add a New Tool
1. Create tool class inheriting from `Tool`
2. Define `input_model` with Pydantic
3. Implement `run()` method
4. Add to agent: `Agent(tools=[..., MyTool()])`

### Change Tool Selection
```python
# LLM-based (intelligent)
agent = Agent(tool_selection_strategy="llm")

# Heuristic (fast, default)
agent = Agent(tool_selection_strategy="heuristic")
```

### Modify UI Steps
Edit `display_execution_steps()` in `page.py` to:
- Add new step displays
- Change visual indicators
- Customize metrics shown

## 🚀 Deployment Tips

1. **Environment Variables** - Ensure all API keys set
2. **Thread Management** - Use unique thread_ids per user
3. **Caching** - Consider Redis for checkpoints
4. **Monitoring** - Log metrics to analytics platform
5. **Rate Limiting** - Add per-user query limits

## 📚 Learn More

- **Full Architecture**: `AGENTIC_SETUP.md`
- **Usage Examples**: `README.md`
- **Implementation Details**: Source code comments
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/

## 🎉 Summary

You now have:
- ✅ Production-grade agentic RAG
- ✅ Interactive step visualization
- ✅ Extensible tool system
- ✅ Conversation memory
- ✅ Comprehensive metrics
- ✅ Type-safe with Pydantic
- ✅ Industry-standard patterns

**Start exploring:**
```bash
streamlit run page.py
```

---
*Made with ❤️ for the Muslim community*
