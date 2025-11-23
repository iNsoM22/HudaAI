"""Quick verification test for the agentic setup.

Run this to verify all components are working:
    python -m pytest test_agentic.py -v
    
Or run directly:
    python test_agentic.py
"""

def test_tool_system():
    """Test tool abstractions and metadata."""
    from app.services import tools as toollib
    from pydantic import BaseModel, Field
    
    # Test base tool
    class TestInput(BaseModel):
        value: str = Field(..., description="Test value")
    
    class TestTool(toollib.Tool):
        name = "test_tool"
        description = "A test tool"
        input_model = TestInput
        
        def run(self, **kwargs):
            data = self.validate(**kwargs)
            return {"result": data.value.upper()}
    
    tool = TestTool()
    assert tool.name == "test_tool"
    result = tool.run(value="hello")
    assert result["result"] == "HELLO"
    print("✅ Tool system working")


def test_enhanced_agent():
    """Test enhanced agent with metrics."""
    from app.services.agent import Agent
    from unittest.mock import Mock, MagicMock
    
    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test answer"
    mock_llm.invoke.return_value = mock_response
    
    # Mock retrieval
    import app.utils.retrieval as retrieval
    original_search = retrieval.semantic_search
    
    def mock_search(query, top_k=5):
        return (
            [{"chunk_key": "test", "surah_id": 1, "verse_range": "1-2", "similarity": 0.9, "text_english": "Test verse"}],
            [{"chunk_key": "test", "surah_id": 1, "verse_range": "1-2", "similarity": 0.9, "text_english": "Test verse"}]
        )
    
    retrieval.semantic_search = mock_search
    
    try:
        agent = Agent(llm_factory=lambda: mock_llm)
        result = agent.answer("test query", include_metrics=True)
        
        assert "answer" in result
        assert "metrics" in result
        assert "total_execution_time_ms" in result["metrics"]
        assert "tools_used" in result
        print("✅ Enhanced agent working")
    finally:
        retrieval.semantic_search = original_search


def test_tool_selection():
    """Test heuristic tool selection."""
    from app.services import tools as toollib
    
    tools = toollib.default_tools(lambda: None)
    
    # Test retrieval only
    selected = toollib.pick_tools_heuristic("What is patience?", tools)
    assert len(selected) == 1
    assert selected[0].name == "retrieval"
    
    # Test retrieval + summarization
    selected = toollib.pick_tools_heuristic("Give me a summary of verses about patience", tools)
    assert len(selected) == 2
    assert selected[0].name == "retrieval"
    assert selected[1].name == "summarize_contexts"
    
    print("✅ Tool selection working")


def test_tool_metadata():
    """Test tool metadata and serialization."""
    from app.services import tools as toollib
    
    tool = toollib.RetrievalTool()
    metadata = tool.to_dict()
    
    assert metadata["name"] == "retrieval"
    assert metadata["category"] == "retrieval"
    assert "parameters" in metadata
    print("✅ Tool metadata working")


def test_graph_agent_import():
    """Test that graph agent can be imported."""
    try:
        from app.services.graph_agent import GraphAgent, AgentState
        print("✅ LangGraph agent imports working")
    except ImportError as e:
        print(f"⚠️  LangGraph dependencies not fully installed: {e}")
        print("   This is optional - traditional agent still works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 Testing Agentic RAG Setup")
    print("="*60 + "\n")
    
    tests = [
        ("Tool System", test_tool_system),
        ("Enhanced Agent", test_enhanced_agent),
        ("Tool Selection", test_tool_selection),
        ("Tool Metadata", test_tool_metadata),
        ("Graph Agent Import", test_graph_agent_import),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Testing {test_name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All tests passed! Your agentic setup is working correctly.")
        print("\nNext steps:")
        print("  1. Run: streamlit run page.py")
        print("  2. Try queries with 'summary' keyword")
        print("  3. Check execution steps in the UI")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)
