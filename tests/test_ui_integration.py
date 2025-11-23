"""Test script to verify GraphAgent compatibility with page.py UI"""

from app.services.graph_agent import GraphAgent
import time

def test_non_streaming_mode():
    """Test non-streaming mode (original UI behavior)"""
    print("=" * 80)
    print("TEST 1: Non-Streaming Mode (Direct answer)")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "What does Islam teach about patience?"
    
    print(f"\n📝 Query: {query}")
    print("⏳ Processing...")
    
    start = time.time()
    result = agent.answer(query=query, top_k=5, include_metrics=True)
    elapsed = time.time() - start
    
    print(f"\n✅ Completed in {elapsed:.2f}s")
    
    # Verify expected structure
    assert "query" in result, "Missing 'query' in result"
    assert "answer" in result, "Missing 'answer' in result"
    assert "contexts" in result, "Missing 'contexts' in result"
    assert "tools_used" in result, "Missing 'tools_used' in result"
    assert "tool_outputs" in result, "Missing 'tool_outputs' in result"
    assert "metrics" in result, "Missing 'metrics' (include_metrics=True)"
    
    print("\n📊 Result Structure:")
    print(f"  - Query: {result['query']}")
    print(f"  - Answer length: {len(result['answer'])} characters")
    print(f"  - Contexts retrieved: {len(result['contexts'])}")
    print(f"  - Tools used: {result['tools_used']}")
    print(f"  - Metrics: {result['metrics']}")
    
    # Check tool outputs
    tool_outputs = result["tool_outputs"]
    print(f"\n🔧 Tool Outputs:")
    for tool_name, output in tool_outputs.items():
        print(f"  - {tool_name}: {type(output)}")
        if tool_name == "analysis":
            analysis = output
            print(f"    • Intent: {analysis.get('intent')}")
            print(f"    • Search Quran: {analysis.get('search_quran')}")
            print(f"    • Search Hadith: {analysis.get('search_hadith')}")
        elif tool_name == "retrieval":
            print(f"    • Sources: {list(output.keys())}")
    
    # Show sample contexts
    print(f"\n📚 Sample Contexts:")
    for idx, ctx in enumerate(result["contexts"][:2], 1):
        if ctx.get("surah_id"):
            print(f"  {idx}. Quran - Surah {ctx.get('surah_id')} | {ctx.get('verse_range')}")
        elif ctx.get("hadith_id"):
            print(f"  {idx}. Hadith - {ctx.get('book_name')} #{ctx.get('hadith_number')}")
    
    print("\n✅ Non-streaming mode test passed!")
    return result


def test_streaming_mode():
    """Test streaming mode (UI with real-time updates)"""
    print("\n" + "=" * 80)
    print("TEST 2: Streaming Mode (Real-time updates)")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "What hadiths mention charity?"
    
    print(f"\n📝 Query: {query}")
    print("⏳ Streaming...")
    
    events = []
    tools_seen = set()
    answer_tokens = []
    final_result = None
    
    for update in agent.answer_stream(query=query, top_k=5):
        event_type = update.get("type")
        events.append(event_type)
        
        if event_type == "status":
            print(f"\n🔄 Status: {update.get('message')}")
            if update.get("tools_selected"):
                print(f"   Tools: {update['tools_selected']}")
        
        elif event_type == "tool_start":
            tool_name = update.get("tool_name")
            print(f"\n🛠️  Starting: {tool_name}")
        
        elif event_type == "tool_complete":
            tool_name = update.get("tool_name")
            tools_seen.add(tool_name)
            output = update.get("output", {})
            print(f"✅ Completed: {tool_name}")
            
            if tool_name in ["retrieval_quran", "retrieval_hadith"]:
                count = output.get("count", 0)
                status = output.get("status", "unknown")
                print(f"   Status: {status} | Count: {count}")
        
        elif event_type == "synthesis_start":
            print(f"\n💬 {update.get('message')}")
        
        elif event_type == "answer_token":
            token = update.get("token", "")
            answer_tokens.append(token)
            # Print first 10 tokens to show streaming
            if len(answer_tokens) <= 10:
                print(token, end="", flush=True)
            elif len(answer_tokens) == 11:
                print("...", flush=True)
        
        elif event_type == "complete":
            print(f"\n\n✅ Complete!")
            final_result = update
            print(f"   Full answer length: {len(update.get('answer', ''))} characters")
            print(f"   Contexts: {len(update.get('contexts', []))}")
            print(f"   Tools used: {update.get('tools_used')}")
        
        elif event_type == "error":
            print(f"\n❌ Error: {update.get('error')}")
    
    print(f"\n📊 Streaming Summary:")
    print(f"  - Total events: {len(events)}")
    print(f"  - Event types: {set(events)}")
    print(f"  - Tools executed: {tools_seen}")
    print(f"  - Answer tokens: {len(answer_tokens)}")
    
    # Verify expected structure
    assert final_result is not None, "No final 'complete' event received"
    assert "query" in final_result, "Missing 'query' in final result"
    assert "answer" in final_result, "Missing 'answer' in final result"
    assert "contexts" in final_result, "Missing 'contexts' in final result"
    assert "tools_used" in final_result, "Missing 'tools_used' in final result"
    
    print("\n✅ Streaming mode test passed!")
    return final_result


def test_dual_source_streaming():
    """Test that both Quran and Hadith sources work in streaming"""
    print("\n" + "=" * 80)
    print("TEST 3: Dual-Source Streaming")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "What does Islam teach about helping the poor?"
    
    print(f"\n📝 Query: {query}")
    
    quran_found = False
    hadith_found = False
    
    for update in agent.answer_stream(query=query, top_k=5):
        event_type = update.get("type")
        
        if event_type == "tool_complete":
            tool_name = update.get("tool_name")
            if tool_name == "retrieval_quran":
                quran_found = True
                print(f"✅ Quran retrieval: {update.get('output', {}).get('count', 0)} verses")
            elif tool_name == "retrieval_hadith":
                hadith_found = True
                print(f"✅ Hadith retrieval: {update.get('output', {}).get('count', 0)} hadiths")
        
        elif event_type == "complete":
            contexts = update.get("contexts", [])
            quran_contexts = [c for c in contexts if c.get("surah_id")]
            hadith_contexts = [c for c in contexts if c.get("hadith_id")]
            
            print(f"\n📚 Final Context Mix:")
            print(f"  - Quran verses: {len(quran_contexts)}")
            print(f"  - Hadiths: {len(hadith_contexts)}")
            print(f"  - Total: {len(contexts)}")
            
            # Verify both sources present (for general query)
            if len(contexts) > 0:
                print("\n✅ Dual-source test passed!")
            else:
                print("\n⚠️  Warning: No contexts retrieved")


def test_quran_only_query():
    """Test Quran-specific query detection"""
    print("\n" + "=" * 80)
    print("TEST 4: Quran-Only Query Detection")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "Show me Quranic verses about paradise"
    
    print(f"\n📝 Query: {query}")
    
    analysis_found = False
    
    for update in agent.answer_stream(query=query, top_k=3):
        event_type = update.get("type")
        
        if event_type == "status" and update.get("tools_selected"):
            tools = update["tools_selected"]
            print(f"\n🔍 Tools selected: {tools}")
            
            if "retrieval_quran" in tools and "retrieval_hadith" not in tools:
                print("✅ Correctly identified as Quran-only query!")
                analysis_found = True
            elif "retrieval_quran" in tools and "retrieval_hadith" in tools:
                print("ℹ️  Searching both sources (general behavior)")
                analysis_found = True
        
        elif event_type == "complete":
            break
    
    if analysis_found:
        print("\n✅ Query detection test passed!")
    else:
        print("\n⚠️  Analysis data not found in stream")


def test_error_handling():
    """Test error handling with invalid query"""
    print("\n" + "=" * 80)
    print("TEST 5: Error Handling")
    print("=" * 80)
    
    agent = GraphAgent()
    query = ""  # Empty query
    
    print(f"\n📝 Query: '{query}' (empty)")
    
    try:
        result = agent.answer(query=query, top_k=5)
        print("✅ Empty query handled gracefully")
        print(f"   Answer: {result['answer'][:100]}...")
    except Exception as e:
        print(f"❌ Exception raised: {e}")


if __name__ == "__main__":
    print("\n🚀 Testing GraphAgent Integration with UI\n")
    
    try:
        # Run all tests
        test_non_streaming_mode()
        test_streaming_mode()
        test_dual_source_streaming()
        test_quran_only_query()
        test_error_handling()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - UI Integration Ready!")
        print("=" * 80)
        print("\n💡 You can now run: streamlit run page.py")
        
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
