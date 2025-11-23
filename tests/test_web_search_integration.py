"""Test script for GraphAgent with web search integration"""

from app.services.graph_agent import GraphAgent

def test_web_search_trigger():
    """Test that web search is triggered for contemporary queries"""
    print("=" * 80)
    print("TEST 1: Web Search Trigger Detection")
    print("=" * 80)
    
    agent = GraphAgent()
    
    # Queries that should trigger web search
    web_search_queries = [
        "What do modern Islamic scholars say about cryptocurrency?",
        "How are Muslims responding to climate change?",
        "What is the ruling on COVID-19 vaccines in Islam?",
    ]
    
    for query in web_search_queries:
        print(f"\n📝 Query: {query}")
        
        result = agent.answer(query=query, top_k=3, include_metrics=True)
        
        analysis = result.get("tool_outputs", {}).get("analysis", {})
        needs_web_search = analysis.get("needs_web_search", False)
        
        print(f"   🔍 Needs web search: {needs_web_search}")
        print(f"   📊 Tools used: {result.get('tools_used')}")
        
        if "web_search" in result.get("tool_outputs", {}):
            web_result = result["tool_outputs"]["web_search"]
            if web_result.get("status") == "success":
                print(f"   ✅ Web search executed: {web_result.get('count', 0)} results")
            else:
                print(f"   ❌ Web search failed: {web_result.get('error')}")
        
        print(f"   ⏱️  Total time: {result.get('metrics', {}).get('total_execution_time_ms', 0)}ms")


def test_no_web_search_for_traditional():
    """Test that traditional Islamic queries don't trigger web search"""
    print("\n" + "=" * 80)
    print("TEST 2: Traditional Queries (No Web Search)")
    print("=" * 80)
    
    agent = GraphAgent()
    
    # Traditional queries that should NOT trigger web search
    traditional_queries = [
        "What does Islam teach about patience?",
        "Show me verses about charity",
        "What hadiths mention prayer?",
    ]
    
    for query in traditional_queries:
        print(f"\n📝 Query: {query}")
        
        result = agent.answer(query=query, top_k=3, include_metrics=True)
        
        analysis = result.get("tool_outputs", {}).get("analysis", {})
        needs_web_search = analysis.get("needs_web_search", False)
        
        print(f"   🔍 Needs web search: {needs_web_search}")
        print(f"   📊 Tools used: {result.get('tools_used')}")
        
        if needs_web_search:
            print("   ⚠️  WARNING: Web search triggered for traditional query!")
        else:
            print("   ✅ Correctly avoided web search")


def test_web_search_streaming():
    """Test web search in streaming mode"""
    print("\n" + "=" * 80)
    print("TEST 3: Web Search in Streaming Mode")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "What is the Islamic perspective on artificial intelligence?"
    
    print(f"\n📝 Query: {query}\n")
    
    web_search_executed = False
    web_results_count = 0
    
    for update in agent.answer_stream(query=query, top_k=3):
        event_type = update.get("type")
        
        if event_type == "status":
            print(f"🔄 {update.get('message')}")
            if update.get("tools_selected"):
                tools = update["tools_selected"]
                print(f"   Tools: {tools}")
                if "web_search" in str(tools):
                    print("   🌐 Web search will be executed!")
        
        elif event_type == "tool_start":
            tool_name = update.get("tool_name")
            if tool_name == "web_search":
                print(f"\n🌐 Starting web search...")
                web_search_executed = True
        
        elif event_type == "tool_complete":
            tool_name = update.get("tool_name")
            if tool_name == "web_search":
                output = update.get("output", {})
                if output.get("status") == "success":
                    web_results_count = output.get("count", 0)
                    print(f"✅ Web search completed: {web_results_count} results")
                    
                    # Show sample results
                    results = output.get("results", [])
                    for idx, result in enumerate(results[:2], 1):
                        print(f"\n   {idx}. {result['title']}")
                        print(f"      Source: {result['source']}")
                        print(f"      {result['snippet'][:100]}...")
                else:
                    print(f"❌ Web search failed: {output.get('error')}")
        
        elif event_type == "complete":
            print(f"\n✅ Query complete!")
            print(f"   Answer length: {len(update.get('answer', ''))} characters")
            break
    
    if web_search_executed:
        print(f"\n✅ Web search integration working! Found {web_results_count} results")
    else:
        print("\n⚠️  Web search was not triggered")


def test_web_search_api():
    """Test web search API directly"""
    print("\n" + "=" * 80)
    print("TEST 4: Direct Web Search API Test")
    print("=" * 80)
    
    from app.services.graph_agent import web_search
    
    query = "Islamic perspective on space exploration"
    print(f"\n📝 Search query: {query}\n")
    
    try:
        result = web_search.invoke({"query": query, "num_results": 3})
        
        if result.get("status") == "success":
            print(f"✅ API call successful!")
            print(f"   Results found: {result.get('count', 0)}")
            print(f"   Total results available: {result.get('search_metadata', {}).get('total_results', 'Unknown')}")
            print(f"   Search time: {result.get('search_metadata', {}).get('time_taken', 'Unknown')}")
            
            print(f"\n📚 Search Results:")
            for idx, res in enumerate(result.get("results", []), 1):
                print(f"\n{idx}. **{res['title']}**")
                print(f"   URL: {res['link']}")
                print(f"   Source: {res['source']}")
                print(f"   Snippet: {res['snippet'][:150]}...")
        else:
            print(f"❌ API call failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")


def test_combined_context():
    """Test that web search results are integrated with Quran/Hadith"""
    print("\n" + "=" * 80)
    print("TEST 5: Combined Context (Quran + Hadith + Web)")
    print("=" * 80)
    
    agent = GraphAgent()
    query = "What does Islam say about environmental protection and climate change?"
    
    print(f"\n📝 Query: {query}\n")
    
    result = agent.answer(query=query, top_k=5, include_metrics=True)
    
    # Check all sources used
    contexts = result.get("contexts", [])
    tool_outputs = result.get("tool_outputs", {})
    
    quran_count = len([c for c in contexts if c.get("surah_id")])
    hadith_count = len([c for c in contexts if c.get("hadith_id")])
    web_count = 0
    
    if "web_search" in tool_outputs:
        web_result = tool_outputs["web_search"]
        if web_result.get("status") == "success":
            web_count = web_result.get("count", 0)
    
    print(f"📊 Context Sources:")
    print(f"   📚 Quran verses: {quran_count}")
    print(f"   📜 Hadiths: {hadith_count}")
    print(f"   🌐 Web results: {web_count}")
    print(f"   📝 Total contexts: {len(contexts)}")
    
    # Show sample answer
    answer = result.get("answer", "")
    print(f"\n💬 Answer preview (first 300 chars):")
    print(f"   {answer[:300]}...")
    
    if web_count > 0 and (quran_count > 0 or hadith_count > 0):
        print("\n✅ Successfully integrated multiple sources!")
    elif web_count > 0:
        print("\n⚠️  Web search used but no Quran/Hadith context")
    else:
        print("\n⚠️  Web search not triggered")


if __name__ == "__main__":
    print("\n🚀 Testing Web Search Integration with GraphAgent\n")
    
    try:
        # Test 4 first (API test) to verify Serper is working
        test_web_search_api()
        
        # Then run full integration tests
        test_web_search_trigger()
        test_no_web_search_for_traditional()
        test_web_search_streaming()
        test_combined_context()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 80)
        print("\n💡 Web search is now integrated with dual-source retrieval")
        print("   - Traditional queries: Quran + Hadith only")
        print("   - Contemporary queries: Quran + Hadith + Web")
        print("   - Intelligent routing based on query analysis")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
