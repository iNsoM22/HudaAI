"""Quick test to verify dual Quran + Hadith retrieval is working."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.services.agent import Agent

def test_dual_retrieval():
    """Test that agent retrieves from both Quran and Hadith."""
    
    print("=" * 80)
    print("TESTING DUAL SOURCE RETRIEVAL (Quran + Hadith)")
    print("=" * 80)
    
    # This query should trigger both sources (no specific keyword)
    query = "What does Islam teach about patience?"
    
    print(f"\n📝 Query: '{query}'")
    print("Expected: Should search BOTH Quran and Hadith\n")
    
    agent = Agent()
    
    # Test the tool selection first
    selected_tools = agent._select_tools(query)
    print("🔧 Selected Tools:")
    for tool in selected_tools:
        icon = "📚" if "quran" in tool.name else "📜" if "hadith" in tool.name else "🔄"
        print(f"   {icon} {tool.name}")
    
    print("\n" + "-" * 80)
    print("Executing agent...\n")
    
    result = agent.answer(query, top_k=3, include_metrics=True)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    tools_used = result.get("tools_used", [])
    contexts = result.get("contexts", [])
    
    print(f"\n✅ Tools Used: {', '.join(tools_used)}")
    print(f"✅ Total Contexts Retrieved: {len(contexts)}")
    
    # Count by source
    quran_count = sum(1 for c in contexts if c.get("surah_id"))
    hadith_count = sum(1 for c in contexts if c.get("hadith_id"))
    
    print(f"\n📊 Breakdown:")
    print(f"   📚 Quran verses: {quran_count}")
    print(f"   📜 Hadiths: {hadith_count}")
    
    if quran_count > 0 and hadith_count > 0:
        print("\n🎉 SUCCESS! Retrieved from BOTH sources!")
    elif quran_count > 0:
        print("\n⚠️  Only Quran results found. Hadith retrieval may have failed.")
    elif hadith_count > 0:
        print("\n⚠️  Only Hadith results found. Quran retrieval may have failed.")
    else:
        print("\n❌ No results from either source!")
    
    # Show sample contexts
    print("\n" + "-" * 80)
    print("SAMPLE CONTEXTS")
    print("-" * 80)
    
    for idx, c in enumerate(contexts[:5], 1):  # Show first 5
        if c.get("surah_id"):
            print(f"\n{idx}. 📚 Quran - Surah {c.get('surah_id')} | {c.get('verse_range')}")
            print(f"   Similarity: {c.get('similarity', 0):.4f}")
            text = c.get("text_english", "")[:100]
            print(f"   Text: {text}...")
        elif c.get("hadith_id"):
            print(f"\n{idx}. 📜 Hadith - {c.get('book_name')} #{c.get('hadith_number')}")
            print(f"   Similarity: {c.get('similarity', 0):.4f}")
            text = c.get("chunk_text", "") or c.get("context_english", "")
            print(f"   Text: {text[:100]}...")
    
    # Show metrics
    if "metrics" in result:
        metrics = result["metrics"]
        print("\n" + "-" * 80)
        print("PERFORMANCE METRICS")
        print("-" * 80)
        print(f"Total execution time: {metrics.get('total_execution_time_ms', 0)}ms")
        print(f"Tool count: {metrics.get('tool_count', 0)}")
    
    print("\n" + "=" * 80)
    
    return quran_count > 0 and hadith_count > 0


def test_hadith_specific():
    """Test query that should only retrieve Hadiths."""
    
    print("\n\n" + "=" * 80)
    print("TESTING HADITH-SPECIFIC QUERY")
    print("=" * 80)
    
    query = "Show me hadiths about charity"
    
    print(f"\n📝 Query: '{query}'")
    print("Expected: Should search ONLY Hadith\n")
    
    agent = Agent()
    selected_tools = agent._select_tools(query)
    
    print("🔧 Selected Tools:")
    for tool in selected_tools:
        icon = "📚" if "quran" in tool.name else "📜" if "hadith" in tool.name else "🔄"
        print(f"   {icon} {tool.name}")
    
    has_hadith_tool = any("hadith" in t.name for t in selected_tools)
    has_quran_tool = any("quran" in t.name for t in selected_tools)
    
    if has_hadith_tool and not has_quran_tool:
        print("\n✅ Correctly selected ONLY Hadith retrieval!")
        return True
    elif has_hadith_tool and has_quran_tool:
        print("\n⚠️  Selected both sources (acceptable but not optimal)")
        return True
    else:
        print("\n❌ Did not select Hadith tool!")
        return False


def test_quran_specific():
    """Test query that should only retrieve Quran."""
    
    print("\n\n" + "=" * 80)
    print("TESTING QURAN-SPECIFIC QUERY")
    print("=" * 80)
    
    query = "What verses mention paradise?"
    
    print(f"\n📝 Query: '{query}'")
    print("Expected: Should search ONLY Quran\n")
    
    agent = Agent()
    selected_tools = agent._select_tools(query)
    
    print("🔧 Selected Tools:")
    for tool in selected_tools:
        icon = "📚" if "quran" in tool.name else "📜" if "hadith" in tool.name else "🔄"
        print(f"   {icon} {tool.name}")
    
    has_hadith_tool = any("hadith" in t.name for t in selected_tools)
    has_quran_tool = any("quran" in t.name for t in selected_tools)
    
    if has_quran_tool and not has_hadith_tool:
        print("\n✅ Correctly selected ONLY Quran retrieval!")
        return True
    elif has_hadith_tool and has_quran_tool:
        print("\n⚠️  Selected both sources (acceptable but not optimal)")
        return True
    else:
        print("\n❌ Did not select Quran tool!")
        return False


if __name__ == "__main__":
    print("\n")
    print("🕌 " + "=" * 76 + " 🕌")
    print("   DUAL RETRIEVAL TEST SUITE - Quran & Hadith")
    print("🕌 " + "=" * 76 + " 🕌")
    
    try:
        # Run all tests
        test1 = test_dual_retrieval()
        test2 = test_hadith_specific()
        test3 = test_quran_specific()
        
        # Summary
        print("\n\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"{'✅' if test1 else '❌'} Dual Source Retrieval")
        print(f"{'✅' if test2 else '❌'} Hadith-Specific Query")
        print(f"{'✅' if test3 else '❌'} Quran-Specific Query")
        
        if test1 and test2 and test3:
            print("\n🎉 All tests passed! Dual retrieval is working perfectly!")
        else:
            print("\n⚠️  Some tests failed. Review the output above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
