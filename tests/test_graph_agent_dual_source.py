"""Test script for GraphAgent with dual-source retrieval (Quran + Hadith)"""

from app.services.graph_agent import GraphAgent
from app.utils.retrieval import get_expanded_context

def test_dual_source_retrieval():
    """Test retrieval from both Quran and Hadith sources."""
    print("=" * 80)
    print("TEST 1: Dual-Source Retrieval (Quran + Hadith)")
    print("=" * 80)
    
    agent = GraphAgent()
    
    # Test with a general Islamic query that should retrieve from both sources
    query = "What does Islam teach about patience and perseverance?"
    
    print(f"\n📝 Query: {query}\n")
    
    result = agent.answer(query=query)
    
    # Display analysis results
    analysis = result.get("tool_outputs", {}).get("analysis", {})
    print("🔍 Query Analysis:")
    print(f"  - Intent: {analysis.get('intent')}")
    print(f"  - Search Quran: {analysis.get('search_quran')}")
    print(f"  - Search Hadith: {analysis.get('search_hadith')}")
    print(f"  - Top K: {analysis.get('top_k')}")
    print(f"  - Needs Summary: {analysis.get('needs_summary')}")
    print(f"  - Reasoning: {analysis.get('reasoning')}")
    
    # Display retrieval results
    retrieval_info = result.get("tool_outputs", {}).get("retrieval", {})
    print(f"\n📚 Retrieval Results:")
    if "quran" in retrieval_info:
        print(f"  - Quran verses retrieved: {retrieval_info['quran']['count']}")
    if "hadith" in retrieval_info:
        print(f"  - Hadiths retrieved: {retrieval_info['hadith']['count']}")
    print(f"  - Total items: {result.get('tool_outputs', {}).get('total_retrieved', 0)}")
    
    # Display sample contexts
    contexts = result.get("contexts", [])
    print(f"\n📖 Sample Contexts (first 2):")
    for idx, ctx in enumerate(contexts[:2], 1):
        if ctx.get("surah_id"):
            print(f"\n  {idx}. 📚 QURAN - Surah {ctx.get('surah_id')} | Verses {ctx.get('verse_range')}")
            print(f"     Similarity: {ctx.get('similarity', 0):.4f}")
            print(f"     Text: {ctx.get('text_english', '')[:150]}...")
        elif ctx.get("hadith_id"):
            print(f"\n  {idx}. 📜 HADITH - {ctx.get('book_name')} #{ctx.get('hadith_number')}")
            print(f"     Similarity: {ctx.get('similarity', 0):.4f}")
            print(f"     Text: {ctx.get('context_english', '')[:150]}...")
    
    # Display answer
    print(f"\n💬 Generated Answer:")
    print(f"{result.get('answer', '')[:500]}...")
    
    return result


def test_quran_only_retrieval():
    """Test retrieval from Quran only."""
    print("\n" + "=" * 80)
    print("TEST 2: Quran-Only Retrieval")
    print("=" * 80)
    
    agent = GraphAgent()
    
    query = "Show me verses from the Quran about Paradise"
    
    print(f"\n📝 Query: {query}\n")
    
    result = agent.answer(query=query)
    
    analysis = result.get("tool_outputs", {}).get("analysis", {})
    print("🔍 Query Analysis:")
    print(f"  - Search Quran: {analysis.get('search_quran')}")
    print(f"  - Search Hadith: {analysis.get('search_hadith')}")
    
    retrieval_info = result.get("tool_outputs", {}).get("retrieval", {})
    print(f"\n📚 Retrieval Results:")
    print(f"  - Sources used: {list(retrieval_info.keys())}")
    
    return result


def test_hadith_only_retrieval():
    """Test retrieval from Hadith only."""
    print("\n" + "=" * 80)
    print("TEST 3: Hadith-Only Retrieval")
    print("=" * 80)
    
    agent = GraphAgent()
    
    query = "What hadiths mention charity and helping the poor?"
    
    print(f"\n📝 Query: {query}\n")
    
    result = agent.answer(query=query)
    
    analysis = result.get("tool_outputs", {}).get("analysis", {})
    print("🔍 Query Analysis:")
    print(f"  - Search Quran: {analysis.get('search_quran')}")
    print(f"  - Search Hadith: {analysis.get('search_hadith')}")
    
    retrieval_info = result.get("tool_outputs", {}).get("retrieval", {})
    print(f"\n📚 Retrieval Results:")
    print(f"  - Sources used: {list(retrieval_info.keys())}")
    
    return result


def test_context_expansion():
    """Test context expansion for both Quran and Hadith."""
    print("\n" + "=" * 80)
    print("TEST 4: Context Expansion")
    print("=" * 80)
    
    # Test Quran context expansion
    print("\n📚 Testing Quran Context Expansion:")
    print("   Expanding around chunk_key: '2_1_5'")
    
    quran_expansion = get_expanded_context(
        source_type="quran",
        identifier="2_1_5",
        context_window=2
    )
    
    if quran_expansion.get("status") == "success":
        print(f"   ✅ Success!")
        print(f"   - Surah: {quran_expansion.get('surah_id')}")
        print(f"   - Expanded range: {quran_expansion.get('expanded_range')}")
        print(f"   - Contexts retrieved: {quran_expansion.get('count')}")
    else:
        print(f"   ❌ Error: {quran_expansion.get('message')}")
    
    # Test Hadith context expansion (use a sample hadith_id)
    print("\n📜 Testing Hadith Context Expansion:")
    print("   Expanding around hadith_id: 1 (example)")
    
    hadith_expansion = get_expanded_context(
        source_type="hadith",
        identifier="1",
        context_window=2
    )
    
    if hadith_expansion.get("status") == "success":
        print(f"   ✅ Success!")
        print(f"   - Book: {hadith_expansion.get('book_name')}")
        print(f"   - Expanded range: {hadith_expansion.get('expanded_range')}")
        print(f"   - Contexts retrieved: {hadith_expansion.get('count')}")
    else:
        print(f"   ❌ Error: {hadith_expansion.get('message')}")
    
    return quran_expansion, hadith_expansion


if __name__ == "__main__":
    print("\n🚀 Testing GraphAgent with Dual-Source Retrieval\n")
    
    try:
        # Run all tests
        result1 = test_dual_source_retrieval()
        result2 = test_quran_only_retrieval()
        result3 = test_hadith_only_retrieval()
        result4 = test_context_expansion()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
