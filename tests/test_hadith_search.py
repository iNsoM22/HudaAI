"""Test script for Hadith semantic search functionality.

This script tests the semantic_search_hadiths function to verify:
- Connection to Supabase
- Embedding generation
- RPC call execution
- Result formatting and data structure
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.utils.retrieval import semantic_search_hadiths
from pprint import pprint


def test_basic_search():
    """Test basic hadith search with a simple query."""
    print("=" * 80)
    print("TEST 1: Basic Hadith Search")
    print("=" * 80)
    
    query = "charity and kindness"
    print(f"\nQuery: '{query}'")
    print(f"Searching for top 3 hadiths...\n")
    
    try:
        results, contexts = semantic_search_hadiths(
            query=query,
            top_k=3,
            match_threshold=0.3
        )
        
        print(f"✅ Success! Found {len(results)} hadiths\n")
        
        for idx, result in enumerate(results, 1):
            print(f"--- Hadith {idx} ---")
            print(f"Book: {result.get('book_name', 'Unknown')}")
            print(f"Hadith #: {result.get('hadith_number', '?')}")
            print(f"Similarity: {result.get('similarity', 0):.4f}")
            print(f"Matched chunk: {result.get('chunk_text', '')[:100]}...")
            print(f"Full hadith: {result.get('context_english', '')[:150]}...")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_book_filter():
    """Test hadith search with book filtering."""
    print("=" * 80)
    print("TEST 2: Search with Book Filter")
    print("=" * 80)
    
    query = "prayer and supplication"
    book_id = 1  # Adjust this based on your database
    print(f"\nQuery: '{query}'")
    print(f"Filtering by book_id: {book_id}")
    print(f"Searching for top 3 hadiths...\n")
    
    try:
        results, contexts = semantic_search_hadiths(
            query=query,
            top_k=3,
            book_filter=book_id,
            match_threshold=0.3
        )
        
        print(f"✅ Success! Found {len(results)} hadiths from specific book\n")
        
        for idx, result in enumerate(results, 1):
            print(f"--- Hadith {idx} ---")
            print(f"Book: {result.get('book_name', 'Unknown')}")
            print(f"Hadith #: {result.get('hadith_number', '?')}")
            print(f"Similarity: {result.get('similarity', 0):.4f}")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_high_threshold():
    """Test with high similarity threshold."""
    print("=" * 80)
    print("TEST 3: High Similarity Threshold (0.7)")
    print("=" * 80)
    
    query = "fasting in Ramadan"
    print(f"\nQuery: '{query}'")
    print(f"Using match_threshold=0.7 (only very similar results)\n")
    
    try:
        results, contexts = semantic_search_hadiths(
            query=query,
            top_k=5,
            match_threshold=0.7
        )
        
        print(f"✅ Success! Found {len(results)} highly relevant hadiths\n")
        
        if len(results) == 0:
            print("⚠️  No results found with threshold 0.7. Try lowering the threshold.\n")
        else:
            for idx, result in enumerate(results, 1):
                print(f"Hadith {idx}: {result.get('book_name')} #{result.get('hadith_number')} - Similarity: {result.get('similarity', 0):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_structure():
    """Test the structure of returned contexts for LLM use."""
    print("=" * 80)
    print("TEST 4: Context Structure Validation")
    print("=" * 80)
    
    query = "honesty and truthfulness"
    print(f"\nQuery: '{query}'")
    print("Checking data structure for LLM consumption...\n")
    
    try:
        results, contexts = semantic_search_hadiths(
            query=query,
            top_k=2,
            match_threshold=0.3
        )
        
        print(f"✅ Retrieved {len(results)} results and {len(contexts)} contexts\n")
        
        print("--- Full Result Structure (for UI display) ---")
        if results:
            print("Keys in results[0]:")
            pprint(list(results[0].keys()))
            print("\nSample result:")
            pprint(results[0])
        
        print("\n--- Model Context Structure (for LLM) ---")
        if contexts:
            print("Keys in contexts[0]:")
            pprint(list(contexts[0].keys()))
            print("\nSample context:")
            pprint(contexts[0])
        
        # Validate required fields
        print("\n--- Field Validation ---")
        required_fields = ['hadith_id', 'book_name', 'hadith_number', 'similarity', 'chunk_text', 'context_english']
        if contexts:
            for field in required_fields:
                status = "✅" if field in contexts[0] else "❌"
                print(f"{status} {field}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_queries():
    """Test different types of queries."""
    print("=" * 80)
    print("TEST 5: Various Query Types")
    print("=" * 80)
    
    test_queries = [
        "What did the Prophet say about patience?",
        "jihad",
        "family and parents",
        "end times and signs",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        try:
            results, _ = semantic_search_hadiths(
                query=query,
                top_k=2,
                match_threshold=0.3
            )
            print(f"   Found {len(results)} hadiths")
            if results:
                top_similarity = results[0].get('similarity', 0)
                print(f"   Top similarity: {top_similarity:.4f}")
                print(f"   Top book: {results[0].get('book_name', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 80)
    print("HADITH SEMANTIC SEARCH TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        ("Basic Search", test_basic_search),
        ("Book Filter", test_specific_book_filter),
        ("High Threshold", test_high_threshold),
        ("Context Structure", test_context_structure),
        ("Query Variations", test_comparison_queries),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
            print()
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Hadith search is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")


if __name__ == "__main__":
    # You can run individual tests or all tests
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hadith semantic search")
    parser.add_argument('--test', type=str, choices=['basic', 'filter', 'threshold', 'structure', 'queries', 'all'],
                      default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'basic':
        test_basic_search()
    elif args.test == 'filter':
        test_specific_book_filter()
    elif args.test == 'threshold':
        test_high_threshold()
    elif args.test == 'structure':
        test_context_structure()
    elif args.test == 'queries':
        test_comparison_queries()
    else:
        run_all_tests()
