# Dual Source Retrieval: Quran & Hadith

## Overview
HudaAI now intelligently searches both **Quranic verses** and **Hadith collections** to provide comprehensive Islamic knowledge answers.

## What Changed

### 🔧 New Tools Added

1. **QuranRetrievalTool** (`retrieval_quran`)
   - Searches Quranic verses using semantic embeddings
   - Returns verses with surah, verse range, and similarity scores

2. **HadithRetrievalTool** (`retrieval_hadith`)
   - Searches Hadith collections (Bukhari, Muslim, etc.)
   - Returns hadiths with book name, hadith number, and similarity
   - Supports optional book filtering and similarity threshold

3. **Enhanced SummarizeContextsTool**
   - Now handles both Quran verses and Hadiths
   - Automatically detects source type and adjusts formatting

### 🤖 Intelligent Source Selection

The agent automatically decides which sources to search based on your query:

**Hadith-Only Search** (when query contains):
- "hadith", "sunnah", "prophet said", "narrated"
- "sahih", "bukhari", "muslim", book names
- "prophet muhammad", "messenger said"

**Quran-Only Search** (when query contains):
- "quran", "verse", "surah", "ayah"
- "revelation", "allah said", "quranic"

**Both Sources** (default for general questions):
- General Islamic questions
- Questions mentioning both sources
- Queries without specific source keywords

### 📊 Example Queries

```python
# Searches both sources
"What does Islam teach about patience?"

# Searches Hadith only
"Find hadiths about charity from Bukhari"

# Searches Quran only
"What verses mention paradise?"

# Explicitly searches both
"Compare Quran verses and hadiths about prayer"
```

## Technical Details

### Retrieval Functions

```python
# In app/utils/retrieval.py

def semantic_search_quran(query, top_k=5) -> (results, contexts)
def semantic_search_hadiths(query, top_k=5, book_filter=None, match_threshold=0.5) -> (results, contexts)
```

### Tool Configuration

```python
# Default tools now include both retrieval sources
default_tools = [
    QueryExpansionTool,      # Expands query for better results
    QuranRetrievalTool,      # Searches Quran
    HadithRetrievalTool,     # Searches Hadith
    SummarizeContextsTool    # Summarizes mixed content
]
```

### Context Format

**Quran Context:**
```python
{
    "chunk_key": "2_153_154",
    "surah_id": 2,
    "verse_range": "153 - 154",
    "text_english": "...",
    "text_uthmani": "...",
    "context_english": "...",
    "similarity": 0.87
}
```

**Hadith Context:**
```python
{
    "hadith_id": 123,
    "book_name": "Sahih Bukhari",
    "hadith_number": "5234",
    "chunk_text": "...",  # Matched portion
    "context_english": "...",  # Full hadith
    "context_arabic": "...",
    "similarity": 0.92
}
```

### Prompt Building

The `build_prompt()` function now:
- Detects source types in contexts
- Formats Quran verses and Hadiths differently
- Provides appropriate citation examples
- Adjusts instructions based on sources used

## UI Updates

### Display Changes

1. **Title**: Now shows "Islamic Knowledge Explorer" instead of "Verse Explorer"
2. **Context Display**: Shows distinct formatting for:
   - 📚 Quran verses (with surah/verse info)
   - 📜 Hadiths (with book name and number)
3. **Execution Steps**: Separate steps for Quran and Hadith retrieval
4. **Streaming Updates**: Different icons and messages per source

### Visual Indicators

- 📚 = Quran content
- 📜 = Hadith content  
- 🌙 = Arabic text
- 📖 = Extended context
- 🎯 = Matched portion

## Performance Considerations

- **Query Expansion**: Still applied once, used for both sources
- **Deduplication**: Uses `chunk_key` for Quran, `hadith_id` for Hadith
- **Top K**: Applied per source (e.g., top 5 from each)
- **Combined Results**: Sorted by similarity across sources

## Testing

```python
# Test Hadith retrieval
from app.utils.retrieval import semantic_search_hadiths

results, contexts = semantic_search_hadiths(
    query="charity and kindness",
    top_k=5,
    book_filter=None,  # Search all books
    match_threshold=0.5
)

# Test Quran retrieval
from app.utils.retrieval import semantic_search_quran

results, contexts = semantic_search_quran(
    query="patience in adversity",
    top_k=5
)

# Test agent with both
from app.services.agent import Agent

agent = Agent()
result = agent.answer("What does Islam teach about patience?", top_k=5)
# Will search both sources and combine results
```

## Future Enhancements

- [ ] Add book-specific filtering UI
- [ ] Grade hadith by authenticity level
- [ ] Cross-reference Quran verses cited in Hadiths
- [ ] Add tafsir (commentary) as third source
- [ ] Support filtering by hadith narrator
- [ ] Add hadith collections metadata

## Migration Notes

### Backward Compatibility

- Old `retrieval` tool name still works via `RetrievalTool` (deprecated)
- Existing code using `semantic_search()` continues to work
- UI displays both old and new format contexts gracefully

### Breaking Changes

None - all changes are additive and backward compatible.

## Summary

✅ **Dual source retrieval** - Quran + Hadith  
✅ **Intelligent selection** - Automatic based on query  
✅ **Enhanced UI** - Clear source indicators  
✅ **Better answers** - More comprehensive coverage  
✅ **Backward compatible** - No breaking changes  

Now your Islamic knowledge assistant can provide answers grounded in both the Quran and authentic Hadith collections! 🕌
