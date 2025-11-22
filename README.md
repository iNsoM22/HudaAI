---
title: HudaAI
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: page.py
pinned: false
---

# HudaAI

An intelligent Islamic assistant powered by AI that provides contextual answers from the Quran and Hadith using Retrieval-Augmented Generation (RAG) technology.

## 📖 Overview

HudaAI is a comprehensive Islamic knowledge system that combines:
- **Quranic Verses** (Arabic, English translations, and Uthmani script)
- **Hadith Collections** from authenticated sources
- **AI-Powered Search** using vector embeddings
- **Natural Language Processing** for context-aware responses

## ✨ Features

- 🔍 **Semantic Search**: Find relevant Quranic verses and Hadiths based on meaning, not just keywords
- 🌐 **Multi-language Support**: Access content in Arabic (Simple & Uthmani script) and English
- 💬 **Context-Aware Responses**: Get intelligent answers grounded in Islamic texts
- 📚 **Comprehensive Database**: Integration with authentic Quran and Hadith APIs
- ⚡ **Fast & Efficient**: Vector-based retrieval for quick response times
- 🔒 **Secure**: Built with Supabase for reliable data storage

## 🏗️ Project Structure

```
HudaAI/
├── app/
│   ├── scripts/
│   │   ├── db.py                # Supabase client management
│   │   ├── ingest_verses.py     # Ingest ayahs.csv into verse tables
│   │   └── supabase_sql.sql     # SQL schema & RPC setup
│   ├── services/
│   │   └── agent.py             # Agentic RAG orchestration
│   ├── utils/
│   │   ├── retrieval.py         # Embeddings, storage, semantic search
│   │   └── rag.py               # (Legacy) resume/doc chunks utilities
│   └── ui/
│       └── app.py               # Streamlit demo interface
├── notebooks/
│   └── Data_Preprocessing.ipynb  # Data collection and preprocessing
├── .env.example            # Environment variables template
├── .gitignore             # Git ignore rules
└── SRS.pdf                # Software Requirements Specification
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Supabase account
- HuggingFace API token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iNsoM22/HudaAI.git
   cd HudaAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Copy `.env.example` to `.env` and fill in your credentials:
   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   HUGGINGFACE_API_KEY=your_huggingface_api_token
   ```

5. **Set up Supabase Database (Verses RAG)**

   Open `app/scripts/supabase_sql.sql` in the Supabase SQL editor and run it. This will:
   - Enable `pgvector`
   - Create `verse_chunks` and `verse_embeddings` tables (dimension 384 for MiniLM)
   - Create `match_verses` RPC function for cosine similarity search
   
   Ensure you have the environment variables in `.env`:
   ```env
   SUPABASE_URL=...
   SUPABASE_KEY=...
   HUGGINGFACE_API_KEY=...
   GOOGLE_API_KEY=...          # For generation (Gemini model)
   GOOGLE_MODEL=gemini-1.5-flash
   EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

### Usage

1. **Data Preprocessing**
   
   Run the Jupyter notebook to fetch and preprocess Quranic data:
   ```bash
   jupyter notebook notebooks/Data_Preprocessing.ipynb
   ```
   
   This will:
   - Fetch Quranic verses from AlQuran Cloud API
   - Process Arabic (Simple & Uthmani) and English translations
   - Generate structured CSV files (`surahs.csv`, `ayahs.csv`)

2. **Ingest Verses Into Vector Store**
   After generating `ayahs.csv` via the notebook, ingest:
   ```bash
   python -m app.scripts.ingest_verses --csv ayahs.csv --limit 200
   ```
   (Remove `--limit` to process all verses.)

3. **Run Streamlit Demo**
   ```bash
   streamlit run app/ui/app.py
   ```

4. **Programmatic Semantic Search**
   ```python
   from app.utils.retrieval import semantic_search
   hits = semantic_search("mercy", top_k=5)
   for h in hits:
       print(h["verse_id"], h["similarity"], h["content"][:120])
   ```

5. **Agentic Answer Generation**
   ```python
   from app.services.agent import answer_query
   result = answer_query("What does the Quran say about patience?", top_k=5)
   print(result["answer"])  # Grounded response
   ```

## 🔧 Technologies Used

### Backend
- **Python**: Core programming language
- **LangChain**: Framework for building LLM applications
- **HuggingFace**: Embeddings model (`sentence-transformers/all-MiniLM-L6-v2`)
- **Supabase**: PostgreSQL database with vector search capabilities

### Data Sources
- **[AlQuran Cloud API](https://alquran.cloud/)**: Quranic verses and metadata
- **[Hadith API](https://hadithapi.com/docs/chapters)**: Authenticated Hadith collections

### AI/ML
- **Sentence Transformers**: MiniLM embeddings (384-dim)
- **Agentic RAG**: Retrieval + structured prompt construction
- **Google Gemini**: Generative layer (configurable)
- **Vector Search (pgvector)**: Cosine similarity via RPC

## 📊 Data Processing

The project uses the following Quran editions:

1. **Arabic Quran (Simple)** - `quran-simple`
   - Format: Text
   - Direction: RTL (Right-to-Left)

2. **English Translation (Saheeh International)** - `en.sahih`
   - Format: Text
   - Type: Translation

3. **Arabic Quran (Uthmani Script)** - `quran-uthmani`
   - Format: Uthmani Script
   - Direction: RTL

### Verse Structure
Each verse contains:
- Surah number and name
- Verse number and key
- Juz (part) number
- Ruku (section) number
- Text in multiple formats (Simple Arabic, Uthmani, English)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [AlQuran Cloud](https://alquran.cloud/) for providing the Quranic API
- [Hadith API](https://hadithapi.com/) for authenticated Hadith collections
- [Supabase](https://supabase.com/) for the database infrastructure
- [HuggingFace](https://huggingface.co/) for the embeddings models

## 📧 Contact & Next Steps

Open an issue for feature requests or improvements. Planned enhancements:
- Hadith ingestion & unified retrieval across Quran + Hadith
- FastAPI backend exposing `/search` and `/answer` endpoints
- Authentication & per-user history tracking

---

**Made with ❤️ for the Muslim community**
