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
│   │   └── db.py           # Database connection setup
│   ├── services/           # Business logic services
│   └── utils/
│       └── rag.py          # RAG implementation & vector operations
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

5. **Set up Supabase Database**
   
   Create the following tables in your Supabase project:
   
   - `documents` table with columns:
     - `id` (uuid, primary key)
     - `user_id` (text)
     - `resume_id` (text)
     - `content` (text)
     - `embedding` (vector)
     - `metadata` (jsonb)
   
   - Create the `match_resume_chunks` RPC function for similarity search

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

2. **Vector Storage**
   
   Use the RAG utilities to store embeddings:
   ```python
   from app.utils.rag import get_embeddings, split_text, store_vector
   
   # Initialize embeddings model
   embeddings = get_embeddings()
   
   # Split text into chunks
   chunks = split_text(your_text)
   
   # Store in vector database
   store_vector(user_id, resume_id, chunks, embeddings)
   ```

3. **Semantic Search**
   
   Retrieve relevant content:
   ```python
   from app.utils.rag import retrieve_vectors, get_embeddings
   
   embeddings = get_embeddings()
   query_vector = embeddings.embed_query("your search query")
   
   results = retrieve_vectors(user_id, query_vector, top_k=5)
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
- **Sentence Transformers**: Vector embeddings for semantic search
- **RAG (Retrieval-Augmented Generation)**: Context-aware response generation
- **Vector Search**: Similarity-based document retrieval

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

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for the Muslim community**
