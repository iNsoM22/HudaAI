from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets["SUPABASE_URL"]
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets["SUPABASE_KEY"]


GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash") or st.secrets["GOOGLE_MODEL"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

HF_MODEL = os.getenv("EMBEDDINGS_MODEL") or st.secrets["EMBEDDINGS_MODEL"]
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or st.secrets["HUGGINGFACE_API_KEY"]
EMBED_MODE_OFFLINE = os.getenv("EMBED_MODE_OFFLINE") or st.secrets["EMBED_MODE_OFFLINE"]
EMBED_MODE = EMBED_MODE_OFFLINE == "TRUE"
EMBED_DIM = 768

GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY") or st.secrets["GOOGLE_SERPER_KEY"]