from contextvars import ContextVar
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

base_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

supabase_context: ContextVar = ContextVar("supabase_context", default=None)

def set_supabase_client(client):
    supabase_context.set(client)

def get_supabase_client():
    client = supabase_context.get()
    if client is None:
        return base_supabase
    return client
