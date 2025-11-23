from contextvars import ContextVar
from supabase import create_client
from app.config.secrets import SUPABASE_KEY, SUPABASE_URL

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY missing in environment. Check your .env file.")

# Base client used as fallback when no contextual client is set.
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_supabase_context: ContextVar = ContextVar("supabase_context", default=None)

def set_supabase_client(client):
    """Override client for current context (e.g. per request)."""
    _supabase_context.set(client)

def get_supabase_client():
    """Return contextual client if set, else the base `supabase`."""
    client = _supabase_context.get()
    return client or supabase

