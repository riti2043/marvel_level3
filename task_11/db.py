import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and cache the Supabase connection pool."""
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def get_threads():
    """Fetch all conversation threads ordered by newest first."""
    supabase = get_supabase_client()
    if not supabase:
        return st.session_state.get("local_threads", [{"id": "local_default", "title": "New Chat"}])
    try:
        response = (
            supabase.table("threads")
            .select("id, title, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception:
        # Fallback to local session state if tables haven't been created yet in Supabase
        return st.session_state.get("local_threads", [{"id": "local_default", "title": "New Chat"}])


def create_thread(title: str = "New Chat") -> str:
    """Create a new conversation thread and return its UUID."""
    supabase = get_supabase_client()
    if not supabase:
        new_id = f"local_{len(st.session_state.get('local_threads', [])) + 1}"
        if "local_threads" not in st.session_state:
            st.session_state["local_threads"] = []
        st.session_state["local_threads"].insert(0, {"id": new_id, "title": title})
        return new_id
    try:
        response = supabase.table("threads").insert({"title": title}).execute()
        if response.data:
            return response.data[0]["id"]
    except Exception:
        # Local session fallback
        new_id = f"local_{len(st.session_state.get('local_threads', [])) + 1}"
        if "local_threads" not in st.session_state:
            st.session_state["local_threads"] = []
        st.session_state["local_threads"].insert(0, {"id": new_id, "title": title})
        return new_id
    return "local_default"


def update_thread_title(thread_id: str, new_title: str):
    """Update a thread's title (e.g., using the first question asked)."""
    supabase = get_supabase_client()
    clean_title = new_title[:40]
    if "local_threads" in st.session_state:
        for t in st.session_state["local_threads"]:
            if t["id"] == thread_id:
                t["title"] = clean_title
                break
    if supabase:
        try:
            supabase.table("threads").update({"title": clean_title}).eq("id", thread_id).execute()
        except Exception:
            pass


def get_messages(thread_id: str):
    """Retrieve full conversation history for a specific thread."""
    if not thread_id:
        return []
    supabase = get_supabase_client()
    if not supabase or thread_id.startswith("local_"):
        local_store = st.session_state.get("local_messages", {})
        return local_store.get(thread_id, [])
    try:
        response = (
            supabase.table("messages")
            .select("id, role, content, sources, created_at")
            .eq("thread_id", thread_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data or []
    except Exception:
        local_store = st.session_state.get("local_messages", {})
        return local_store.get(thread_id, [])


def save_message(thread_id: str, role: str, content: str, sources: list = None):
    """Persist a user question or assistant answer."""
    if not thread_id:
        return
    msg = {"id": thread_id, "role": role, "content": content, "sources": sources or []}
    if "local_messages" not in st.session_state:
        st.session_state["local_messages"] = {}
    if thread_id not in st.session_state["local_messages"]:
        st.session_state["local_messages"][thread_id] = []
    st.session_state["local_messages"][thread_id].append(msg)

    supabase = get_supabase_client()
    if supabase and not thread_id.startswith("local_"):
        try:
            payload = {
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "sources": sources or []
            }
            supabase.table("messages").insert(payload).execute()
        except Exception:
            pass