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


def cleanup_empty_threads():
    """Purge empty threads with no messages from the database to remove clutter."""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        threads_res = supabase.table("threads").select("id").execute()
        msgs_res = supabase.table("messages").select("thread_id").execute()
        used_tids = set(m["thread_id"] for m in (msgs_res.data or []))
        for t in (threads_res.data or []):
            if t["id"] not in used_tids:
                supabase.table("threads").delete().eq("id", t["id"]).execute()
    except Exception:
        pass


def get_threads():
    """
    Fetch conversation threads that actually contain messages,
    plus the currently active thread, ordered newest first.
    """
    supabase = get_supabase_client()
    if not supabase:
        return [
            t for t in st.session_state.get("local_threads", [])
            if t.get("has_messages") or t.get("id") == st.session_state.get("active_thread_id")
        ]
    try:
        # Fetch thread IDs with messages to avoid showing blank ghost threads
        msg_res = supabase.table("messages").select("thread_id").execute()
        active_thread_ids = set(m["thread_id"] for m in (msg_res.data or []))

        response = (
            supabase.table("threads")
            .select("id, title, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        threads = response.data or []
        current_tid = st.session_state.get("active_thread_id")

        return [
            t for t in threads
            if t["id"] in active_thread_ids or (current_tid and t["id"] == current_tid)
        ]
    except Exception:
        return st.session_state.get("local_threads", [])


def create_thread(title: str = "New Chat") -> str:
    """Create a new conversation thread and return its UUID."""
    supabase = get_supabase_client()
    if not supabase:
        new_id = f"local_{len(st.session_state.get('local_threads', [])) + 1}"
        if "local_threads" not in st.session_state:
            st.session_state["local_threads"] = []
        st.session_state["local_threads"].insert(0, {"id": new_id, "title": title, "has_messages": False})
        return new_id
    try:
        response = supabase.table("threads").insert({"title": title}).execute()
        if response.data:
            return response.data[0]["id"]
    except Exception:
        new_id = f"local_{len(st.session_state.get('local_threads', [])) + 1}"
        if "local_threads" not in st.session_state:
            st.session_state["local_threads"] = []
        st.session_state["local_threads"].insert(0, {"id": new_id, "title": title, "has_messages": False})
        return new_id
    return "local_default"


def update_thread_title(thread_id: str, new_title: str):
    """Update a thread's title (e.g., using the first question asked)."""
    supabase = get_supabase_client()
    clean_title = new_title[:35].strip()
    if not clean_title:
        clean_title = "New Chat"

    if "local_threads" in st.session_state:
        for t in st.session_state["local_threads"]:
            if t["id"] == thread_id:
                t["title"] = clean_title
                t["has_messages"] = True
                break

    if supabase and not thread_id.startswith("local_"):
        try:
            supabase.table("threads").update({"title": clean_title}).eq("id", thread_id).execute()
        except Exception:
            pass


def delete_all_threads():
    """Wipe all conversations and messages."""
    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table("messages").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            supabase.table("threads").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        except Exception:
            pass
    st.session_state["local_threads"] = []
    st.session_state["local_messages"] = {}
    st.session_state["active_thread_id"] = None


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

    # Mark local thread as having messages
    if "local_threads" in st.session_state:
        for t in st.session_state["local_threads"]:
            if t["id"] == thread_id:
                t["has_messages"] = True

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
