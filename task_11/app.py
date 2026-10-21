import os
import streamlit as st
from ui import inject_custom_styles, render_sidebar, render_empty_state, render_chat_area
from rag_engine import parse_and_chunk_pdf, build_vectorstore
from db import create_thread, get_threads

# 1. Page Configuration & Custom CSS Injection
st.set_page_config(
    page_title="PDF Question Answering",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_custom_styles()

# 2. Session State Initialization
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "active_doc_name" not in st.session_state:
    st.session_state["active_doc_name"] = None
if "active_thread_id" not in st.session_state:
    threads = get_threads()
    if threads:
        st.session_state["active_thread_id"] = threads[0]["id"]
    else:
        st.session_state["active_thread_id"] = create_thread("New Chat")

# Retrieve Gemini API Key securely from secrets or environment
gemini_api_key = (
    st.secrets.get("GEMINI_API_KEY", "")
    or st.secrets.get("GOOGLE_API_KEY", "")
    or os.environ.get("GEMINI_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
)


def index_document(file):
    """Parses and indexes the uploaded PDF into the in-memory vectorstore."""
    with st.spinner("Processing document & building vector index..."):
        try:
            chunks = parse_and_chunk_pdf(file)
            vs = build_vectorstore(chunks, gemini_api_key)
            st.session_state["vectorstore"] = vs
            st.session_state["active_doc_name"] = file.name
            st.toast(f"Indexed {len(chunks)} chunks from {file.name}!")
        except Exception as e:
            st.error(f"Error indexing document: {str(e)}")

# 3. Render Persistent Sidebar
sidebar_file = render_sidebar()

# Handle document uploaded via sidebar
if sidebar_file and sidebar_file.name != st.session_state.get("active_doc_name"):
    index_document(sidebar_file)
    st.rerun()

# 4. Render Main View: Center Dropzone vs. Active Chat Interface
if st.session_state["vectorstore"] is None:
    # If no PDF is loaded yet, show the prominent center drag-and-drop hero
    center_file = render_empty_state()
    if center_file and center_file.name != st.session_state.get("active_doc_name"):
        index_document(center_file)
        st.rerun()
else:
    # Once a PDF is indexed, show the full conversational RAG interface
    render_chat_area(
        vectorstore=st.session_state["vectorstore"],
        gemini_api_key=gemini_api_key
    )