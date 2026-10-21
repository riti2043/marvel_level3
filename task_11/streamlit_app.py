import os
import streamlit as st
from ui import inject_custom_styles, render_sidebar, render_main_area
from rag_engine import parse_and_chunk_pdf, build_vectorstore
from db import cleanup_empty_threads

# 1. Page Configuration & Typography
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
    st.session_state["active_thread_id"] = None
if "db_cleaned" not in st.session_state:
    cleanup_empty_threads()
    st.session_state["db_cleaned"] = True

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

# 4. Render Unified Main Chat Area (works before and after upload)
render_main_area(
    vectorstore=st.session_state["vectorstore"],
    gemini_api_key=gemini_api_key,
    on_file_upload_cb=index_document
)
