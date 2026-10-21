import streamlit as st
from db import (
    get_threads,
    create_thread,
    get_messages,
    save_message,
    update_thread_title,
    cleanup_empty_threads,
    delete_all_threads
)
from rag_engine import retrieve_relevant_docs, stream_rag_answer, stream_general_answer


def inject_custom_styles():
    """Inject custom Geist typography and dark minimalist CSS."""
    try:
        with open("assets/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def render_sidebar():
    """
    Renders the persistent left sidebar containing:
    1. Document Hub (Active Document & Upload/Replace option)
    2. '+ New Chat' action
    3. Saved thread history from Supabase (cleaned of empty threads)
    4. Clear history option
    """
    with st.sidebar:
        st.markdown("### Document Hub")

        active_doc = st.session_state.get("active_doc_name")
        if active_doc:
            st.success(f"Active: **{active_doc}**")
            col_a, col_b = st.columns(2)
            with col_a:
                uploaded_file = st.file_uploader(
                    "Replace PDF",
                    type=["pdf"],
                    key="sidebar_pdf_replace",
                    label_visibility="collapsed",
                    help="Upload a new PDF to replace the current document"
                )
            with col_b:
                if st.button("Unload Doc", use_container_width=True):
                    st.session_state["vectorstore"] = None
                    st.session_state["active_doc_name"] = None
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                key="sidebar_pdf_upload",
                help="Upload a PDF to enable grounded document Q&A"
            )
            st.caption("Drop a document to ground answers with citations.")

        st.divider()

        # Thread Management
        st.markdown("### Conversations")
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            st.session_state["active_thread_id"] = None
            st.rerun()

        # Render list of saved threads
        threads = get_threads()
        if not threads:
            st.caption("No saved conversations yet.")
        else:
            for thread in threads:
                tid = thread["id"]
                title = thread.get("title") or "Conversation"
                is_active = (tid == st.session_state.get("active_thread_id"))

                label = f"> {title}" if is_active else f"  {title}"
                if st.button(label, key=f"thread_btn_{tid}", use_container_width=True):
                    st.session_state["active_thread_id"] = tid
                    st.rerun()

            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Clear History", use_container_width=True):
                delete_all_threads()
                st.rerun()

    return uploaded_file


def render_main_area(vectorstore, gemini_api_key: str, on_file_upload_cb):
    """
    Unified chat interface. Works both before and after document upload.
    - If no document is uploaded, provides direct assistant answers and an upload card.
    - If document is indexed, provides grounded RAG answers with citations.
    """
    thread_id = st.session_state.get("active_thread_id")
    active_doc = st.session_state.get("active_doc_name")

    # Document Header / Uploader
    if active_doc:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**Active Document:** `{active_doc}`")
        with col2:
            st.caption("Citations Grounded")
        st.markdown("<hr style='border-color: #1F1F23; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    else:
        # Before document upload: show a clean top header and compact dropzone
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <h2 style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.25rem;">
                    PDF Question Answering & AI Assistant
                </h2>
                <p style="color: #A1A1AA; font-size: 0.95rem; margin-bottom: 1rem;">
                    Chat with the AI directly, or upload a document below to ask grounded questions with source citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Upload Document for Grounded Q&A", expanded=True):
            center_file = st.file_uploader(
                "Upload your PDF document",
                type=["pdf"],
                key="main_pdf_uploader",
                label_visibility="collapsed"
            )
            if center_file and center_file.name != active_doc:
                on_file_upload_cb(center_file)
                st.rerun()

        st.markdown("<hr style='border-color: #1F1F23; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    # 1. Render message history
    messages = get_messages(thread_id) if thread_id else []

    if not messages:
        with st.chat_message("assistant"):
            if active_doc:
                st.markdown(f"Document **{active_doc}** is loaded and indexed. Ask any question about its content!")
            else:
                st.markdown("Hello! You can ask me anything directly, or upload a PDF above to ground answers in your document.")
    else:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("View Document Citations", expanded=False):
                        for src in msg["sources"]:
                            st.caption(f"**Page {src.get('page', '?')}:** {src.get('snippet', '')}")

    # 2. Chat Input Bar (ALWAYS ACTIVE)
    input_placeholder = (
        f"Ask a question about {active_doc}..."
        if active_doc
        else "Ask anything, or ask about an uploaded document..."
    )

    if prompt := st.chat_input(input_placeholder):
        # Initialize thread if this is the first turn
        if not thread_id:
            thread_id = create_thread(title=prompt)
            st.session_state["active_thread_id"] = thread_id
        elif not messages:
            update_thread_title(thread_id, prompt)

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message(thread_id=thread_id, role="user", content=prompt)

        # Generate response
        with st.chat_message("assistant"):
            if vectorstore is not None:
                # Grounded RAG mode
                docs, sources = retrieve_relevant_docs(vectorstore, prompt, k=4)
                stream_gen = stream_rag_answer(docs, prompt, messages, gemini_api_key)
                full_response = st.write_stream(stream_gen)

                if sources:
                    with st.expander("View Document Citations", expanded=False):
                        for src in sources:
                            st.caption(f"**Page {src.get('page', '?')}:** {src.get('snippet', '')}")

                save_message(thread_id=thread_id, role="assistant", content=full_response, sources=sources)
            else:
                # Direct conversational mode (before uploading doc)
                stream_gen = stream_general_answer(prompt, messages, gemini_api_key)
                full_response = st.write_stream(stream_gen)
                save_message(thread_id=thread_id, role="assistant", content=full_response, sources=[])

        st.rerun()
