import streamlit as st
from db import get_threads, create_thread, get_messages, save_message, update_thread_title
from rag_engine import retrieve_relevant_docs, stream_rag_answer


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
    1. Document Hub (Active Document & Replace option)
    2. '+ New Chat' action
    3. Saved thread history from Supabase
    """
    with st.sidebar:
        st.markdown("### Document Hub")

        active_doc = st.session_state.get("active_doc_name")
        if active_doc:
            st.success(f"Active: **{active_doc}**")
            uploaded_file = st.file_uploader(
                "Replace Active PDF",
                type=["pdf"],
                key="sidebar_pdf_replace",
                help="Upload a new PDF to replace the current document"
            )
        else:
            uploaded_file = None
            st.caption("Drop a document in the main area to begin.")

        st.divider()

        # Thread Management
        st.markdown("### Conversations")
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            new_id = create_thread(title="New Chat")
            st.session_state["active_thread_id"] = new_id
            st.rerun()

        # Render list of saved threads
        threads = get_threads()
        if not threads:
            st.caption("No conversations yet.")
        else:
            for thread in threads:
                tid = thread["id"]
                title = thread.get("title", "New Chat")
                is_active = (tid == st.session_state.get("active_thread_id"))

                label = f"> {title}" if is_active else f"  {title}"
                if st.button(label, key=f"thread_btn_{tid}", use_container_width=True):
                    st.session_state["active_thread_id"] = tid
                    st.rerun()

    return uploaded_file


def render_empty_state():
    """Prominent centered dropzone when no PDF is loaded yet."""
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 4rem; margin-bottom: 2rem;">
                <h1 style="font-size: 2.4rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                    PDF Question Answering
                </h1>
                <p style="color: #A1A1AA; font-size: 1.05rem; line-height: 1.6; max-width: 550px; margin: 0 auto 2rem auto;">
                    Drop any syllabus, research paper, or textbook below to extract insights, summarize chapters, and ask grounded questions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        center_file = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            key="main_pdf_uploader",
            label_visibility="collapsed"
        )
    return center_file


def render_chat_area(vectorstore, gemini_api_key: str):
    """
    Renders active document bar, message history, live token streaming,
    and source citations.
    """
    thread_id = st.session_state.get("active_thread_id")
    if not thread_id:
        st.session_state["active_thread_id"] = create_thread("New Chat")
        st.rerun()

    # Pinned document badge at the top of the chat
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Active Document:** `{st.session_state.get('active_doc_name')}`")
    with col2:
        st.caption("Citations Grounded")

    st.markdown("<hr style='border-color: #1F1F23; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    # 1. Render all previous messages
    messages = get_messages(thread_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Document Citations", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"**Page {src.get('page', '?')}:** {src.get('snippet', '')}")

    # 2. Accept new question from user
    if prompt := st.chat_input("Ask a question about your document..."):
        # Auto-title thread from the first question
        if not messages:
            update_thread_title(thread_id, prompt)

        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message(thread_id=thread_id, role="user", content=prompt)

        # Retrieve relevant passages
        docs, sources = retrieve_relevant_docs(vectorstore, prompt, k=4)

        # Stream assistant response in real time
        with st.chat_message("assistant"):
            stream_gen = stream_rag_answer(docs, prompt, messages, gemini_api_key)
            full_response = st.write_stream(stream_gen)

            # Display source citations directly under the answer
            if sources:
                with st.expander("View Document Citations", expanded=False):
                    for src in sources:
                        st.caption(f"**Page {src.get('page', '?')}:** {src.get('snippet', '')}")

        # Persist assistant response & citations
        save_message(thread_id=thread_id, role="assistant", content=full_response, sources=sources)
        st.rerun()
