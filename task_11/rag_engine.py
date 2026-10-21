import io
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def parse_and_chunk_pdf(uploaded_file, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Extracts text from an uploaded PDF stream and splits it into semantic chunks
    with page metadata attached.
    """
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
    docs = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": page_idx + 1, "source": uploaded_file.name}
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, gemini_api_key: str):
    """
    Generates embeddings via Google Gemini and stores
    them in an in-memory FAISS index.
    Attempts active models (gemini-embedding-001) with graceful fallbacks.
    """
    candidate_models = [
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
    ]
    last_err = None
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=gemini_api_key
            )
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            last_err = e
            continue
    raise last_err

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def retrieve_relevant_docs(vectorstore, question: str, k: int = 4):
    """
    Retrieves the top k most relevant text chunks from the FAISS vector index
    and formats their page citations.
    """
    docs = vectorstore.similarity_search(question, k=k)
    sources = [
        {
            "page": doc.metadata.get("page", 1),
            "snippet": doc.page_content[:300].strip() + "..."
        }
        for doc in docs
    ]
    return docs, sources


def stream_rag_answer(docs, question: str, chat_history: list, gemini_api_key: str):
    """
    Constructs a grounded RAG prompt with retrieved context and streams the LLM
    response token-by-token.
    """
    context_text = "\n\n---\n\n".join(
        [f"[Page {d.metadata.get('page', '?')}]: {d.page_content}" for d in docs]
    )

    system_prompt = (
        "You are a helpful and precise assistant answering questions about an uploaded PDF document.\n"
        "Use ONLY the following retrieved context to answer the user's question. "
        "If the answer cannot be found in the context, clearly state that the document does not contain that information.\n\n"
        f"Context from Document:\n{context_text}"
    )

    messages = [SystemMessage(content=system_prompt)]

    # Include the last 4 chat turns for conversational awareness
    for msg in chat_history[-4:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))

    messages.append(HumanMessage(content=question))

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.2,
        streaming=True
    )

    # Yield tokens one by one as they arrive from Gemini
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content