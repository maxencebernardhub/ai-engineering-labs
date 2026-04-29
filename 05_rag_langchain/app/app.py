"""
Lab 05 — RAG with LangChain: Streamlit application entry point.

Provides an interactive UI for the full RAG pipeline:
- Sidebar: LLM provider/model selector, vector store stats and controls
- Left column: document upload (PDF, TXT, MD) with ingestion feedback
- Right column: question input, generated answer, and sourced chunks

Run with: streamlit run app/app.py (from the 05_rag_langchain/ directory)
"""

import sys
import tempfile
from pathlib import Path

# When Streamlit runs 'streamlit run app/app.py', it prepends the app/ directory
# to sys.path. This makes 'from app import config' resolve to app.py itself
# (circular import) instead of the app/ package. Fix: ensure the lab root
# (parent of app/) is first in sys.path.
_lab_root = str(Path(__file__).parent.parent)
if _lab_root not in sys.path:
    sys.path.insert(0, _lab_root)

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config
from app.ingest import ingest_document
from app.query import answer_question

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources — initialised once, shared across all reruns
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-m3)…")
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Initialising vector store…")
def _get_vectorstore() -> Chroma:
    config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(config.CHROMA_DB_DIR),
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    selected_label = st.selectbox(
        "LLM provider / model",
        options=list(config.AVAILABLE_MODELS.keys()),
        index=list(config.AVAILABLE_MODELS.keys()).index(config.DEFAULT_MODEL),
    )
    model_id = config.AVAILABLE_MODELS[selected_label]
    provider = model_id.split(":")[0]
    st.caption(f"`{model_id}`")

    st.divider()

    vectorstore = _get_vectorstore()
    doc_count = vectorstore._collection.count()
    st.metric("Indexed chunks", doc_count)
    if st.button("🗑️ Clear vector store", use_container_width=True):
        vectorstore._collection.delete(where={"md5": {"$ne": ""}})
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption(
        f"Embeddings: `{config.EMBEDDING_MODEL}`  \n"
        f"Reranker: `{config.RERANKER_MODEL}`  \n"
        f"Chunk size: {config.CHUNK_SIZE} | Overlap: {config.CHUNK_OVERLAP}  \n"
        f"Retrieval top-k: {config.RETRIEVAL_TOP_K} → rerank top-{config.RERANK_TOP_K}"
    )

# ---------------------------------------------------------------------------
# Main layout — two columns
# ---------------------------------------------------------------------------

st.title("🔍 RAG Assistant")
st.caption(
    "Upload a document, then ask questions. Answers are grounded in your "
    "documents and include source references."
)

col_upload, col_qa = st.columns([1, 2], gap="large")

# ---------------------------------------------------------------------------
# Left column — document upload
# ---------------------------------------------------------------------------

with col_upload:
    st.subheader("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        help="Supported formats: PDF, TXT, Markdown",
    )

    if uploaded_file is not None:
        # Use (name, size) as a session key to avoid re-ingesting on every rerun
        session_key = f"ingested__{uploaded_file.name}__{uploaded_file.size}"

        if session_key not in st.session_state:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)
            try:
                with st.spinner(f"Ingesting **{uploaded_file.name}**…"):
                    skipped, num_chunks = ingest_document(
                        tmp_path, _get_vectorstore(), source_name=uploaded_file.name
                    )
                st.session_state[session_key] = (skipped, num_chunks)
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                st.session_state[session_key] = None
            finally:
                tmp_path.unlink(missing_ok=True)

        result = st.session_state.get(session_key)
        if result is not None:
            skipped, num_chunks = result
            if skipped:
                st.info(
                    f"**{uploaded_file.name}** is already indexed — skipped.",
                    icon="ℹ️",
                )
            else:
                st.success(
                    f"**{uploaded_file.name}** indexed into **{num_chunks}** chunks.",
                    icon="✅",
                )

# ---------------------------------------------------------------------------
# Right column — Q&A
# ---------------------------------------------------------------------------

with col_qa:
    st.subheader("💬 Ask a question")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    question = st.text_input(
        "Your question",
        placeholder="What does this document say about…?",
        label_visibility="collapsed",
    )

    ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    if ask_clicked and question.strip():
        if vectorstore._collection.count() == 0:
            st.warning(
                "The knowledge base is empty. Please upload a document first.",
                icon="⚠️",
            )
        else:
            try:
                llm = init_chat_model(model_id)
            except Exception as exc:
                st.error(f"Could not initialise **{selected_label}**: {exc}")
                st.stop()

            with st.spinner("Thinking…"):
                try:
                    result = answer_question(question, vectorstore, llm)
                    st.session_state.last_result = result
                except Exception as exc:
                    st.error(
                        f"Generation failed — check that your **{provider.upper()}** "
                        f"API key is set in the `.env` file at the repo root.  \n"
                        f"Error: `{exc}`"
                    )
                    st.stop()

    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("### Answer")
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                for i, src in enumerate(result["sources"], 1):
                    page_info = (
                        f" — page {src['page'] + 1}" if src["page"] is not None else ""
                    )
                    st.markdown(
                        f"**[{i}] {src['source_file']}{page_info}**  \n"
                        f"> {src['content']}"
                    )
                    if i < len(result["sources"]):
                        st.divider()
