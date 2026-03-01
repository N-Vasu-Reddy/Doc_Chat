import os
import tempfile
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Agent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F0;
    color: #1A1A1A;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8E4DC;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }
.block-container { max-width: 860px; padding: 2.5rem 2rem; }

h1 { font-family: 'DM Serif Display', serif !important;
     font-size: 2.4rem !important; letter-spacing: -0.02em;
     color: #1A1A1A; margin-bottom: 0.2rem !important; }
h3 { font-family: 'DM Serif Display', serif !important;
     font-size: 1.15rem !important; font-weight: 400; color: #666; }

.stButton > button {
    background: #1A1A1A; color: #F7F5F0;
    border: none; border-radius: 6px;
    padding: 0.55rem 1.4rem; font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    letter-spacing: 0.03em; transition: background 0.2s;
}
.stButton > button:hover { background: #333; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #FFFFFF; border: 1px solid #DDD8CE;
    border-radius: 6px; font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem; color: #1A1A1A;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #1A1A1A !important; box-shadow: none !important;
}

.chat-user {
    background: #1A1A1A; color: #F7F5F0;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem; margin: 0.4rem 0 0.4rem auto;
    max-width: 76%; font-size: 0.9rem; line-height: 1.55;
}
.chat-ai {
    background: #FFFFFF; color: #1A1A1A;
    border: 1px solid #E8E4DC;
    border-radius: 12px 12px 12px 2px;
    padding: 0.75rem 1rem; margin: 0.4rem 0;
    max-width: 76%; font-size: 0.9rem; line-height: 1.6;
}
.source-pill {
    display: inline-block; background: #F0EDE6;
    border: 1px solid #DDD8CE; border-radius: 20px;
    padding: 0.18rem 0.65rem; font-size: 0.74rem;
    color: #666; margin: 0.12rem 0.1rem;
}
.badge-free {
    display: inline-block; background: #DCFCE7;
    border: 1px solid #86EFAC; border-radius: 4px;
    padding: 0.1rem 0.45rem; font-size: 0.7rem;
    color: #15803D; font-weight: 500; margin-left: 6px;
}
.badge-paid {
    display: inline-block; background: #FEF9C3;
    border: 1px solid #FDE047; border-radius: 4px;
    padding: 0.1rem 0.45rem; font-size: 0.7rem;
    color: #854D0E; font-weight: 500; margin-left: 6px;
}
.status-ok   { color: #16A34A; font-size: 0.82rem; }
.status-info { color: #6B7280; font-size: 0.82rem; }
.info-box {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 6px; padding: 0.65rem 0.9rem;
    font-size: 0.83rem; color: #1E40AF; margin-bottom: 0.5rem;
}

hr { border: none; border-top: 1px solid #E8E4DC; margin: 1.25rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Core dependency check ────────────────────────────────────────────────────
ALWAYS_REQUIRED = {
    "langchain_community": "langchain-community>=0.4.1",
    "langchain_groq":      "langchain-groq>=1.1.2",
    "langchain_text_splitters": "langchain-text-splitters>=1.1.1",
    "langchain_core":      "langchain-core>=1.2.0",
    "faiss":               "faiss-cpu>=1.8.0",
    "pypdf":               "pypdf>=4.0.0",
}

missing_core = []
for mod, pkg in ALWAYS_REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        missing_core.append(pkg)

if missing_core:
    st.error("**Missing core packages.** Run the command below and restart:")
    st.code("pip install " + " ".join(f'"{p}"' for p in missing_core), language="bash")
    st.stop()

# ─── Stable imports (always available after core check) ──────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# ─── Embedding provider helper ────────────────────────────────────────────────
EMBED_PROVIDERS = {
    "🟢 HuggingFace Local (FREE — no API key)": "hf_local",
    "🔵 HuggingFace Inference API (FREE tier)": "hf_api",
    "🟡 OpenAI (Paid)":                          "openai",
}

HF_LOCAL_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",       # fast, small, great quality
    "sentence-transformers/all-mpnet-base-v2",       # higher quality
    "BAAI/bge-small-en-v1.5",                        # best small open-source
    "BAAI/bge-large-en-v1.5",                        # best large open-source
]

HF_API_MODELS = [
    "Qwen/Qwen3-Embedding-8B",
    "BAAI/bge-small-en-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
]

OPENAI_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]


def build_embeddings(provider: str, model: str, openai_key: str = "", hf_token: str = ""):
    """
    Returns a LangChain Embeddings object for the chosen provider.
    All imports are done inside the function so missing optional packages
    only error when actually selected — not at startup.
    """
    if provider == "hf_local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings   # langchain-huggingface>=0.1
        except ImportError:
            st.error(
                "Missing package for local HuggingFace embeddings. Install and restart:\n"
                "`pip install langchain-huggingface>=0.1.0 sentence-transformers>=3.0.0`"
            )
            st.stop()
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "hf_api":
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
        except ImportError:
            st.error(
                "Missing package. Install and restart:\n"
                "`pip install langchain-huggingface>=0.1.0`"
            )
            st.stop()
        return HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token=hf_token or None,
        )

    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            st.error(
                "Missing package. Install and restart:\n"
                "`pip install langchain-openai>=1.1.10`"
            )
            st.stop()
        if not openai_key:
            st.error("OpenAI API key is required for OpenAI embeddings.")
            st.stop()
        return OpenAIEmbeddings(api_key=openai_key, model=model)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vectorstore": None,
    "indexed_files": [],
    "embed_provider": None,   # track which provider was used to build the index
    "embed_model": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ Settings")
    st.markdown("---")

    # ── Groq ──
    groq_key = st.text_input(
        "Groq API Key", type="password",
        placeholder="gsk_...",
        help="Get at console.groq.com — free tier available",
    )
    groq_model = st.selectbox(
        "LLM Model (Groq)",
        ["openai/gpt-oss-20b", "llama-3.1-8b-instant",
         "mixtral-8x7b-32768", "gemma2-9b-it"],
    )

    st.markdown("---")

    # ── Embedding provider ──
    st.markdown("**Embedding Provider**")
    embed_provider_label = st.selectbox(
        "Provider",
        list(EMBED_PROVIDERS.keys()),
        label_visibility="collapsed",
    )
    embed_provider = EMBED_PROVIDERS[embed_provider_label]

    # Show relevant model + key fields per provider
    if embed_provider == "hf_local":
        st.markdown(
            '<div class="info-box">🟢 Runs entirely on your machine.<br>'
            'First use downloads the model (~90–440 MB). No API key needed.</div>',
            unsafe_allow_html=True,
        )
        embed_model = st.selectbox("Model", HF_LOCAL_MODELS)
        openai_key = ""
        hf_token = ""

    elif embed_provider == "hf_api":
        st.markdown(
            '<div class="info-box">🔵 Uses HuggingFace Inference API.<br>'
            'Free tier works without a token but may rate-limit. '
            'Get a free token at huggingface.co/settings/tokens</div>',
            unsafe_allow_html=True,
        )
        embed_model = st.selectbox("Model", HF_API_MODELS)
        hf_token = st.text_input(
            "HF Token (optional)", type="password", placeholder="hf_..."
        )
        openai_key = ""

    else:  # openai
        st.markdown(
            '<div class="info-box">🟡 OpenAI paid API.<br>'
            'Requires an active billing plan at platform.openai.com</div>',
            unsafe_allow_html=True,
        )
        embed_model = st.selectbox("Model", OPENAI_MODELS)
        openai_key = st.text_input(
            "OpenAI API Key", type="password", placeholder="sk-..."
        )
        hf_token = ""

    st.markdown("---")

    top_k       = st.slider("Top-K chunks retrieved", 1, 10, 4)
    chunk_size  = st.slider("Chunk size (chars)", 500, 4000, 1500, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 200, step=50)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, step=0.05)

    st.markdown("---")

    if st.session_state.indexed_files:
        st.markdown("**Indexed documents**")
        for f in st.session_state.indexed_files:
            st.markdown(f'<span class="source-pill">📄 {f}</span>', unsafe_allow_html=True)
        if st.button("🗑 Clear index & chat"):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.embed_provider = None
            st.session_state.embed_model = None
            st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# RAG Agent")
st.markdown("### Upload PDFs · Ask questions · Groq LLM + your choice of embeddings")
st.markdown("---")


# ─── LCEL RAG chain ───────────────────────────────────────────────────────────
def build_rag_chain(retriever, groq_api_key: str, model: str, temp: float):
    llm = ChatGroq(api_key=groq_api_key, model=model,
                   temperature=temp, max_tokens=1024)

    prompt = ChatPromptTemplate.from_template("""
You are a precise and helpful assistant. Answer ONLY using the context below.
If the answer is not found, say "I couldn't find that in the provided documents."
Be concise and cite the source document when relevant.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source','?')}, Page: {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


# ─── PDF Upload & Indexing ─────────────────────────────────────────────────────
col_upload, col_btn = st.columns([3, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col_btn:
    index_btn = st.button("⊕ Index Documents", use_container_width=True)

if index_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    elif not groq_key:
        st.error("Groq API key is required (sidebar).")
    else:
        # Warn if provider changed after an index already exists
        if (st.session_state.vectorstore is not None
                and st.session_state.embed_provider != embed_provider):
            st.warning(
                "⚠ You changed the embedding provider but an index already exists. "
                "Clear the index first (sidebar) to avoid dimension mismatch errors."
            )
            st.stop()

        new_files = [f for f in uploaded_files
                     if f.name not in st.session_state.indexed_files]
        if not new_files:
            st.info("All uploaded files are already indexed.")
        else:
            all_docs: list[Document] = []
            progress = st.progress(0, text="Parsing PDFs…")

            for i, uploaded_file in enumerate(new_files):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages = PyPDFLoader(tmp_path).load()
                    for page in pages:
                        page.metadata["source"] = uploaded_file.name
                    all_docs.extend(pages)
                    st.session_state.indexed_files.append(uploaded_file.name)
                    progress.progress(
                        (i + 1) / len(new_files),
                        text=f"Parsed {uploaded_file.name} ({len(pages)} pages)",
                    )
                except Exception as e:
                    st.error(f"Failed to parse {uploaded_file.name}: {e}")
                finally:
                    if tmp_path:
                        try: os.unlink(tmp_path)
                        except Exception: pass

            if all_docs:
                progress.progress(0.78, text="Splitting into chunks…")
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    add_start_index=True,
                    separators=["\n\n", "\n", ". ", " ", ""],
                ).split_documents(all_docs)

                provider_label = embed_provider_label.split("(")[0].strip()
                progress.progress(0.86, text=f"Embedding {len(chunks)} chunks via {provider_label}…")

                try:
                    embeddings_obj = build_embeddings(
                        embed_provider, embed_model, openai_key, hf_token
                    )

                    if st.session_state.vectorstore is None:
                        vs = FAISS.from_documents(chunks, embeddings_obj)
                    else:
                        new_vs = FAISS.from_documents(chunks, embeddings_obj)
                        st.session_state.vectorstore.merge_from(new_vs)
                        vs = st.session_state.vectorstore

                    st.session_state.vectorstore = vs
                    st.session_state.embed_provider = embed_provider
                    st.session_state.embed_model = embed_model

                    progress.progress(1.0, text="Done ✓")
                    st.success(
                        f"✅ Indexed **{len(chunks)} chunks** from "
                        f"**{len(new_files)} file(s)** using **{provider_label}**."
                    )
                except Exception as e:
                    st.error(f"Embedding error: {e}")

st.markdown("---")


# ─── Chat UI ──────────────────────────────────────────────────────────────────
if st.session_state.vectorstore:
    n = len(st.session_state.indexed_files)
    prov = st.session_state.embed_provider or ""
    badge = "free" if prov in ("hf_local", "hf_api") else "paid"
    prov_name = {"hf_local": "HF Local", "hf_api": "HF API", "openai": "OpenAI"}.get(prov, prov)
    st.markdown(
        f'<span class="status-ok">● {n} doc(s) indexed</span> '
        f'<span class="badge-{"free" if badge == "free" else "paid"}">{prov_name}</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-info">○ No documents indexed — upload PDFs above</span>',
        unsafe_allow_html=True,
    )

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            pills = " ".join(f'<span class="source-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(pills, unsafe_allow_html=True)

# Input form
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        query = st.text_input(
            "Q", placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
        )
    with cols[1]:
        send = st.form_submit_button("Ask →", use_container_width=True)

if send and query:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.warning("Please index at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )

        with st.spinner("Thinking…"):
            try:
                source_docs = retriever.invoke(query)
                sources = list(dict.fromkeys(
                    d.metadata.get("source", "unknown") for d in source_docs
                ))
                answer = build_rag_chain(retriever, groq_key, groq_model, temperature).invoke(query)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", "content": f"⚠ Error: {e}", "sources": []
                })
        st.rerun()