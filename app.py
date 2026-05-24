## RAG Q&A Conversational App — Premium UI
import streamlit as st
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_community.chains import create_history_aware_retriever, create_retrieval_chain  # older builds
    from langchain_community.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS / HTML Injection ─────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════
   FONTS
══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

/* ══════════════════════════════════════════════
   ROOT VARIABLES
══════════════════════════════════════════════ */
:root {
  --bg-base:      #080c14;
  --bg-surface:   #0e1420;
  --bg-raised:    #141c2e;
  --bg-card:      #1a2236;
  --accent:       #00d4aa;
  --accent-dim:   #00d4aa33;
  --accent-glow:  #00d4aa18;
  --accent2:      #7b61ff;
  --accent2-dim:  #7b61ff33;
  --danger:       #ff4d6d;
  --text-primary: #e8edf5;
  --text-muted:   #6b7a99;
  --text-faint:   #3a4560;
  --border:       #1e2d47;
  --border-light: #263550;
  --font-mono:    'Space Mono', monospace;
  --font-body:    'DM Sans', sans-serif;
  --font-display: 'Playfair Display', serif;
  --radius:       12px;
  --radius-sm:    8px;
  --shadow:       0 4px 32px rgba(0,0,0,0.5);
  --glow:         0 0 24px rgba(0,212,170,0.15);
}

/* ══════════════════════════════════════════════
   GLOBAL RESET
══════════════════════════════════════════════ */
* { box-sizing: border-box; }

html, body, .stApp {
  background-color: var(--bg-base) !important;
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}

/* Animated mesh background */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 40% at 10% 15%, #00d4aa0a 0%, transparent 70%),
    radial-gradient(ellipse 50% 60% at 90% 85%, #7b61ff0a 0%, transparent 70%),
    radial-gradient(ellipse 40% 50% at 50% 50%, #00d4aa05 0%, transparent 80%);
  pointer-events: none;
  z-index: 0;
}

/* ══════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* Sidebar header block */
.sidebar-logo {
  background: linear-gradient(135deg, var(--bg-raised), var(--bg-card));
  border-bottom: 1px solid var(--border);
  padding: 28px 24px 20px;
  margin-bottom: 8px;
}

.sidebar-logo .brand {
  font-family: var(--font-mono);
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.5px;
}

.sidebar-logo .brand span {
  color: var(--text-muted);
  font-weight: 400;
}

.sidebar-logo .tagline {
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 4px;
}

/* Sidebar section label */
.sidebar-section {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-faint);
  padding: 0 20px;
  margin: 20px 0 8px;
}

/* ══════════════════════════════════════════════
   INPUTS
══════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea textarea {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.9rem !important;
  padding: 10px 14px !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
  outline: none !important;
}

.stTextInput label,
.stTextArea label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  font-weight: 700 !important;
}

/* Password input eye icon */
[data-testid="stTextInputRootElement"] button {
  background: transparent !important;
  border: none !important;
  color: var(--text-muted) !important;
}

/* ══════════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
  background: var(--bg-raised) !important;
  border: 2px dashed var(--border-light) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  transition: border-color 0.2s, background 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-glow) !important;
}

[data-testid="stFileUploader"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
  color: var(--text-muted) !important;
}

/* ══════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #00b899) !important;
  color: var(--bg-base) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 10px 24px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 16px rgba(0,212,170,0.25) !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(0,212,170,0.4) !important;
  filter: brightness(1.08) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

/* ══════════════════════════════════════════════
   SPINNERS & ALERTS
══════════════════════════════════════════════ */
.stSpinner > div {
  border-top-color: var(--accent) !important;
}

.stAlert {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border-light) !important;
}

[data-testid="stNotification"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-sm) !important;
}

/* ══════════════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* ══════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ══════════════════════════════════════════════
   MARKDOWN CONTENT
══════════════════════════════════════════════ */
.stMarkdown p, .stMarkdown li {
  font-family: var(--font-body) !important;
  line-height: 1.7 !important;
}

.stMarkdown code {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82em !important;
  color: var(--accent) !important;
}

/* ══════════════════════════════════════════════
   METRIC TILES
══════════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 20px !important;
}

[data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  color: var(--accent) !important;
  font-size: 1.5rem !important;
}

/* ══════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
}

/* ══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_link__qRIco { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Component Helpers ────────────────────────────────────────────────────────
def sidebar_header():
    st.sidebar.markdown("""
    <div class="sidebar-logo">
      <div class="brand">Doc<span>Mind</span></div>
      <div class="tagline">Retrieval-Augmented Generation</div>
    </div>
    """, unsafe_allow_html=True)

def section_label(text):
    st.markdown(f"""
    <div style="
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #3a4560;
      margin: 28px 0 10px;
      display: flex;
      align-items: center;
      gap: 10px;
    ">
      <span>{text}</span>
      <div style="flex:1; height:1px; background: linear-gradient(90deg, #1e2d47, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

def chat_bubble(role, content, index=0):
    is_user = role == "user"
    avatar = "👤" if is_user else "🧠"
    bubble_style = f"""
      background: {'linear-gradient(135deg, #1a2a4a, #1e3055)' if is_user else 'linear-gradient(135deg, #141c2e, #1a2438)'};
      border: 1px solid {'#263a5e' if is_user else '#1e2d47'};
      border-radius: {'16px 16px 4px 16px' if is_user else '4px 16px 16px 16px'};
      padding: 14px 18px;
      margin: {'0 0 0 auto' if is_user else '0 auto 0 0'};
      max-width: 78%;
      word-wrap: break-word;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.92rem;
      line-height: 1.65;
      color: #e8edf5;
      {'border-left: 3px solid #00d4aa;' if not is_user else 'border-right: 3px solid #7b61ff;'}
      animation: fadeSlideIn 0.3s ease forwards;
      opacity: 0;
      animation-delay: {index * 0.05}s;
    """
    align = "flex-end" if is_user else "flex-start"

    st.markdown(f"""
    <style>
    @keyframes fadeSlideIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    <div style="display:flex; flex-direction:column; align-items:{align}; margin-bottom:16px;">
      <div style="
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #3a4560;
        margin-bottom: 5px;
        {'padding-right:8px;' if is_user else 'padding-left:8px;'}
      ">{avatar} {'You' if is_user else 'DocMind'}</div>
      <div style="{bubble_style}">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def status_pill(text, color="#00d4aa"):
    st.sidebar.markdown(f"""
    <div style="
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: {color}18;
      border: 1px solid {color}44;
      border-radius: 20px;
      padding: 4px 12px;
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: {color};
      margin-top: 8px;
    ">
      <span style="width:6px; height:6px; border-radius:50%; background:{color}; animation: pulse 1.5s infinite;"></span>
      {text}
    </div>
    <style>
    @keyframes pulse {{
      0%, 100% {{ opacity:1; transform: scale(1); }}
      50% {{ opacity:0.5; transform: scale(0.85); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def file_badge(name, pages):
    st.sidebar.markdown(f"""
    <div style="
      background: #141c2e;
      border: 1px solid #1e2d47;
      border-left: 3px solid #00d4aa;
      border-radius: 8px;
      padding: 10px 14px;
      margin: 6px 0;
      font-family: 'DM Sans', sans-serif;
    ">
      <div style="font-size:0.82rem; color:#e8edf5; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">📄 {name}</div>
      <div style="font-size:0.68rem; color:#6b7a99; margin-top:2px; font-family: 'Space Mono', monospace;">{pages} pages indexed</div>
    </div>
    """, unsafe_allow_html=True)

def main_header():
    st.markdown("""
    <div style="
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 32px 0 24px;
      border-bottom: 1px solid #1e2d47;
      margin-bottom: 32px;
    ">
      <div style="
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #00d4aa22, #7b61ff22);
        border: 1px solid #00d4aa44;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
      ">🧠</div>
      <div>
        <h1 style="
          font-family: 'Playfair Display', serif;
          font-size: 1.8rem;
          color: #e8edf5;
          margin: 0;
          letter-spacing: -0.5px;
        ">DocMind Chat</h1>
        <p style="
          font-family: 'Space Mono', monospace;
          font-size: 0.65rem;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: #6b7a99;
          margin: 4px 0 0;
        ">PDF · RAG · Conversational AI</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

def empty_state():
    st.markdown("""
    <div style="
      text-align: center;
      padding: 80px 20px;
      color: #3a4560;
    ">
      <div style="font-size: 4rem; margin-bottom: 16px; opacity: 0.4;">📂</div>
      <div style="
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        color: #6b7a99;
        margin-bottom: 8px;
      ">No documents loaded yet</div>
      <div style="
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #3a4560;
      ">Upload PDFs in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Init State ───────────────────────────────────────────────────────────────
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'chat_history_display' not in st.session_state:
    st.session_state.chat_history_display = []
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False
if 'doc_meta' not in st.session_state:
    st.session_state.doc_meta = []
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# ─── Embeddings ───────────────────────────────────────────────────────────────
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
sidebar_header()

st.sidebar.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)

api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_···")
session_id = st.sidebar.text_input("Session ID", value="default_session", placeholder="my-session")

st.sidebar.markdown('<div class="sidebar-section">Documents</div>', unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# ─── Main Panel ───────────────────────────────────────────────────────────────
main_header()

# ─── Guard: API Key ───────────────────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div style="
      background: linear-gradient(135deg, #1a2236, #141c2e);
      border: 1px solid #1e2d47;
      border-left: 4px solid #7b61ff;
      border-radius: 12px;
      padding: 20px 24px;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.9rem;
      color: #6b7a99;
      line-height: 1.6;
    ">
      <span style="color:#7b61ff; font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;">
        ⚡ Getting Started
      </span><br/><br/>
      Enter your <strong style="color:#e8edf5;">Groq API key</strong> in the sidebar to activate DocMind.
      Then upload one or more PDF files and start asking questions.
      <br/><br/>
      <span style="font-size:0.78rem; color:#3a4560;">
        Get a free key at <code style="color:#00d4aa;">console.groq.com</code>
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Build RAG Pipeline ───────────────────────────────────────────────────────
if uploaded_files and api_key:
    file_names = [f.name for f in uploaded_files]
    cache_key = "_".join(sorted(file_names))

    if cache_key != st.session_state.get("loaded_files_key", ""):
        with st.spinner("🔍 Indexing documents…"):
            embeddings = load_embeddings()
            documents = []
            doc_meta = []

            for uploaded_file in uploaded_files:
                temppdf = f"/tmp/docmind_{uploaded_file.name}"
                with open(temppdf, "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
                doc_meta.append({"name": uploaded_file.name, "pages": len(docs)})

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            st.session_state.doc_meta = doc_meta
            st.session_state.total_chunks = len(splits)
            st.session_state.loaded_files_key = cache_key
            st.session_state.rag_ready = True
            st.session_state.chat_history_display = []

    # Build LLM + chain (always rebuild if key changes)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference context "
         "in the chat history, formulate a standalone question. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are DocMind, an expert assistant for question-answering over documents. "
         "Use the retrieved context to answer accurately. If unsure, say so clearly. "
         "Be concise but complete. Use markdown formatting when helpful.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Sidebar stats
    for meta in st.session_state.doc_meta:
        file_badge(meta["name"], meta["pages"])

    st.sidebar.markdown('<div class="sidebar-section">Index Stats</div>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.metric("Docs", len(st.session_state.doc_meta))
    with c2:
        st.metric("Chunks", st.session_state.total_chunks)

    status_pill("READY", "#00d4aa")

    if st.sidebar.button("🗑  Clear Chat"):
        st.session_state.chat_history_display = []
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]
        st.rerun()

    # ─── Chat Area ────────────────────────────────────────────────────────────
    section_label("Conversation")

    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history_display:
            st.markdown("""
            <div style="
              background: linear-gradient(135deg, #141c2e, #0e1420);
              border: 1px dashed #1e2d47;
              border-radius: 12px;
              padding: 20px 24px;
              margin-bottom: 24px;
              font-family: 'DM Sans', sans-serif;
            ">
              <div style="font-size:0.72rem; font-family:'Space Mono',monospace; letter-spacing:2px; text-transform:uppercase; color:#00d4aa; margin-bottom:10px;">💡 Try asking</div>
              <div style="color:#6b7a99; font-size:0.88rem; line-height:1.8;">
                • <em>"Summarize the key points of this document"</em><br/>
                • <em>"What does it say about [topic]?"</em><br/>
                • <em>"List all [items] mentioned in the PDF"</em>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, msg in enumerate(st.session_state.chat_history_display):
                chat_bubble(msg["role"], msg["content"], index=i)

    # ─── Input Bar ────────────────────────────────────────────────────────────
    section_label("Ask a question")

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
            key="user_input_field",
        )
    with col_btn:
        send = st.button("Send →")

    if (send or user_input) and user_input.strip():
        st.session_state.chat_history_display.append({
            "role": "user", "content": user_input
        })

        with st.spinner(""):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            answer = response["answer"]

        st.session_state.chat_history_display.append({
            "role": "assistant", "content": answer
        })
        st.rerun()

else:
    empty_state()

    if api_key and not uploaded_files:
        status_pill("AWAITING DOCS", "#7b61ff")