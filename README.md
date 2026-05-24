# 🧠 DocMind — Conversational RAG Chat

> Upload PDFs. Ask questions. Get answers grounded in your documents.

A sleek, production-ready Retrieval-Augmented Generation (RAG) app built with **Streamlit**, **LangChain**, **Groq (LLaMA 3)**, and **ChromaDB** — with a fully custom dark UI.

![DocMind Screenshot](assets/preview.png)

---

## ✨ Features

- 📄 **Multi-PDF upload** — load one or many documents at once
- 🔍 **Semantic search** via HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🧠 **History-aware retrieval** — follow-up questions understood in context
- ⚡ **Groq LLaMA 3** — blazing-fast inference
- 💬 **Session management** — multiple isolated chat sessions
- 🎨 **Premium dark UI** — custom CSS, animated chat bubbles, index stats

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/docmind-rag.git
cd docmind-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 API Keys

| Key | Where to get it | Required |
|-----|-----------------|----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ Yes (entered in UI) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Optional |

---

## 🏗 Architecture

```
User Query
    │
    ▼
History-Aware Retriever  ←─── Chat History
    │
    ▼
ChromaDB Vector Search  ←─── HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ▼
Context + Query → Groq LLaMA 3
    │
    ▼
Answer
```

---

## 📁 Project Structure

```
docmind-rag/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── assets/
    └── preview.png         # (add your own screenshot)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS/HTML/JS |
| LLM | Groq · LLaMA 3 8B |
| Embeddings | HuggingFace · all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Parsing | PyPDF |

---

## 🎨 UI Highlights

- **Dark theme** with custom CSS variables
- Animated chat bubbles with role-based styling
- Sidebar with document badges and index stats
- Pulsing status indicators
- Minimal, distraction-free layout

---

## 📝 License

MIT — free to use, fork, and build on.
