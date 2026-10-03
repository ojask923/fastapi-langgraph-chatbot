# ⚡ Fastapi Langgraph Chatbot

A clean, modern, and lightweight local chatbot built with **FastAPI** and **LangGraph**.

Designed for effortless local execution with zero complex dependencies (no external Docker, PostgreSQL, or Redis required).

---

## ✨ Features

- **🚀 Instant Local Run**: Runs directly on `http://localhost:8000`.
- **🧠 Multi-Provider LLM Support**:
  - **Groq**: Ultra-fast `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`
  - **Ollama**: 100% free local LLMs running privately on your machine (`llama3.2`, `mistral`, `deepseek-r1`, `qwen2.5`).
  - **OpenAI**: `gpt-4o`, `gpt-4o-mini`
  - **Google Gemini**: `gemini-1.5-flash`, `gemini-2.0-flash`
  - **Anthropic**: `claude-3-5-sonnet`
- **💾 Database Persistence (SQLModel / SQLite / PostgreSQL)**:
  - Persistent chat history and session management saved to local `chatbot.db`.
  - Zero-setup SQLite by default, with seamless support for PostgreSQL via `DATABASE_URL`.
- **📚 Retrieval-Augmented Generation (RAG)**:
  - Upload `.txt`, `.pdf`, and `.md` documents via API or UI.
  - Asynchronous, background pipeline for intelligent chunking, deduplication, and vector storage (powered by local Qdrant).
  - Context-aware retrieval for specific sessions/users.
- **🔄 Rolling Memory & Summarization**:
  - Conversation summarization using LangGraph to keep state context within LLM token limits without losing important facts.
- **🛠️ Built-in Agent Tools (LangGraph)**:
  - Math Evaluator (`calculate`)
  - Real-time Date/Time tool (`get_current_time`)
  - Web Search (`search_web`) via DuckDuckGo
- **💬 Real-Time Streaming**: Server-Sent Events (SSE) for token-by-token instant typing.
- **🎨 Glassmorphic Web UI**:
  - Dark / Light mode toggle
  - Code syntax highlighting with copy buttons
  - Chat history session manager synced with database
  - Markdown rendering (tables, bold, lists, code blocks)
  - Export chat history as Markdown

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ojask923/Chatbot.git
cd Chatbot
```

---

### 2. Run the Application

#### Option A: Double-Click (Windows - Easiest)
Simply double click [`run.bat`](run.bat). It will automatically create a `.venv`, install all dependencies, and open `http://localhost:8000` in your browser.

#### Option B: Command Line

```powershell
# 1. (Optional) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install requirements
pip install -r requirements.txt

# 3. Start the server
python start.py
```

Open `http://localhost:8000` in your browser!

---

## ⚙️ Configuration & API Keys

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your preferred provider:

```env
# Default provider: groq | ollama | openai | gemini | anthropic
DEFAULT_PROVIDER=groq
DEFAULT_MODEL=openai/gpt-oss-120b

# Set your API keys (if using cloud providers)
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database URL (defaults to local SQLite database chatbot.db, or postgresql://user:pass@host/db)
DATABASE_URL=sqlite:///./chatbot.db
```

---

## 📁 Project Structure

```text
├── app/
│   ├── agent/
│   │   ├── graph.py            # LangGraph state machine & multi-provider factory
│   │   └── tools.py            # Math, time, and web search tools
│   ├── api/
│   │   ├── routes.py           # FastAPI chat, stream, history & session endpoints
│   │   └── rag_routes.py       # RAG document ingestion & retrieval admin endpoints
│   ├── models/                 # Database Models (SQLModel)
│   │   ├── session.py          # ChatSession table
│   │   └── message.py          # ChatMessage table
│   ├── services/               # Services Layer
│   │   ├── database.py         # Database connection & CRUD operations
│   │   ├── memory.py           # Mem0 user preferences & long-term memory
│   │   ├── rag_service.py      # Document parsing, chunking, and vector embedding
│   │   ├── summarizer.py       # Rolling conversation compactor for AgentState
│   │   ├── context_engine.py   # Advanced context retrieval logic
│   │   ├── query_rewriter.py   # Query rewriting for optimal vector search
│   │   └── qdrant_shared.py    # Qdrant client connection handling
│   ├── static/                 # Modern Web Chat UI
│   │   ├── index.html          # HTML Layout
│   │   ├── style.css           # Glassmorphism styling & animations
│   │   └── app.js              # SSE streaming & database-synced session manager
│   └── config.py               # Pydantic environment configuration
├── chatbot.db                  # Local SQLite database file (auto-generated)
├── main.py                     # FastAPI app initialization & database lifespan
├── start.py                    # Launcher script with auto browser popup
├── run.bat                     # Windows 1-click batch launcher
├── requirements.txt            # Dependencies
└── .env                        # Environment config
```
