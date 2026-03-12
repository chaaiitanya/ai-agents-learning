# Multi-Agent AI Platform

A production-ready AI orchestration platform that automates complex tasks through a pipeline of specialized agents — Researcher, Coder, and Reviewer — with RAG-powered knowledge retrieval, human-in-the-loop checkpoints, and full session persistence.

---

## How It Works

```
Task Input
    │
    ▼
[Researcher] → Analyzes task, retrieves RAG context, synthesizes findings
    │
    ▼
[Coder] → Implements solution based on research
    │
    ▼
[Reviewer] → Evaluates code quality, correctness, security
    │
    ├── APPROVE → [Human Checkpoint] → Final Output
    └── REVISE  → [Coder] (loops up to max_review_cycles)
```

---

## Agents

| Agent | Role |
|-------|------|
| **Researcher** | Analyzes the task, queries the RAG knowledge base, and synthesizes structured findings |
| **Coder** | Implements a complete, production-quality solution from the research output |
| **Reviewer** | Evaluates the code for correctness, security, and completeness — decides to approve or request revisions |
| **Human Checkpoint** | Optional pause before finalization; accepts feedback to restart the pipeline |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph (state machine with conditional routing) |
| **LLM Providers** | Anthropic Claude (default), OpenAI GPT |
| **API** | FastAPI (async, auto-docs at `/docs`) |
| **Vector DB** | ChromaDB (local persistent store) |
| **Embeddings** | OpenAI `text-embedding-3-small` / HuggingFace `all-MiniLM-L6-v2` (fallback) |
| **Session DB** | SQLite via SQLAlchemy (async) |
| **CLI** | Typer + Rich |
| **Deployment** | Docker + Docker Compose |

---

## Project Structure

```
multi-agent-platform/
├── main.py                  # FastAPI app entry point
├── cli.py                   # CLI interface
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── agents/
│   ├── state.py             # AgentState TypedDict (shared state schema)
│   ├── graph.py             # LangGraph workflow definition & routing
│   ├── llm.py               # LLM factory (Anthropic / OpenAI)
│   ├── researcher.py        # Researcher agent node
│   ├── coder.py             # Coder agent node
│   └── reviewer.py          # Reviewer agent node + decision parsing
│
├── api/
│   ├── routes.py            # FastAPI route handlers
│   └── schemas.py           # Pydantic request/response models
│
├── db/
│   ├── models.py            # SQLAlchemy ORM (Session, Message, Checkpoint)
│   └── database.py          # DB init, session repo
│
├── rag/
│   ├── embeddings.py        # Embedding model factory
│   ├── vector_store.py      # ChromaDB wrapper
│   └── retriever.py         # RAG retriever (semantic search)
│
├── data/                    # Created at runtime
│   ├── sessions.db          # SQLite session store
│   └── chroma/              # ChromaDB vector store
│
└── tests/
    └── test_graph.py
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- An Anthropic or OpenAI API key

### Local Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/chaaiitanya/ai-agents-learning
cd multi-agent-platform
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 3. Start the API server
python main.py
# Server runs at http://localhost:8000
# API docs at  http://localhost:8000/docs
```

### Docker

```bash
docker-compose up -d
# Server runs at http://localhost:8000
```

---

## Environment Variables

```env
# LLM Provider (required — pick one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Model configuration
LLM_PROVIDER=anthropic                   # anthropic | openai
LLM_MODEL=claude-sonnet-4-6             # model identifier
EMBEDDING_MODEL=text-embedding-3-small  # for OpenAI embeddings

# Storage paths
SQLITE_DB_PATH=./data/sessions.db
CHROMA_PERSIST_DIR=./data/chroma

# API server
API_HOST=0.0.0.0
API_PORT=8000

# Agent behavior
MAX_REVIEW_CYCLES=3              # max Coder→Reviewer loops before forced approval
HUMAN_CHECKPOINT_ENABLED=true    # enable human-in-the-loop pauses
```

---

## API Reference

**Base URL:** `http://localhost:8000/api/v1`

### Run a Task

```http
POST /run
Content-Type: application/json

{
  "task": "Build a Python REST API with authentication and rate limiting",
  "max_review_cycles": 3
}
```

**Response:**

```json
{
  "session_id": "abc-123",
  "status": "paused | completed",
  "awaiting_human": true,
  "research_output": "...",
  "code_output": "...",
  "review_output": "...",
  "review_cycle": 1,
  "final_output": "..."
}
```

### Submit Human Feedback

```http
POST /feedback
Content-Type: application/json

{
  "session_id": "abc-123",
  "feedback": "Add input validation and better error messages"
}
```

### Ingest Knowledge (Text)

```http
POST /ingest/text
Content-Type: application/json

{
  "text": "Your documentation or reference material here...",
  "metadata": { "source": "internal-docs" }
}
```

### Ingest Knowledge (File)

```http
POST /ingest/file
Content-Type: multipart/form-data

file: <upload PDF, CSV, or TXT>
```

### Health Check

```http
GET /health
→ { "status": "ok" }
```

---

## CLI Usage

### Run a Task

```bash
python cli.py run "Build a FastAPI service with JWT authentication"

# Limit review iterations
python cli.py run "Write a binary search implementation" --max-cycles 2
```

The CLI displays each agent's output in formatted panels and prompts for human feedback at checkpoints.

### Ingest Knowledge

```bash
# Add raw text
python cli.py ingest "REST APIs use HTTP methods like GET, POST, PUT, DELETE..."

# Add a file (PDF, CSV, or TXT)
python cli.py ingest path/to/docs.pdf --file
```

---

## Database

### SQLite (`data/sessions.db`)

| Table | Purpose |
|-------|---------|
| `sessions` | Tracks task metadata, status, and timestamps |
| `messages` | Stores the full message history per session |
| `checkpoints` | Records workflow pause points and human feedback |

### ChromaDB (`data/chroma/`)

Persists document embeddings for semantic retrieval. Populated via the `/ingest` endpoints or CLI. The Researcher agent queries this on every run.

---

## Running Tests

```bash
pytest tests/
```

---

## Example Workflow

```bash
# 1. Ingest relevant documentation into the knowledge base
python cli.py ingest "FastAPI supports async route handlers, dependency injection, and auto-generated OpenAPI docs."

# 2. Run a task — agents will use the knowledge base for context
python cli.py run "Create a FastAPI service with user authentication, rate limiting, and auto-generated docs"

# Output:
# [Researcher] Findings: ...
# [Coder]      Solution: ...
# [Reviewer]   Decision: APPROVE / REVISE
# [Checkpoint] Awaiting feedback... (press Enter to skip)
```
