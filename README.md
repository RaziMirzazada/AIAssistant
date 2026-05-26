# AI Assistant V3 — Advanced Edition

A production-ready, fully asynchronous RAG assistant built for **RunPod Cloud GPU** deployment, designed to scale to **200+ ingested books** with disk-persisted vector storage, an Azerbaijani-first multilingual UI (AZ / EN / RU), and a hardened security perimeter.

## Highlights

| Capability | Details |
|---|---|
| **Workspace isolation** | All runtime paths anchored to `/workspace/ai-assistant-v3/` |
| **Disk-persisted RAG** | `chromadb.PersistentClient` + `BM25Okapi` rebuilt from disk on boot |
| **Hybrid retrieval** | Cosine similarity (65%) + BM25 (35%), top-10 chunks |
| **Local embeddings** | Ollama `nomic-embed-text` — documents never leave the box |
| **Three response modes** | 🔒 Offline (Ollama `qwen2.5:14b`) · 🌐 Online (Gemini 2.5 Flash / Grok-2) · 🤫 Secret (local Fernet/AES-256 + pseudonymisation, then cloud) |
| **AZ↔EN bridge** | UI in Azerbaijani / English / Russian; pipeline runs on canonical English for peak accuracy |
| **Conversational memory** | Sliding window of the last 5 turns per session |
| **Hardened API** | `X-API-KEY` gatekeeper on every endpoint, explicit CORS allow-list |
| **Streaming UI** | NDJSON over HTTP, blinking cursor, abort controller, "Save to KB" recycle |

## File layout

```
ai-assistant-v3/
├── config.py                # Pydantic settings, workspace paths, secrets
├── main.py                  # FastAPI app, AZ↔EN router, streaming chat
├── frontend.html            # Single-file GitHub-dark UI (AZ / EN / RU)
├── requirements.txt
├── .env.example
└── services/
    ├── crypto.py            # Fernet (AES-256) + pseudonymisation pipeline
    ├── llm.py               # Unified async streamer (Ollama / Gemini / Grok) + memory
    └── rag.py               # HybridSearchEngine: ChromaDB + BM25, recursive splitter
```

## Deploy to RunPod

```bash
mkdir -p /workspace/ai-assistant-v3 && cd /workspace/ai-assistant-v3
git clone <this-repo> .

# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull local models (Ollama must be running)
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 3. Configure secrets
cp .env.example .env
vi .env   # fill in API_KEY, GEMINI_API_KEY, GROK_API_KEY

# 4. Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://<runpod-host>:8000/` and enter your `X-API-KEY` in the top-right.

## API

All endpoints require an `X-API-KEY` header.

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/api/health`              | Engine stats + Ollama / Gemini / Grok probes |
| `POST`   | `/api/chat`                | NDJSON streaming chat (AZ→EN→retrieve→LLM→EN→AZ) |
| `POST`   | `/api/sources/upload`      | Multipart PDF / TXT / MD upload |
| `POST`   | `/api/sources/url`         | Fetch + clean + index a URL |
| `POST`   | `/api/sources/text`        | Raw text ingestion |
| `GET`    | `/api/sources`             | List indexed sources |
| `DELETE` | `/api/sources/{id}`        | Remove a source and all its chunks |
| `POST`   | `/api/session/{id}/reset`  | Clear conversational memory |

## Secret Mode pipeline

```
user input
   │
   ▼
pseudonymise (emails, phones, IBAN, IPv4, names, custom keywords)
   │
   ▼
Fernet (AES-256) round-trip integrity check  ← key lives only in process RAM
   │
   ▼
cloud LLM (Gemini / Grok) sees only masked placeholders
   │
   ▼
streamed tokens → local rehydration → user
```

The Fernet key is generated on boot and never written to disk. Restarting the process rotates it.

## License

Private / internal — set as you see fit.
