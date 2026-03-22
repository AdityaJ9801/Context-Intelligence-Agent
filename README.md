# Agent 01 — Context Intelligence Agent

A standalone FastAPI microservice that acts as the **data ingestion gateway** for a multi-agent architecture. It accepts data from heterogeneous sources, profiles it mathematically, enriches it with LLM-generated semantic types, caches the result in Redis, and exposes a clean REST API for downstream agents.

---

## Architecture at a glance

```
HTTP Request
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FastAPI  (POST /profile  |  POST /profile/stream) │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Redis Cache check  │  ← hit → return immediately
          └──────────┬──────────┘
                     │ miss
          ┌──────────▼──────────┐
          │  Connector + Sampler │  CSV / Parquet / S3 / Postgres …
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   DataProfiler       │  pure Pandas/NumPy, no LLM
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  SemanticEnricher    │  Groq / OpenAI / Anthropic / Ollama
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Redis  set_context  │  TTL = 1 hour (configurable)
          └──────────┬──────────┘
                     │
                ContextObject  →  JSON response
```

---

## Quick start

### Prerequisites
- Docker & Docker Compose
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient)

### 1. Configure environment

```bash
cp .env.free .env
# Open .env and paste your Groq key:
#   GROQ_API_KEY=gsk_...
```

### 2. Start the service

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 3. Profile a CSV file

```bash
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d '{"source": {"type": "local_file", "path": "/app/data/sample.csv", "format": "csv"}}'
```

### 4. Stream progress via SSE

```bash
curl -N -X POST http://localhost:8000/profile/stream \
  -H "Content-Type: application/json" \
  -d '{"source": {"type": "local_file", "path": "/app/data/sample.csv", "format": "csv"}}'
```

---

## Supported data sources

| Source type | Key fields |
|-------------|-----------|
| `local_file` | `path`, `format` (csv / parquet / json) |
| `s3` | `bucket`, `key` (glob patterns supported), `region` |
| `gcs` | `bucket`, `blob` |
| `sftp` | `host`, `username`, `remote_path` |
| `database` | `database_url`, `query` |
| `kafka` | `bootstrap_servers`, `topic` |
| `api` | `url`, `method`, `headers` |

---

## 4096-token ContextObject compression

Every `ContextObject` is automatically kept within a **4 096-token budget** before being sent to the LLM.

The heuristic is `1 token ≈ 4 characters`, so the budget maps to ~16 384 characters of JSON. If the serialised payload exceeds this limit, a progressive halving loop trims the `sample_values` and `top_values` lists on each `ColumnProfile` — cutting them in half per pass — until the payload fits. Convergence is fast: a worst-case object with 50 columns and 100 sample values each reaches budget in ≤ 7 passes.

This ensures the LLM always receives a valid, complete structural summary of the data without hitting context-window limits, regardless of how wide the dataset is.

---

## Running tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## Environment files

| File | Purpose |
|------|---------|
| `.env.free` | Local dev — Groq LLM + DuckDB + Redis (no cloud keys needed beyond Groq) |
| `.env.paid` | Production — full cloud integrations (never commit this file) |
| `.env.example` | Template showing all available variables |

> `.env.free` and `.env.paid` are in `.gitignore` and will never be committed.

---

## Project structure

```
app/
├── cache/          Redis async cache
├── connectors/     Pluggable data source connectors
├── llm/            LLM provider factory + semantic enricher
├── models/         Pydantic data models (DataSource, ContextObject)
├── profilers/      Mathematical schema profiler (no LLM)
├── routers/        FastAPI routers (profile, context)
└── utils/          Smart sampler, pipeline orchestrator
tests/              Pytest suite
```
