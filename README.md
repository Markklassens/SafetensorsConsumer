# SRHN Assembly — Production AI for Real-World Domains

Self-learning AI memory layer + domain applications.
Works on any CPU. No GPU required. Edge-ready.

## What's included

```
srhn_assembly/
├── main.py                    ← Unified CLI entry point
├── __init__.py                ← create_engine / create_app / create_multi_app
├── core/                      ← SRHN v5 engine (all fixes applied)
│   ├── engine.py              ← SRHNEngine + all LLM adapters
│   ├── memory.py              ← SQLite WAL, injection-safe, thread-safe
│   ├── embeddings.py          ← Cluster+TF-IDF, sentence-transformers upgrade
│   ├── lora.py                ← LoRA adapters (10 domains, warm-restart lr)
│   ├── config.py              ← Config / Config.edge() / Config.micro()
│   ├── safetensors_loader.py  ← Header-only shard indexing
│   ├── ingestor.py            ← Progressive ingest with progress bar
│   └── progressive.py         ← ProgressiveSession (checkpoint, resume, bench)
├── api/
│   └── server.py              ← HTTP API (auth, injection guard, semaphore)
├── cli/
│   └── cli.py                 ← Full CLI (load chat ask status progressive...)
└── apps/
    ├── medical/assistant.py   ← Emergency triage decision support
    ├── legal/analyser.py      ← Contract risk analysis
    ├── devops/agent.py        ← Incident response & runbook agent
    └── sports/analyst.py      ← Sports analytics assistant
```

## Quick start

```bash
pip install numpy scipy

# See all domain apps in action (no API key needed)
python main.py demo

# Medical triage (with real LLM)
python main.py medical --llm ollama --model phi3.5

# Legal clause analysis
python main.py legal --llm openai --api-key sk-...

# DevOps incident response
python main.py devops --llm ollama --model llama3.2

# Sports analytics
python main.py sports --llm ollama

# HTTP server with all domains
python main.py server --llm ollama --model phi3.5 --port 8765

# Load a safetensors model
python main.py load /models/llama-3.2-1b/ --name "llama-3.2-1b"

# Progressive shard-by-shard loading (14-shard model)
python cli/cli.py progressive start --shard-dir /models/llama-70b/ -i
```

## Python API

```python
from __init__ import create_engine, create_app, create_multi_app

# Single domain
engine = create_engine(store="./store", llm="ollama", model="phi3.5")
app    = create_app("medical", engine)
result = app.triage("65yo chest pain radiating to jaw, sweating", age=65)
print(result.display())
app.confirm(result.query_id)   # positive feedback
app.correct(result.query_id, "URGENT", "missed red flag")  # correction

# Multiple domains sharing one memory store
apps = create_multi_app(["medical","legal","devops"], store="./store", llm="ollama")
apps["medical"].add_protocol("Chest pain → ECG within 10 minutes.")
apps["devops"].add_runbook("K8s OOM", ["kubectl top pods", "check limits", "scale up"])

# Edge device (Raspberry Pi 4)
engine = create_engine(store="./store", llm="ollama", model="phi3.5", edge=True)
```

## Domain applications

### Medical Triage
- Triage categories: IMMEDIATE / URGENT / SEMI-URGENT / NON-URGENT / REFER
- Pre-loaded: ACS, sepsis, stroke, respiratory, paediatric protocols
- Safety override: keyword-based escalation regardless of LLM output
- Feedback: clinician confirms or corrects → LoRA adapter learns

### Legal Analyser
- Risk levels: HIGH / MEDIUM / LOW / ACCEPTABLE
- Pre-loaded: Employment Rights Act, GDPR, UCTA, Consumer Rights Act, IP law
- Full contract analysis with overall risk summary
- Add local regulations and case precedents

### DevOps Agent
- Severity: P1 / P2 / P3 / P4
- Pre-loaded: CPU, OOM, disk, database, Kubernetes, deployment runbooks
- Learns from resolutions and post-mortems
- Accumulates institutional knowledge over time

### Sports Analytics
- Load season stats, player data, match results
- Analyse form, head-to-head, upcoming matches
- Supports: football, cricket, basketball, tennis

## Deployment

```bash
# systemd (Raspberry Pi)
sudo cp deploy/srhn.service /etc/systemd/system/
sudo systemctl enable srhn && sudo systemctl start srhn

# Docker
docker build -t srhn-assembly:latest .
docker run -d -p 8765:8765 -v ./data:/data -e SRHN_API_KEY=your-key srhn-assembly:latest
```

## API endpoints

All require `X-SRHN-Key` header (set `--srhn-key` or `SRHN_API_KEY` env).

```
POST /query              {"text": "..."}
POST /feedback           {"query_id": N, "reward": 1.0}
POST /learn/fact         {"fact": "...", "domain": "medical"}
POST /learn/workflow     {"name": "...", "steps": [...]}
GET  /status
GET  /health
POST /progressive/start  {"shard_dir": "/models/llama-70b/"}
POST /progressive/next   {"n": 1}
GET  /progressive/status
```
