# SRHN v5 — Step-by-Step Guide
## Consuming Safetensors Models on Edge Devices

> Works on: Raspberry Pi 4 (2GB+), Jetson Nano, x86 laptop, cloud VMs, M1/M2 Mac.
> Requirements: Python 3.9+, 512MB RAM minimum, numpy, scipy.

---

## What This System Does

SRHN does **not** run the model weights directly.
It does two things:

1. **Reads the model structure** from the `.safetensors` file (architecture, layers, dimensions) and injects that knowledge into a persistent memory store.
2. **Wraps any LLM backend** (Ollama local, OpenAI, Anthropic, HuggingFace) with that memory — so the LLM knows about the model it's running alongside, learns from every interaction, and remembers across sessions.

This means a 7B model's `.safetensors` file (14GB) becomes a 50KB index + architecture facts in memory, and any lightweight LLM (phi3.5 at 2.2GB) can answer questions about it, reason about its capabilities, and improve with feedback.

---

## Part 1 — Install

```bash
# Minimum (works everywhere including Pi 4)
pip install numpy scipy

# Recommended for better embeddings (80MB download, one-time)
pip install sentence-transformers

# For local LLM on CPU (recommended for edge)
# Install Ollama: https://ollama.ai
ollama pull phi3.5          # 2.2GB, runs on 4GB RAM
ollama pull nomic-embed-text # 274MB, real semantic embeddings

# For cloud LLM backends (optional)
pip install openai          # OpenAI / LM Studio / vLLM
pip install anthropic        # Claude

# Clone / download srhn_v5 then:
cd srhn_v5
```

---

## Part 2 — Your First Model Load

### Step 1: Single .safetensors file

```bash
python cli/cli.py load /path/to/model.safetensors
```

You will see:

```
  Ingesting: model.safetensors  (2841.3 MB)
  Tensors in file: 291
  Peak RAM needed: 128.0 MB (streaming one tensor at a time)

  Indexing [████████████████████████████████████████] 100%  done  (3.2s)
  291 tensors indexed

  Architecture facts extracted: 8
    my-model has 1.0B parameters (small (1-4B))
    my-model has 32 transformer layers
    my-model hidden dimension is 2048
    my-model has 32 attention heads
    my-model vocabulary size is 32,000 tokens
    my-model FFN intermediate dimension is 5632
    my-model estimated context length is 8,192 tokens
    my-model uses half-precision (FP16/BF16) weights

  Done in 3.2s — 11 facts in agent memory

  ─── Ingest complete ───
  Model:     my-model (a3f2c1b0)
  Params:    1.00B  (1004M)
  Tensors:   291 across 1 shard(s)
  Size:      2841.3 MB on disk
  Peak RAM:  128.0 MB during ingest
  Injected:  11 facts into agent memory
  Time:      3.2s

  State saved to: srhn_store
```

**Note the Peak RAM: 128MB** — the 2.8GB file was never fully loaded into memory. The loader reads header metadata (a few KB), builds an index of where each tensor lives in the file, then streams individual tensors on demand.

### Step 2: Sharded model directory (e.g. Llama-3.2-1B from HuggingFace)

```bash
# Download model first
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.2-1B',
                  local_dir='./llama-3.2-1b')
"

# Ingest all shards
python cli/cli.py load ./llama-3.2-1b/ --name "llama-3.2-1b"
```

The loader reads `model.safetensors.index.json` to get correct shard order, then processes each shard sequentially. The registry persists between sessions — you only need to ingest once.

### Step 3: Edge device (limited RAM)

```bash
# --edge flag: 192-dim embeddings, 50k memory cap, rank-4 LoRA, 2-thread cap
python cli/cli.py load /path/to/model.safetensors --edge --name "phi-3.5"
```

```bash
# --micro flag: for 512MB RAM devices
python cli/cli.py load /path/to/model.safetensors --micro --name "tinyllama"
```

| Flag       | embed_dim | max_entries | lora_rank | max_concurrent | Peak RAM |
|------------|-----------|-------------|-----------|----------------|----------|
| (default)  | 384       | 500,000     | 8         | 8              | ~200MB   |
| `--edge`   | 192       | 50,000      | 4         | 2              | ~80MB    |
| `--micro`  | 64        | 5,000       | 2         | 1              | ~20MB    |

---

## Part 3 — Interact

### Step 4: Interactive chat

```bash
# With Ollama (local, CPU, free)
python cli/cli.py chat --llm ollama --model phi3.5

# With OpenAI
python cli/cli.py chat --llm openai --api-key sk-... --model gpt-4o-mini

# With Claude
python cli/cli.py chat --llm anthropic --api-key sk-ant-...

# Edge device — Ollama on the same device
python cli/cli.py chat --edge --llm ollama --model phi3.5
```

**Chat commands while inside the session:**

| Command | What it does |
|---------|-------------|
| `/+`    | Positive feedback on last answer — strengthens LoRA adapter |
| `/-`    | Negative feedback — weakens LoRA, stores failure in memory |
| `/status` | Query count, memory size, LoRA health |
| `/save`   | Force save to disk |
| `/quit`   | Save and exit |

**Example session:**

```
  You > How many layers does this model have?
  [model_structure | conf=0.82 | mem=3 | 12ms]

  Based on the loaded model (llama-3.2-1b), it has 32 transformer layers
  with a hidden dimension of 2048 and 32 attention heads. The model uses
  grouped query attention (GQA) and FP16/BF16 weights.

  You > /+

  You > What is the context length?
  [model_structure | conf=0.74 | mem=2 | 8ms]

  The estimated context length for this model is 8,192 tokens...
```

### Step 5: Single question (scriptable)

```bash
# Returns just the answer text
python cli/cli.py ask "What architecture does the loaded model use?"

# Returns full JSON (for automation pipelines)
python cli/cli.py ask "How many attention heads?" --json
```

```bash
# Pipe into other tools
python cli/cli.py ask "Summarise the model capabilities" | tee summary.txt

# Use in shell scripts
CONF=$(python cli/cli.py ask "Is this model quantized?" --json | python -c "import sys,json;print(json.load(sys.stdin)['confidence'])")
echo "Confidence: $CONF"
```

---

## Part 4 — Teach the System

### Step 6: Inject domain knowledge

```bash
# Medical facts
python cli/cli.py learn --fact "Paracetamol reduces fever by inhibiting COX enzymes" \
  --domain medical --confidence 0.95

python cli/cli.py learn --fact "Ibuprofen is contraindicated in peptic ulcer disease" \
  --domain medical --source "BNF 2024" --confidence 0.98

# Legal facts
python cli/cli.py learn --fact "A contract requires offer, acceptance, and consideration" \
  --domain legal --confidence 0.99

# Sports facts
python cli/cli.py learn --fact "The Premier League has 20 clubs playing 38 matches per season" \
  --domain sports --confidence 1.0

# Model-specific facts
python cli/cli.py learn \
  --fact "This llama-3.2-1b model was fine-tuned on medical Q&A from PubMed" \
  --domain model_structure --confidence 0.9
```

### Step 7: Define workflows (procedures the agent remembers)

```bash
python cli/cli.py learn \
  --workflow "Patient triage protocol" \
  --steps \
    "Check vital signs: BP, HR, RR, SpO2, temperature" \
    "Assess level of consciousness using AVPU scale" \
    "Identify chief complaint and onset" \
    "Check for red flags: chest pain, difficulty breathing, altered consciousness" \
    "Assign triage category: immediate/urgent/non-urgent" \
  --domain medical

python cli/cli.py learn \
  --workflow "Contract review checklist" \
  --steps \
    "Verify parties are correctly named and have capacity" \
    "Check governing law and jurisdiction clause" \
    "Review termination conditions and notice periods" \
    "Identify liability caps and indemnity obligations" \
    "Check IP ownership clauses" \
  --domain legal
```

### Step 8: Set user preferences

```bash
python cli/cli.py learn --preference "Always give answers in bullet points"
python cli/cli.py learn --preference "Use metric units (kg, cm, celsius)"
python cli/cli.py learn --preference "Explain technical terms when first used"
```

---

## Part 5 — Run as a Service

### Step 9: HTTP API server

```bash
# Generate an API key first
python -c "import secrets; print(secrets.token_urlsafe(32))"
# → dGhpcyBpcyBhIHNlY3JldCBrZXk...

# Start server with auth
python cli/cli.py server \
  --llm ollama --model phi3.5 \
  --srhn-key YOUR_KEY_HERE \
  --port 8765

# Edge device
python cli/cli.py server \
  --edge --llm ollama --model phi3.5 \
  --srhn-key YOUR_KEY_HERE \
  --port 8765

# Load model at startup
python cli/cli.py server \
  --edge --llm ollama \
  --safetensors /models/llama-3.2-1b/ \
  --srhn-key YOUR_KEY_HERE
```

### Step 10: Query the API

```bash
KEY="your-key-here"
BASE="http://localhost:8765"

# Ask a question
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -H "X-SRHN-Key: $KEY" \
  -d '{"text": "What are the symptoms of appendicitis?"}' | python -m json.tool

# Give feedback
curl -s -X POST $BASE/feedback \
  -H "Content-Type: application/json" \
  -H "X-SRHN-Key: $KEY" \
  -d '{"query_id": 1, "reward": 1.0, "note": "Correct and complete"}'

# Inject a fact
curl -s -X POST $BASE/learn/fact \
  -H "Content-Type: application/json" \
  -H "X-SRHN-Key: $KEY" \
  -d '{"fact": "Appendicitis pain typically starts periumbilical then moves to RLQ",
       "domain": "medical", "confidence": 0.95}'

# Upload a safetensors model
curl -s -X POST $BASE/model/upload \
  -H "X-SRHN-Key: $KEY" \
  -F "model.safetensors=@/path/to/model.safetensors"

# Load from server filesystem
curl -s -X POST $BASE/model/add_dir \
  -H "Content-Type: application/json" \
  -H "X-SRHN-Key: $KEY" \
  -d '{"path": "/models/llama-3.2-1b/"}'

# Check status
curl -s $BASE/status -H "X-SRHN-Key: $KEY" | python -m json.tool

# Search memory
curl -s -X POST $BASE/memory/search \
  -H "Content-Type: application/json" \
  -H "X-SRHN-Key: $KEY" \
  -d '{"text": "fever treatment", "domain": "medical", "top_k": 5}'
```

---

## Part 6 — Python API

### Step 11: Use programmatically

```python
import sys
sys.path.insert(0, "/path/to/srhn_v5")

from core.config import Config
from core.engine import SRHNEngine, OllamaLLM, OpenAILLM
from core.safetensors_loader import SafetensorsLoader
from core.ingestor import ingest_any

# ── Setup ──────────────────────────────────────────────────────────
# Standard
cfg    = Config(store_dir="./my_store")

# Edge device (Raspberry Pi 4)
cfg    = Config.edge()
cfg.store_dir = "./my_store"

# Very constrained device
cfg    = Config.micro()

# ── Build engine ───────────────────────────────────────────────────
llm    = OllamaLLM(model="phi3.5")           # local CPU
# llm = OpenAILLM(api_key="sk-...")          # cloud
# llm = AnthropicLLM(api_key="sk-ant-...")   # Claude

engine = SRHNEngine(cfg, llm)
loader = SafetensorsLoader(cfg, engine.embed)

# ── Optional: upgrade embeddings (one line, requires sentence-transformers) ──
from core.embeddings import try_load_sentence_transformers
try_load_sentence_transformers(engine.embed)  # 80MB, 10× better quality

# ── Load a safetensors model ───────────────────────────────────────
# Single file
result = ingest_any("model.safetensors", engine, loader)

# Sharded directory
result = ingest_any("/models/llama-3.2-1b/", engine, loader,
                    model_name="llama-3.2-1b")

print(result.summary())
# Model:     llama-3.2-1b (a3f2c1b0)
# Params:    1.00B  (1004M)
# Tensors:   291 across 1 shard(s)
# Injected:  11 facts into agent memory

# ── Teach it domain knowledge ──────────────────────────────────────
engine.learn_fact(
    "Paracetamol reduces fever by inhibiting COX-1 and COX-2 enzymes",
    domain="medical", confidence=0.95)

engine.learn_workflow(
    "Appendicitis diagnosis",
    steps=[
        "Check for RLQ tenderness (McBurney's point)",
        "Perform Rovsing's sign test",
        "Order FBC: elevated WBC suggests infection",
        "Order CT abdomen with contrast",
        "Calculate Alvarado score",
    ],
    domain="medical")

engine.learn_preference("Always cite sources when discussing medications")

# ── Query ──────────────────────────────────────────────────────────
result = engine.query("What are the symptoms of appendicitis?")

print(result["response"])
print(f"Confidence:    {result['confidence']}")
print(f"Domain:        {result['domain']}")
print(f"Memories used: {result['memories_used']}")
print(f"Latency:       {result['elapsed_ms']}ms")

# ── Feedback (trains LoRA adapters) ───────────────────────────────
engine.feedback(result["query_id"], reward=+1.0,
                note="Complete and clinically accurate")

engine.feedback(result["query_id"], reward=-1.0,
                note="Missing the important Rovsing sign test")

# ── Direct tensor access (lazy, RAM-bounded) ───────────────────────
ms     = loader.get_model(result_from_ingest.model_id)
tensor = loader.load_tensor(ms.model_id, "model.layers.0.self_attn.q_proj.weight")
# Returns np.ndarray, shape (4096, 4096) — loaded from disk on demand

# Iterate ALL tensors without loading model into RAM
for key, tensor in loader.iter_tensors(ms.model_id):
    print(f"{key}: {tensor.shape}  norm={float(np.linalg.norm(tensor)):.2f}")
    # Process and discard — never holds more than 1 tensor in RAM

# ── Status ─────────────────────────────────────────────────────────
print(engine.status())

# ── Save and restore ───────────────────────────────────────────────
engine.save()         # saves memory DB + LoRA adapters + query count

# Later session:
engine2 = SRHNEngine(cfg, llm)  # automatically loads saved state
loader2 = SafetensorsLoader(cfg, engine2.embed)  # loads registry
# All memories, LoRA adapters, and model index are restored
```

### Step 12: Domain-specific setups

```python
# ── Medical assistant ──────────────────────────────────────────────
cfg = Config.edge() if on_pi else Config()
cfg.store_dir = "./medical_store"
engine = SRHNEngine(cfg, OllamaLLM("phi3.5"))

# Pre-load medical knowledge base
medical_facts = [
    ("Paracetamol: max dose 4g/day in adults, 1g per dose", 0.99),
    ("Ibuprofen contraindicated in renal impairment, peptic ulcer, pregnancy T3", 0.99),
    ("Metformin: first-line for T2DM, hold 48h before contrast", 0.98),
    ("Warfarin interactions: amiodarone, fluconazole, rifampicin", 0.95),
]
for fact, conf in medical_facts:
    engine.learn_fact(fact, domain="medical", confidence=conf)

result = engine.query("Can I give ibuprofen to a patient with CKD stage 3?")

# ── Legal compliance assistant ──────────────────────────────────────
engine = SRHNEngine(Config(), OllamaLLM("llama3.2"))
engine.learn_workflow("GDPR data breach response",
    steps=[
        "Assess scope: what data, how many subjects, which categories",
        "Determine risk to individuals: low/medium/high",
        "If high risk: notify ICO within 72 hours of becoming aware",
        "If high risk to individuals: notify affected data subjects without delay",
        "Document the breach, response, and decisions made",
    ],
    domain="legal")

# ── Sports analytics assistant ─────────────────────────────────────
engine = SRHNEngine(Config(), OllamaLLM("phi3.5"))
engine.learn_fact("Premier League 2024-25: Manchester City, Arsenal, Liverpool top 3", domain="sports")
engine.learn_fact("Champions League format: 36 teams, league phase 8 games each", domain="sports")

# Add custom vocabulary for better sports retrieval
engine.embed.add_domain_terms({
    "wicket": 21, "over": 21, "innings": 21, "century": 21,   # cricket
    "offside": 20, "penalty": 20, "relegation": 20,             # football
    "slam": 20, "fault": 20, "deuce": 20,                       # tennis
})
```

---

## Part 7 — Automation Pipelines

### Step 13: Batch processing

```python
import json, pathlib
from core.ingestor import ingest_any

# Ingest a whole model zoo
model_dir = pathlib.Path("/models")
results   = {}

for model_path in model_dir.iterdir():
    if model_path.is_dir() or model_path.suffix == ".safetensors":
        print(f"Ingesting {model_path.name}...")
        try:
            r = ingest_any(str(model_path), engine, loader, silent=True)
            results[r.model_id] = r.summary()
        except Exception as e:
            results[str(model_path)] = f"ERROR: {e}"

with open("ingest_report.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Step 14: Continuous feedback loop (automation)

```python
# Pattern: process documents, query, collect feedback, improve
import time

documents = load_my_documents()   # your document loader

for doc in documents:
    # Teach facts from document
    for sentence in extract_key_sentences(doc):
        engine.learn_fact(sentence, domain=doc.domain, confidence=0.8)

# Query loop with external feedback signal
while True:
    user_query, user_id = get_next_query()    # your queue/webhook
    result = engine.query(user_query)
    send_to_user(user_id, result["response"]) # your delivery channel

    # Collect feedback from user (e.g. thumbs up/down button)
    feedback = wait_for_feedback(user_id, timeout=30)
    if feedback is not None:
        reward = +1.0 if feedback == "positive" else -1.0
        engine.feedback(result["query_id"], reward=reward)

    time.sleep(0.1)
```

---

## Part 8 — Deployment

### Step 15: systemd service (Raspberry Pi / Linux)

```bash
# /etc/systemd/system/srhn.service
cat > /etc/systemd/system/srhn.service << 'EOF'
[Unit]
Description=SRHN v5 Private AI Memory Layer
After=network.target ollama.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/srhn_v5
Environment=SRHN_STORE=/home/pi/srhn_data
Environment=SRHN_LLM=ollama
Environment=SRHN_MODEL=phi3.5
Environment=SRHN_API_KEY=YOUR_KEY_HERE
Environment=PORT=8765
ExecStart=/usr/bin/python3 cli/cli.py server --edge --llm ollama --model phi3.5
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable srhn
systemctl start srhn
systemctl status srhn
```

### Step 16: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scipy
COPY . .
ENV SRHN_STORE=/data
ENV SRHN_LLM=mock
VOLUME ["/data"]
EXPOSE 8765
CMD ["python", "cli/cli.py", "server", "--host", "0.0.0.0"]
```

```bash
# Build and run
docker build -t srhn:v5 .
docker run -d \
  -v ./srhn_data:/data \
  -e SRHN_API_KEY=your-key \
  -e SRHN_LLM=openai \
  -e LLM_API_KEY=sk-... \
  -p 8765:8765 \
  srhn:v5

# With Ollama sidecar
docker run -d --name ollama ollama/ollama
docker run -d \
  --link ollama:ollama \
  -e SRHN_LLM=ollama \
  -e SRHN_MODEL=phi3.5 \
  -e OLLAMA_HOST=http://ollama:11434 \
  -e SRHN_API_KEY=your-key \
  -p 8765:8765 \
  srhn:v5
```

### Step 17: Load balancing / multiple devices

Each edge device runs its own SRHN instance with its own store. They share facts via the `/learn/fact` endpoint:

```bash
# Push a new fact to all devices
for device in pi1.local pi2.local pi3.local; do
  curl -s -X POST http://$device:8765/learn/fact \
    -H "X-SRHN-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"fact\":\"$FACT\",\"domain\":\"$DOMAIN\"}"
done
```

---

## Part 9 — Troubleshooting

### "No .safetensors files found"

```bash
# Check what's in the directory
ls -la /path/to/model/

# If files are .bin format (old PyTorch), convert first:
pip install transformers safetensors
python -c "
from transformers import AutoModelForCausalLM
import safetensors.torch
model = AutoModelForCausalLM.from_pretrained('/path/to/model')
safetensors.torch.save_file(model.state_dict(), 'model.safetensors')
"
```

### "Peak RAM 4096MB — too high for edge device"

```bash
# Use --edge flag — limits tensors loaded during index
python cli/cli.py load model.safetensors --edge

# Or use --micro for very tight RAM
python cli/cli.py load model.safetensors --micro
```

The actual safetensors file is NEVER fully loaded into RAM. Peak RAM during ingest = size of the largest single tensor, not the model. A 7B model has tensors typically 128-512MB each, so peak RAM is 128-512MB regardless of total model size.

### "LoRA health: cold"

This means the LoRA adapters haven't received enough feedback to start steering retrieval. This is expected on a fresh install. Use `/+` during chat or `POST /feedback` with `reward: 1.0` after good answers. After ~10 positive feedback events the status changes to "learning".

### "Ollama not available"

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve &
ollama pull phi3.5
```

### "Auth disabled warning"

```bash
# Set an API key
export SRHN_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
python cli/cli.py server --edge
```

---

## Quick Reference Card

```
LOAD    python cli/cli.py load /path/to/model.safetensors [--edge] [--name "name"]
CHAT    python cli/cli.py chat [--edge] [--llm ollama] [--model phi3.5]
ASK     python cli/cli.py ask "question" [--json]
STATUS  python cli/cli.py status
LEARN   python cli/cli.py learn --fact "..." --domain medical
SERVER  python cli/cli.py server [--edge] [--llm ollama] [--port 8765]

API ENDPOINTS (all require X-SRHN-Key header except /health):
  POST /query          {"text":"..."}
  POST /feedback       {"query_id":N,"reward":1.0}
  POST /learn/fact     {"fact":"...","domain":"medical"}
  POST /learn/workflow {"name":"...","steps":[...]}
  POST /model/upload   multipart .safetensors upload
  POST /model/add_dir  {"path":"/models/llama-3.2-1b/"}
  GET  /status
  GET  /health
  POST /memory/search  {"text":"...","domain":"medical"}

FEEDBACK IN CHAT:
  /+    positive feedback on last answer
  /-    negative feedback
  /status  engine health
  /save    save to disk
```
