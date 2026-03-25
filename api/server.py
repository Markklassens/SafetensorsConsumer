"""
SRHN v5 — Production API Server
FIX-1: API key authentication on every non-health endpoint.
FIX-3: ThreadingHTTPServer + per-engine semaphore cap (inherited from engine).
       Additional: request size limit (10MB) prevents OOM from huge uploads.
"""
from __future__ import annotations
import json, os, re, sys, time, threading, traceback, logging, hmac, hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import Config
from core.engine import (SRHNEngine, MockLLM, OllamaLLM,
                          OpenAILLM, AnthropicLLM, HuggingFaceLLM)
from core.safetensors_loader import SafetensorsLoader
from core.progressive import ProgressiveSession, resume_session, _run_benchmark

log = logging.getLogger("srhn.server")

ENGINE: SRHNEngine        = None
LOADER: SafetensorsLoader = None
SESSION: ProgressiveSession = None
_CFG:   Config            = None

MAX_BODY_BYTES = 10 * 1024 * 1024   # 10MB upload limit


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── FIX-1: Auth helpers ───────────────────────────────────────────────────────

def _check_auth(handler: "Handler") -> bool:
    """
    Returns True if request is authenticated.
    If api_key is None/empty in config, auth is DISABLED (dev mode only).
    In production, always set SRHN_API_KEY env var.
    """
    if not _CFG or not _CFG.api_key:
        return True   # auth disabled — warn at startup
    provided = handler.headers.get(_CFG.api_key_header, "")
    # Constant-time compare to prevent timing attacks
    return hmac.compare_digest(provided.strip(), _CFG.api_key)

# Public endpoints that don't require auth
_PUBLIC_PATHS = {"/health", "/", "/ui", "/index.html"}


def parse_multipart(body: bytes, ctype: str) -> dict[str, bytes]:
    m = re.search(r'boundary=([^\s;]+)', ctype)
    if not m: return {}
    boundary = ('--' + m.group(1)).encode()
    parts = {}
    for seg in body.split(boundary)[1:]:
        if seg in (b'--\r\n', b'--\r\n\r\n', b'--'): continue
        seg = seg.lstrip(b'\r\n')
        if b'\r\n\r\n' not in seg: continue
        hdr_raw, data = seg.split(b'\r\n\r\n', 1)
        data = data.rstrip(b'\r\n--')
        hdrs = {}
        for line in hdr_raw.decode('utf-8', errors='replace').split('\r\n'):
            if ':' in line:
                k, v = line.split(':', 1); hdrs[k.strip().lower()] = v.strip()
        cd  = hdrs.get('content-disposition', '')
        fn  = re.search(r'filename="([^"]+)"', cd)
        nm  = re.search(r'name="([^"]+)"',     cd)
        key = (fn.group(1) if fn else nm.group(1)) if (fn or nm) else 'data'
        parts[key] = data
    return parts


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, dict): data = json.dumps(body, indent=2, default=str).encode()
        elif isinstance(body, str): data = body.encode()
        else: data = body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        # FIX: hard cap on body size
        if n > MAX_BODY_BYTES:
            raise ValueError(f"Request body too large ({n} bytes, max {MAX_BODY_BYTES})")
        return self.rfile.read(n) if n else b""

    def _json(self) -> dict:
        try: return json.loads(self._body())
        except: return {}

    def _auth(self) -> bool:
        """FIX-1: check auth, send 401 if fails, return bool."""
        if urlparse(self.path).path in _PUBLIC_PATHS:
            return True
        if not _check_auth(self):
            self._send(401, {"error":"unauthorized",
                             "hint":"Set X-SRHN-Key header"})
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(200)
        for h, v in [("Access-Control-Allow-Origin","*"),
                     ("Access-Control-Allow-Methods","GET,POST,OPTIONS"),
                     ("Access-Control-Allow-Headers",
                      f"Content-Type,{_CFG.api_key_header if _CFG else 'X-SRHN-Key'}")]:
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self):
        if not self._auth(): return
        p  = urlparse(self.path); qs = parse_qs(p.query)
        try:
            path = p.path
            if path in ('/', '/ui', '/index.html'):
                self._send(200, _load_ui(), "text/html; charset=utf-8")
            elif path == '/health':
                self._send(200, {"status":"ok","version":"5.0.0","ts":time.time()})
            elif path == '/status':
                self._send(200, ENGINE.status())
            elif path == '/model/list':
                self._send(200, {"models": LOADER.list_models()})
            elif path == '/memory/stats':
                self._send(200, ENGINE.memory.stats())
            elif path == '/memory/recent':
                kind = qs.get("kind",[None])[0]
                n    = min(int(qs.get("n",[20])[0]), 200)
                self._send(200, {"entries": ENGINE.memory.get_recent(n, kind)})
            elif path == '/progressive/status':
                if SESSION is None:
                    SESSION = ProgressiveSession.resume(ENGINE, LOADER)
                if SESSION is None:
                    self._send(200, {'status': 'no_session', 'message': 'POST /progressive/start first'})
                else:
                    self._send(200, SESSION.status())
            else:
                self._send(404, {"error":"not found"})
        except Exception:
            self._send(500, {"error": traceback.format_exc()})

    def do_POST(self):
        if not self._auth(): return
        p  = urlparse(self.path).path
        ct = self.headers.get("Content-Type","")
        global SESSION
        try:
            if p == '/model/upload':
                if 'multipart/form-data' not in ct:
                    return self._send(400, {"error":"expected multipart"})
                parts   = parse_multipart(self._body(), ct)
                results = []
                for fname, data in parts.items():
                    if not fname.endswith('.safetensors'): continue
                    tmp = str(Path(ENGINE._cfg.store_dir) /
                              f"up_{int(time.time()*1000)}_{fname}")
                    try:
                        Path(tmp).write_bytes(data)
                        ms = LOADER.add_shard(tmp, model_name=Path(fname).stem)
                        arch_facts  = LOADER.get_arch_facts(ms.model_id)
                        layer_facts = LOADER.get_layer_profile(ms.model_id)
                        n_injected  = ENGINE.inject_model_knowledge(arch_facts, layer_facts)
                        results.append({
                            "filename":fname,"model_id":ms.model_id,
                            "name":ms.name,"tensors":len(ms.tensors),
                            "params_M":round(ms.total_params/1e6,2),
                            "arch_facts":arch_facts,"facts_injected":n_injected,
                            "architecture":ms.architecture,
                        })
                    except Exception as e:
                        results.append({"filename":fname,"error":str(e)})
                    finally:
                        try: os.remove(tmp)
                        except: pass
                if not results:
                    return self._send(400, {"error":"no .safetensors files"})
                self._send(200, results[0] if len(results)==1 else {"results":results})
                return

            body = self._json()

            if p == '/query':
                text = str(body.get("text","")).strip()
                if not text: return self._send(400, {"error":"text required"})
                self._send(200, ENGINE.query(text, top_k=int(body.get("top_k",8))))

            elif p == '/feedback':
                result = ENGINE.feedback(
                    int(body.get("query_id",0)),
                    float(body.get("reward",0.0)),
                    str(body.get("note","")))
                self._send(200, result)

            elif p == '/learn/fact':
                fact = str(body.get("fact","")).strip()
                if not fact: return self._send(400, {"error":"fact required"})
                eid = ENGINE.learn_fact(
                    fact, domain=body.get("domain","general"),
                    source=body.get("source",""),
                    confidence=float(body.get("confidence",0.8)))
                self._send(200 if eid else 400,
                           {"entry_id":eid,"status":"ok"} if eid
                           else {"error":"rejected (injection detected)"})

            elif p == '/learn/workflow':
                name  = str(body.get("name","")).strip()
                steps = [str(s) for s in body.get("steps",[]) if str(s).strip()]
                if not name or not steps:
                    return self._send(400, {"error":"name and steps required"})
                eid = ENGINE.learn_workflow(
                    name, steps,
                    domain=body.get("domain","general"),
                    success_rate=float(body.get("success_rate",1.0)))
                self._send(200, {"entry_id":eid,"status":"ok"})

            elif p == '/learn/preference':
                pref = str(body.get("preference","")).strip()
                if not pref: return self._send(400, {"error":"preference required"})
                eid = ENGINE.learn_preference(
                    pref, category=body.get("category","style"))
                self._send(200 if eid else 400,
                           {"entry_id":eid,"status":"ok"} if eid
                           else {"error":"rejected"})

            elif p == '/model/add_shard':
                path = str(body.get("path","")).strip()
                if not path: return self._send(400, {"error":"path required"})
                try:
                    ms = LOADER.add_shard(path, model_id=body.get("model_id"),
                                           model_name=body.get("name",""))
                    arch_facts  = LOADER.get_arch_facts(ms.model_id)
                    layer_facts = LOADER.get_layer_profile(ms.model_id)
                    n = ENGINE.inject_model_knowledge(arch_facts, layer_facts)
                    self._send(200, {**ms.summary(), "arch_facts":arch_facts,
                                     "facts_injected":n})
                except FileNotFoundError as e:
                    self._send(404, {"error":str(e)})

            elif p == '/model/add_dir':
                path = str(body.get("path","")).strip()
                if not path: return self._send(400, {"error":"path required"})
                try:
                    ms = LOADER.add_directory(path, model_id=body.get("model_id"),
                                               model_name=body.get("name",""))
                    arch_facts  = LOADER.get_arch_facts(ms.model_id)
                    layer_facts = LOADER.get_layer_profile(ms.model_id)
                    n = ENGINE.inject_model_knowledge(arch_facts, layer_facts)
                    self._send(200, {**ms.summary(),"arch_facts":arch_facts,
                                     "facts_injected":n})
                except (FileNotFoundError, ValueError) as e:
                    self._send(404, {"error":str(e)})

            elif p == '/memory/search':
                text = str(body.get("text","")).strip()
                if not text: return self._send(400, {"error":"text required"})
                emb  = ENGINE.embed.embed(text)
                kind = body.get("kind"); domain = body.get("domain")
                hits = ENGINE.memory.search(
                    emb, top_k=min(int(body.get("top_k",10)), 50),
                    kind=kind if kind else None,
                    domain=domain if domain else None)
                self._send(200, {"results":hits,"query":text})

            elif p == '/save':
                ENGINE.save()
                self._send(200, {"status":"saved","ts":time.time()})

            elif p == '/progressive/start':
                path = str(body.get('shard_dir', '')).strip()
                if not path: return self._send(400, {'error':'shard_dir required'})
                name  = str(body.get('model_name', ''))
                bench = bool(body.get('benchmark', False))
                try:
                    SESSION = ProgressiveSession.create(
                        shard_dir=path, engine=ENGINE, loader=LOADER,
                        model_name=name, benchmark=bench)
                    self._send(200, SESSION.status())
                except (FileNotFoundError, ValueError) as e:
                    self._send(400, {'error': str(e)})

            elif p == '/progressive/next':
                if SESSION is None:
                    SESSION = ProgressiveSession.resume(ENGINE, LOADER)
                if SESSION is None:
                    return self._send(400, {'error':'No active session'})
                n = int(body.get('n', 1))
                loaded = []
                for _ in range(n):
                    shard = SESSION.next()
                    if shard is None: break
                    from dataclasses import asdict
                    loaded.append(asdict(shard))
                self._send(200, {'loaded': loaded, 'status': SESSION.status()})

            elif p == '/progressive/run':
                if SESSION is None:
                    SESSION = ProgressiveSession.resume(ENGINE, LOADER)
                if SESSION is None:
                    return self._send(400, {'error':'No active session'})
                n = None if body.get('all') else int(body.get('n', 999999))
                SESSION.run(stop_after=n)
                self._send(200, SESSION.status())

            elif p == '/progressive/benchmark':
                if SESSION is None:
                    SESSION = ProgressiveSession.resume(ENGINE, LOADER)
                b = _run_benchmark(ENGINE)
                self._send(200, b)

            elif p == '/progressive/skip':
                if SESSION is None:
                    return self._send(400, {'error':'No active session'})
                idx = int(body.get('index', -1))
                if idx < 0: return self._send(400, {'error':'index required'})
                SESSION.skip(idx)
                self._send(200, SESSION.status())

            else:
                self._send(404, {"error":f"not found: {p}"})

        except ValueError as e:
            self._send(413, {"error": str(e)})
        except Exception:
            log.error(traceback.format_exc())
            self._send(500, {"error":"internal server error"})


def _load_ui() -> bytes:
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return ui_path.read_bytes()
    return _BUILTIN_UI.encode()


# ── Built-in minimal UI (shown when ui.html missing) ─────────────────────────
_BUILTIN_UI = """<!DOCTYPE html>
<html><head><title>SRHN v5</title>
<style>
body{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f8f8f8}
h1{font-size:1.4em;font-weight:600;margin-bottom:4px}
.sub{color:#666;font-size:.9em;margin-bottom:24px}
textarea{width:100%;height:80px;padding:10px;border:1px solid #ddd;border-radius:8px;font-size:14px;resize:vertical}
button{background:#1a1a1a;color:#fff;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:14px}
button:hover{background:#333}
#res{margin-top:20px;background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;min-height:60px;white-space:pre-wrap;font-size:14px}
.meta{font-size:12px;color:#888;margin-top:8px}
input[type=text]{width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:13px;margin-bottom:12px;box-sizing:border-box}
label{font-size:13px;color:#444;display:block;margin-bottom:4px}
.section{margin-bottom:20px}
.err{color:#c00}
</style>
</head><body>
<h1>SRHN v5</h1>
<div class="sub">Private AI memory layer — edge-ready</div>
<div class="section">
  <label>API Key (leave blank if auth disabled)</label>
  <input type="text" id="apiKey" placeholder="Your SRHN API key">
</div>
<div class="section">
  <textarea id="q" placeholder="Ask anything..."></textarea><br><br>
  <button onclick="ask()">Ask</button>
  <button onclick="feedback(1)" style="background:#1a7a1a;margin-left:8px">+1 Good</button>
  <button onclick="feedback(-1)" style="background:#7a1a1a;margin-left:8px">-1 Bad</button>
</div>
<div id="res">Results appear here...</div>
<div class="meta" id="meta"></div>
<script>
let lastId=0;
function hdr(){const k=document.getElementById('apiKey').value;return k?{'Content-Type':'application/json','X-SRHN-Key':k}:{'Content-Type':'application/json'};}
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  document.getElementById('res').textContent='Thinking...';
  document.getElementById('meta').textContent='';
  try{
    const r=await fetch('/query',{method:'POST',headers:hdr(),body:JSON.stringify({text:q})});
    const d=await r.json();
    if(d.error){document.getElementById('res').innerHTML='<span class="err">'+d.error+'</span>';return;}
    lastId=d.query_id;
    document.getElementById('res').textContent=d.response||'(empty response)';
    document.getElementById('meta').textContent=
      'conf='+d.confidence+' | domain='+d.domain+' | mem='+d.memories_used+
      ' | lora='+JSON.stringify(d.lora_active)+' | '+d.elapsed_ms+'ms';
  }catch(e){document.getElementById('res').textContent='Error: '+e;}
}
async function feedback(r){
  if(!lastId)return;
  await fetch('/feedback',{method:'POST',headers:hdr(),body:JSON.stringify({query_id:lastId,reward:r})});
  document.getElementById('meta').textContent+=' | feedback: '+(r>0?'+1':'-1')+'✓';
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))ask();});
</script>
</body></html>"""


def build_llm(args):
    b = getattr(args,'llm','mock').lower()
    if b=='ollama':      return OllamaLLM(model=getattr(args,'model','llama3.2'),
                                          host=getattr(args,'ollama_host','http://localhost:11434'))
    if b=='openai':      return OpenAILLM(api_key=getattr(args,'api_key',''),
                                          model=getattr(args,'model','gpt-4o-mini'),
                                          base_url=getattr(args,'base_url',None))
    if b=='anthropic':   return AnthropicLLM(api_key=getattr(args,'api_key',''),
                                              model=getattr(args,'model','claude-3-5-haiku-20241022'))
    if b=='huggingface': return HuggingFaceLLM(model_name=getattr(args,'model',''))
    return MockLLM()


def main():
    import argparse
    p = argparse.ArgumentParser(description="SRHN v5 Server")
    p.add_argument("--store",       default=os.getenv("SRHN_STORE","srhn_store"))
    p.add_argument("--llm",         default=os.getenv("SRHN_LLM","mock"),
                   choices=["mock","ollama","openai","anthropic","huggingface"])
    p.add_argument("--model",       default=os.getenv("SRHN_MODEL","llama3.2"))
    p.add_argument("--api-key",     default=os.getenv("LLM_API_KEY",""), dest="api_key")
    p.add_argument("--base-url",    default=None, dest="base_url")
    p.add_argument("--ollama-host", default="http://localhost:11434", dest="ollama_host")
    p.add_argument("--port",        type=int, default=int(os.getenv("PORT","8765")))
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--safetensors", default=None)
    p.add_argument("--edge",        action="store_true")
    p.add_argument("--micro",       action="store_true")
    p.add_argument("--srhn-key",    default=os.getenv("SRHN_API_KEY",""),
                   dest="srhn_key",
                   help="API key for SRHN endpoints. Generate one: python -c \"import secrets;print(secrets.token_urlsafe(32))\"")
    p.add_argument("--gen-key",     action="store_true", dest="gen_key",
                   help="Auto-generate and print an API key")
    args = p.parse_args()

    global ENGINE, LOADER, _CFG

    if args.micro:
        cfg = Config.micro()
    elif args.edge:
        cfg = Config.edge()
    else:
        cfg = Config.from_env()

    cfg.store_dir = args.store

    # FIX-1: set API key
    if args.gen_key:
        key = cfg.generate_api_key()
        print(f"\n  Generated API key: {key}")
        print(f"  Use header: X-SRHN-Key: {key}\n")
    elif args.srhn_key:
        cfg.api_key = args.srhn_key
    elif not cfg.api_key:
        print("\n  WARNING: No API key set. Auth is DISABLED.")
        print("  To enable: --srhn-key YOUR_KEY  or  SRHN_API_KEY=YOUR_KEY\n")

    _CFG   = cfg
    llm    = build_llm(args)
    ENGINE = SRHNEngine(cfg, llm)
    LOADER = SafetensorsLoader(cfg, ENGINE.embed)

    if args.safetensors:
        path = args.safetensors
        try:
            ms = LOADER.add_directory(path) if os.path.isdir(path) else LOADER.add_shard(path)
            arch_facts  = LOADER.get_arch_facts(ms.model_id)
            layer_facts = LOADER.get_layer_profile(ms.model_id)
            n = ENGINE.inject_model_knowledge(arch_facts, layer_facts)
            log.info(f"Loaded {ms.name}: {len(ms.tensors)} tensors, {n} facts injected")
        except Exception as e:
            log.error(f"Startup model load: {e}")

    def autosave():
        while True:
            time.sleep(cfg.autosave_interval)
            try: ENGINE.save()
            except Exception as e: log.error(f"Autosave: {e}")

    threading.Thread(target=autosave, daemon=True).start()

    mode  = "micro" if args.micro else ("edge" if args.edge else "standard")
    auth  = f"enabled (header: {cfg.api_key_header})" if cfg.api_key else "DISABLED (dev mode)"
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  SRHN v5 — Production Ready                             ║
╠══════════════════════════════════════════════════════════╣
║  LLM:    {llm.name:<49}║
║  Store:  {args.store:<49}║
║  Mode:   {mode:<49}║
║  Auth:   {auth:<49}║
║  Fixes:  auth✓  injection✓  semaphore✓  sub-clusters✓  ║
║          lora-lr-reset✓  sports/ml domains✓             ║
╚══════════════════════════════════════════════════════════╝

  UI:     http://localhost:{args.port}
  Health: http://localhost:{args.port}/health
  Status: http://localhost:{args.port}/status
""")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving…"); ENGINE.close(); print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
    main()
