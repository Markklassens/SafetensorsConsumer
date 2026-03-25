#!/usr/bin/env python3
"""
SRHN v5 — CLI
=============
Commands: load  chat  ask  status  models  learn  export  server

Usage:
  python cli.py load /models/llama-3.2-1b/
  python cli.py chat --llm ollama --model phi3.5
  python cli.py ask "What is paracetamol used for?"
  python cli.py status
  python cli.py learn --fact "Ibuprofen is an NSAID" --domain medical
  python cli.py server --edge --llm ollama --port 8765
"""
from __future__ import annotations
import argparse, json, os, sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.engine import SRHNEngine, MockLLM, OllamaLLM, OpenAILLM, AnthropicLLM
from core.safetensors_loader import SafetensorsLoader
from core.ingestor import ingest_any
from core.progressive import ProgressiveSession, resume_session, interactive_session


def _build_cfg(args) -> Config:
    if getattr(args, "micro", False):
        cfg = Config.micro()
    elif getattr(args, "edge", False):
        cfg = Config.edge()
    else:
        cfg = Config.from_env()
    if hasattr(args, "store") and args.store:
        cfg.store_dir = args.store
    if hasattr(args, "srhn_key") and args.srhn_key:
        cfg.api_key = args.srhn_key
    return cfg


def _build_llm(args):
    b = getattr(args, "llm", "mock").lower()
    if b == "ollama":
        return OllamaLLM(model=getattr(args,"model","llama3.2"),
                         host=getattr(args,"ollama_host","http://localhost:11434"))
    if b == "openai":
        return OpenAILLM(api_key=getattr(args,"api_key",""),
                         model=getattr(args,"model","gpt-4o-mini"),
                         base_url=getattr(args,"base_url",None))
    if b == "anthropic":
        return AnthropicLLM(api_key=getattr(args,"api_key",""),
                            model=getattr(args,"model","claude-3-5-haiku-20241022"))
    return MockLLM()


def _build(args):
    cfg    = _build_cfg(args)
    engine = SRHNEngine(cfg, _build_llm(args))
    loader = SafetensorsLoader(cfg, engine.embed)
    return engine, loader


def _pj(obj):
    print(json.dumps(obj, indent=2, default=str))


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_load(args):
    engine, loader = _build(args)
    print(f"\n  SRHN v5 — Load\n  Store: {engine._cfg.store_dir}\n  Path:  {args.path}\n")
    try:
        result = ingest_any(args.path, engine, loader,
                            model_name=getattr(args,"name",""),
                            silent=getattr(args,"silent",False))
        print("\n  ─── Ingest complete ───")
        print(result.summary())
        if getattr(args,"json",False):
            _pj({"model_id":result.model_id,"params_B":result.params_B,
                 "tensors":result.tensors,"facts":result.arch_facts,
                 "elapsed_s":result.elapsed_s,"warnings":result.warnings})
        engine.save()
        print(f"\n  State saved to: {engine._cfg.store_dir}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}"); sys.exit(1)
    finally:
        engine.close()


def cmd_chat(args):
    engine, loader = _build(args)
    cfg = engine._cfg
    print(f"\n  SRHN v5 — Chat  |  LLM: {engine.llm.name}  |  Store: {cfg.store_dir}")
    print(f"  Memory: {engine.memory.stats()['total']} entries")
    print(f"  Commands: /quit  /status  /save  /+  /-\n")
    last_qid = None; last_q = ""
    while True:
        try:
            user = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving..."); engine.save(); engine.close(); break
        if not user: continue
        if user.lower() in ("/quit","/exit","quit","exit"):
            engine.save(); engine.close(); break
        if user == "/status":
            s = engine.status()
            print(f"  q={s['query_count']} mem={s['memory']['total']} "
                  f"lora={s['lora']['total_adapters']} B_max={s['lora'].get('B_max',0):.4f}")
            continue
        if user == "/save":
            engine.save(); print("  Saved."); continue
        if user in ("/+","+1") and last_qid:
            engine.feedback(last_qid, +1.0, last_q)
            print(f"  +1 feedback on query {last_qid}"); continue
        if user in ("/-","-1") and last_qid:
            engine.feedback(last_qid, -1.0, last_q)
            print(f"  -1 feedback on query {last_qid}"); continue
        r = engine.query(user, top_k=8)
        last_qid = r["query_id"]; last_q = user
        if r.get("error") == "invalid_query":
            print(f"\n  [BLOCKED] Query contains disallowed patterns.\n"); continue
        print(f"\n  [{r['domain']} | conf={r['confidence']:.2f} | "
              f"mem={r['memories_used']} | {r['elapsed_ms']}ms]\n")
        print(f"  {r['response']}\n")
        if r["query_id"] % 20 == 0: engine.save()


def cmd_ask(args):
    engine, loader = _build(args)
    r = engine.query(args.question, top_k=6)
    if getattr(args,"json",False):
        _pj(r)
    else:
        print(r["response"])
    if r.get("error"): sys.exit(1)
    engine.save(); engine.close()


def cmd_status(args):
    engine, loader = _build(args)
    s = engine.status(); models = loader.list_models()
    if getattr(args,"json",False):
        _pj({"engine":s,"models":models}); engine.close(); return
    print(f"\n  SRHN v5 — Status")
    print(f"  Version:   {s['version']}")
    print(f"  LLM:       {s['llm']['name']} ({'ok' if s['llm']['available'] else 'unavailable'})")
    print(f"  Embed:     {s['embed']['engine']} dim={s['embed']['dim']}")
    print(f"  Queries:   {s['query_count']}")
    print(f"  Store:     {s['config']['store_dir']}")
    print(f"\n  Memory breakdown:")
    for k, v in s["memory"].get("by_kind",{}).items():
        print(f"    {k:<15} {v:>6}")
    print(f"    {'TOTAL':<15} {s['memory']['total']:>6}")
    print(f"\n  LoRA adapters: {s['lora']['total_adapters']}  |  "
          f"domains: {', '.join(s['lora']['domains']) or 'none'}")
    print(f"  LoRA B_max:    {s['lora'].get('B_max',0):.4f}  "
          f"({'learning' if s['lora'].get('B_max',0)>0.001 else 'cold — needs feedback'})")
    print(f"  Concurrency:   {s['concurrent']['free']}/{s['concurrent']['max']} free")
    if models:
        print(f"\n  Loaded models ({len(models)}):")
        for m in models:
            print(f"    {m['model_id'][:8]}  {m['name']:<30} {m['params_M']:.0f}M")
    else:
        print(f"\n  No models loaded.  Run: python cli.py load <path>")
    engine.close()


def cmd_models(args):
    engine, loader = _build(args)
    models = loader.list_models()
    if getattr(args,"json",False):
        _pj(models); engine.close(); return
    if not models:
        print("\n  No models loaded.\n  Load: python cli.py load /path/to/model.safetensors\n")
    else:
        print(f"\n  Loaded models ({len(models)}):")
        for m in models:
            print(f"\n  {m['model_id']}  {m['name']}")
            print(f"    Params:  {m['params_M']:.0f}M  |  {m['size_class']}")
            print(f"    Arch:    {m.get('n_layers','?')} layers, "
                  f"hidden={m.get('hidden_dim','?')}, heads={m.get('n_heads','?')}")
            print(f"    Shards:  {m['shards']}  |  Quantized: {'yes' if m.get('quantized') else 'no'}")
    engine.close()


def cmd_learn(args):
    engine, loader = _build(args)
    if args.fact:
        eid = engine.learn_fact(args.fact,
                                domain=getattr(args,"domain","general"),
                                source=getattr(args,"source",""),
                                confidence=float(getattr(args,"confidence",0.8)))
        print(f"  {'Stored: '+eid if eid else 'Rejected (injection pattern detected)'}")
    elif args.workflow:
        steps = args.steps or []
        if not steps:
            print("  Error: --steps required"); engine.close(); sys.exit(1)
        eid = engine.learn_workflow(args.workflow, steps,
                                    domain=getattr(args,"domain","general"))
        print(f"  Workflow stored: {eid}")
    elif args.preference:
        eid = engine.learn_preference(args.preference)
        print(f"  {'Stored: '+eid if eid else 'Rejected'}")
    else:
        print("  Provide --fact, --workflow, or --preference")
    engine.save(); engine.close()


def cmd_export(args):
    engine, loader = _build(args)
    export = {
        "exported_at": time.time(),
        "engine":      engine.status(),
        "models":      loader.list_models(),
        "recent_memory": engine.memory.get_recent(50),
    }
    if args.out:
        with open(args.out,"w") as f: json.dump(export, f, indent=2, default=str)
        print(f"  Exported to: {args.out}")
    else:
        _pj(export)
    engine.close()


def cmd_server(args):
    server_path = Path(__file__).parent.parent / "api" / "server.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("server", str(server_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ── Parser ────────────────────────────────────────────────────────────────────

def _common(p):
    p.add_argument("--store",    default=os.getenv("SRHN_STORE","srhn_store"))
    p.add_argument("--edge",     action="store_true")
    p.add_argument("--micro",    action="store_true")
    p.add_argument("--srhn-key", default=os.getenv("SRHN_API_KEY",""), dest="srhn_key")


def _llm(p):
    p.add_argument("--llm",      default=os.getenv("SRHN_LLM","mock"),
                   choices=["mock","ollama","openai","anthropic"])
    p.add_argument("--model",    default=os.getenv("SRHN_MODEL","llama3.2"))
    p.add_argument("--api-key",  default=os.getenv("LLM_API_KEY",""), dest="api_key")
    p.add_argument("--base-url", default=None, dest="base_url")
    p.add_argument("--ollama-host", default="http://localhost:11434", dest="ollama_host")


def main():
    root = argparse.ArgumentParser(prog="srhn",
        description="SRHN v5 — Private AI memory layer")
    sub = root.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load",    help="Ingest safetensors file or directory")
    _common(p)
    p.add_argument("path"); p.add_argument("--name",default="")
    p.add_argument("--silent",action="store_true"); p.add_argument("--json",action="store_true")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("chat",    help="Interactive chat")
    _common(p); _llm(p); p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask",     help="Single question")
    _common(p); _llm(p)
    p.add_argument("question"); p.add_argument("--json",action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("status",  help="Show status")
    _common(p); p.add_argument("--json",action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("models",  help="List models")
    _common(p); p.add_argument("--json",action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("learn",   help="Inject knowledge")
    _common(p)
    p.add_argument("--fact",default=""); p.add_argument("--workflow",default="")
    p.add_argument("--steps",nargs="+"); p.add_argument("--preference",default="")
    p.add_argument("--domain",default="general"); p.add_argument("--source",default="")
    p.add_argument("--confidence",type=float,default=0.8)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("export",  help="Export state as JSON")
    _common(p); p.add_argument("--out",default="")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("server",  help="Start HTTP API server")
    _common(p); _llm(p)
    p.add_argument("--port",type=int,default=int(os.getenv("PORT","8765")))
    p.add_argument("--host",default="0.0.0.0")
    p.add_argument("--safetensors",default=None)
    p.add_argument("--gen-key",action="store_true",dest="gen_key")
    p.set_defaults(func=cmd_server)

    _add_progressive_parser(sub)
    args = root.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(name)s] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()


# ── Progressive session commands (added for progressive shard loading) ────────

def cmd_progressive(args):
    """
    Progressive shard-by-shard ingest with interactive testing.
    Subcommands: start  resume  status  next  run  interactive
    """
    from core.progressive import ProgressiveSession, resume_session, interactive_session

    engine, loader = _build(args)
    sub = getattr(args, "prog_sub", "interactive")

    if sub == "start":
        if not args.shard_dir:
            print("  --shard-dir required"); engine.close(); sys.exit(1)
        session = ProgressiveSession.create(
            shard_dir=args.shard_dir,
            engine=engine, loader=loader,
            model_name=getattr(args,"name",""),
            benchmark=getattr(args,"benchmark",False))
        print(f"  Session created: {session.state.session_id}")
        print(f"  Shards: {session.state.total_shards}")
        if getattr(args,"interactive",False):
            interactive_session(session)
        else:
            session.print_status()
        engine.close()

    elif sub == "resume":
        session = resume_session(engine, loader,
                                  benchmark=getattr(args,"benchmark",False))
        if session is None:
            engine.close(); sys.exit(1)
        if getattr(args,"interactive",False):
            interactive_session(session)
        else:
            remaining = session.state.pending_count
            if remaining > 0:
                print(f"  Loading {remaining} remaining shards...")
                session.run()
            session.print_status()
        engine.save(); engine.close()

    elif sub == "status":
        session = ProgressiveSession.resume(engine, loader)
        if session is None:
            print("  No active session found."); engine.close(); return
        session.print_status()
        engine.close()

    elif sub == "next":
        session = ProgressiveSession.resume(engine, loader,
                                             benchmark=getattr(args,"benchmark",False))
        if session is None:
            engine.close(); sys.exit(1)
        n = getattr(args,"n",1)
        for _ in range(n):
            shard = session.next()
            if shard is None: break
            print(f"  ✓ Shard {shard.index+1}: {shard.tensors} tensors, "
                  f"{shard.params_M:.0f}M params, {shard.elapsed_s:.1f}s")
        session.print_status()
        session.save_engine(); engine.close()

    elif sub == "run":
        session = ProgressiveSession.resume(engine, loader,
                                             benchmark=getattr(args,"benchmark",False))
        if session is None:
            engine.close(); sys.exit(1)
        n = None if getattr(args,"all",False) else getattr(args,"n",None)
        session.run(stop_after=n)
        session.print_status()
        session.save_engine(); engine.close()

    elif sub == "interactive":
        # Try resume first, else need --shard-dir
        session = ProgressiveSession.resume(engine, loader,
                                             benchmark=getattr(args,"benchmark",False))
        if session is None:
            if not getattr(args,"shard_dir",""):
                print("  No active session. Use: python cli.py progressive start --shard-dir /path/")
                engine.close(); return
            session = ProgressiveSession.create(
                shard_dir=args.shard_dir, engine=engine, loader=loader,
                model_name=getattr(args,"name",""),
                benchmark=getattr(args,"benchmark",False))
        interactive_session(session)
        session.save_engine(); engine.close()

    else:
        print(f"  Unknown subcommand: {sub}")
        engine.close()


def _add_progressive_parser(sub):
    p = sub.add_parser("progressive",
        help="Progressive shard-by-shard loading with testing",
        aliases=["prog"],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new session (interactive mode)
  python cli.py progressive start --shard-dir /models/llama-70b/ --interactive

  # Start, load all shards automatically with benchmarks
  python cli.py progressive start --shard-dir /models/llama-70b/ --benchmark
  python cli.py progressive run --all

  # Load one shard at a time (manual control)
  python cli.py progressive start --shard-dir /models/llama-70b/
  python cli.py progressive next          # load shard 1
  python cli.py progressive next          # load shard 2
  # ... test, query, check status ...
  python cli.py progressive run --n 5    # load next 5 shards

  # Resume an interrupted session
  python cli.py progressive resume --interactive

  # Check session status
  python cli.py progressive status
        """)
    _common(p); _llm(p)
    p.add_argument("prog_sub", nargs="?", default="interactive",
                   choices=["start","resume","status","next","run","interactive"])
    p.add_argument("--shard-dir",   default="", dest="shard_dir")
    p.add_argument("--name",        default="")
    p.add_argument("--benchmark",   action="store_true",
                   help="Run quality probe before/after each shard")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="Drop into interactive REPL after setup")
    p.add_argument("--n",           type=int, default=1,
                   help="Number of shards to load (for 'next' and 'run')")
    p.add_argument("--all",         action="store_true",
                   help="Load all remaining shards (for 'run')")
    p.set_defaults(func=cmd_progressive)
    return p
