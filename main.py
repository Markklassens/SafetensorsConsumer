#!/usr/bin/env python3
"""
SRHN Assembly — Main CLI
=========================
Run any domain app from the command line.

Commands:
  medical   Interactive medical triage session
  legal     Legal clause analysis
  devops    DevOps incident response
  sports    Sports analytics
  server    HTTP API server with all domain apps
  demo      Run all domain demos with mock LLM
  load      Load safetensors model(s)

Quick start (no API key needed):
  python main.py demo
  python main.py medical --llm ollama --model phi3.5
  python main.py server --apps medical,legal,devops --llm ollama
"""
import argparse, os, sys, json, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def _build_engine(args):
    from srhn_assembly import create_engine
    return create_engine(
        store=getattr(args,"store","./srhn_store"),
        llm=getattr(args,"llm","mock"),
        model=getattr(args,"model",""),
        api_key=getattr(args,"api_key",""),
        edge=getattr(args,"edge",False),
        micro=getattr(args,"micro",False),
    )


def cmd_demo(args):
    """Run all domain demos with mock LLM to show capabilities."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    # Import from local __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "assembly_init", os.path.join(os.path.dirname(__file__), "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    create_multi_app = mod.create_multi_app
    import tempfile

    print("\n" + "═"*62)
    print("  SRHN Assembly — Real-World Application Demo")
    print("  All domains, MockLLM (swap --llm ollama for real answers)")
    print("═"*62)

    with tempfile.TemporaryDirectory() as td:
        apps = create_multi_app(
            ["medical","legal","devops","sports"],
            store=td, llm="mock")

        # ── Medical ──────────────────────────────────────────────────
        print("\n  ── MEDICAL TRIAGE ──────────────────────────────────")
        med = apps["medical"]
        cases = [
            ("65yo male, chest pain radiating to jaw, sweating, onset 20min", 65,
             {"hr":110,"bp_sys":95,"spo2":94,"rr":22}),
            ("28yo female, mild headache, no fever, onset 2 days",              28, {}),
            ("45yo, sudden severe headache 'worst of my life', vomiting",       45, {}),
        ]
        for symptoms, age, vitals in cases:
            r = med.triage(symptoms, age=age, vitals=vitals)
            icon = {"IMMEDIATE":"🔴","URGENT":"🟡","SEMI-URGENT":"🟢","NON-URGENT":"⚪"}.get(r.category,"⚪")
            print(f"  {icon} [{r.category}] {symptoms[:70]}")
            print(f"       conf={r.confidence:.0%} mem={r.memories_used} {r.elapsed_ms:.0f}ms")

        # ── Legal ─────────────────────────────────────────────────────
        print("\n  ── LEGAL CLAUSE ANALYSIS ───────────────────────────")
        leg = apps["legal"]
        clauses = [
            "The Company may terminate the employee's contract without notice or reason at any time.",
            "All intellectual property created during employment shall vest in the Company, including work done outside working hours.",
            "Payment terms: 30 days from invoice date. Late payment interest at Bank of England base rate + 8%.",
        ]
        for clause in clauses:
            r = leg.analyse_clause(clause)
            icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢","ACCEPTABLE":"✅"}.get(r.risk_level,"⚪")
            print(f"  {icon} [{r.risk_level}] {clause[:70]}...")
            print(f"       conf={r.confidence:.0%} mem={r.memories_used} {r.elapsed_ms:.0f}ms")

        # ── DevOps ────────────────────────────────────────────────────
        print("\n  ── DEVOPS INCIDENT RESPONSE ────────────────────────")
        ops = apps["devops"]
        alerts = [
            "CPU 98% on prod-api-03, response time >10s, OOM errors in logs, 500 error rate 40%",
            "Disk /var/log 99% full on prod-db-01",
            "k8s pod prod-api crashlooping: OOMKilled, restartCount=15",
        ]
        for alert in alerts:
            r = ops.diagnose(alert, service="production-api", env="production")
            icon = {"P1":"🔴","P2":"🟠","P3":"🟡","P4":"🟢"}.get(r.severity,"⚪")
            print(f"  {icon} [{r.severity}] {alert[:70]}")
            print(f"       runbook={r.runbook} conf={r.confidence:.0%} {r.elapsed_ms:.0f}ms")

        # ── Sports ────────────────────────────────────────────────────
        print("\n  ── SPORTS ANALYTICS ────────────────────────────────")
        spo = apps["sports"]
        # Load some data
        spo.load_season("Premier League 2024-25", [
            "Arsenal: 22W 8D 8L, 74pts, GF:72 GA:38",
            "Liverpool: 24W 6D 8L, 78pts, GF:81 GA:41",
            "Man City: 19W 9D 10L, 66pts, GF:63 GA:45",
        ])
        spo.load_player_stats("Erling Haaland", {"goals":24,"assists":8,"xG":22.1,"games":32})
        questions = [
            "Who is leading the Premier League this season?",
            "What are Haaland's stats this season?",
            "Analyse the Arsenal vs Liverpool match.",
        ]
        for q in questions:
            r = spo.query(q)
            print(f"  Q: {q}")
            print(f"     {r.answer[:100]}  [{r.elapsed_ms:.0f}ms]")

    print("\n" + "═"*62)
    print("  Demo complete. Use --llm ollama --model phi3.5 for real answers.")
    print("═"*62 + "\n")


def cmd_medical(args):
    """Interactive medical triage session."""
    from apps.medical.assistant import MedicalAssistant
    engine = _build_engine(args)
    app    = MedicalAssistant(engine)
    print(f"\n  SRHN Medical Triage  |  LLM: {engine.llm.name}")
    print("  Enter patient presentations. Commands: /quit /status /+ /-\n")
    last_qid = None
    while True:
        try:
            symptoms = input("  Patient > ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save(); engine.close(); break
        if not symptoms: continue
        if symptoms == "/quit": engine.save(); engine.close(); break
        if symptoms == "/status":
            s = engine.status()
            print(f"  Queries: {s['query_count']} | Memory: {s['memory']['total']}")
            continue
        if symptoms == "/+" and last_qid:
            app.confirm(last_qid, "Clinically appropriate triage")
            print("  ✓ Confirmed"); continue
        if symptoms == "/-" and last_qid:
            correction = input("  Correct category? ").strip()
            app.correct(last_qid, correction)
            print("  ✓ Correction recorded"); continue
        r = app.triage(symptoms)
        print(r.display())
        last_qid = r.query_id


def cmd_legal(args):
    """Interactive legal clause analysis session."""
    from apps.legal.analyser import LegalAnalyser
    engine = _build_engine(args)
    app    = LegalAnalyser(engine)
    print(f"\n  SRHN Legal Analyser  |  LLM: {engine.llm.name}")
    print("  Paste contract clauses. Commands: /quit /contract /status\n")
    last_qid = None
    while True:
        try:
            clause = input("  Clause > ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save(); engine.close(); break
        if not clause: continue
        if clause == "/quit": engine.save(); engine.close(); break
        if clause == "/status":
            s = engine.status()
            print(f"  Queries: {s['query_count']} | Memory: {s['memory']['total']}"); continue
        if clause == "/+" and last_qid:
            app.confirm(last_qid); print("  ✓ Confirmed"); continue
        if clause == "/-" and last_qid:
            r = input("  Correct risk level? ").strip()
            app.correct(last_qid, r); print("  ✓ Correction recorded"); continue
        r = app.analyse_clause(clause)
        print(r.display())
        last_qid = r.query_id


def cmd_devops(args):
    """Interactive DevOps incident response session."""
    from apps.devops.agent import RunbookAgent
    engine = _build_engine(args)
    app    = RunbookAgent(engine)
    print(f"\n  SRHN DevOps Agent  |  LLM: {engine.llm.name}")
    print("  Describe infrastructure alerts. Commands: /quit /resolved /worse\n")
    last_qid = None
    while True:
        try:
            alert = input("  Alert > ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save(); engine.close(); break
        if not alert: continue
        if alert == "/quit": engine.save(); engine.close(); break
        if alert == "/resolved" and last_qid:
            fix = input("  What fixed it? ").strip()
            app.resolved(last_qid, fix); print("  ✓ Resolution stored"); continue
        if alert == "/worse" and last_qid:
            what = input("  What made it worse? ").strip()
            app.made_worse(last_qid, what); print("  ✓ Failure recorded"); continue
        r = app.diagnose(alert, env=getattr(args,"env","production"))
        print(r.display())
        last_qid = r.query_id


def cmd_sports(args):
    """Interactive sports analytics session."""
    from apps.sports.analyst import SportsAnalyst
    engine = _build_engine(args)
    app    = SportsAnalyst(engine)
    print(f"\n  SRHN Sports Analyst  |  LLM: {engine.llm.name}")
    print("  Ask any sports question. /load to load season data. /quit to exit.\n")
    last_qid = None
    while True:
        try:
            q = input("  Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save(); engine.close(); break
        if not q: continue
        if q == "/quit": engine.save(); engine.close(); break
        if q == "/load":
            print("  Paste facts (one per line, blank line to finish):")
            lines = []
            while True:
                l = input("  > ").strip()
                if not l: break
                lines.append(l)
            season = input("  Season name: ").strip()
            app.load_season(season, lines)
            print(f"  ✓ {len(lines)} facts loaded")
            continue
        if q == "/+" and last_qid:
            app.confirm(last_qid); print("  ✓ Confirmed"); continue
        if q == "/-" and last_qid:
            correct = input("  Correct answer: ").strip()
            app.correct(last_qid, correct); print("  ✓ Correction stored"); continue
        r = app.query(q)
        print(r.display())
        last_qid = r.query_id


def cmd_load(args):
    """Load a safetensors model into the SRHN store."""
    engine = _build_engine(args)
    from core.safetensors_loader import SafetensorsLoader
    from core.ingestor import ingest_any
    loader = SafetensorsLoader(engine._cfg, engine.embed)
    result = ingest_any(args.path, engine, loader,
                        model_name=getattr(args,"name",""))
    print(result.summary())
    engine.save(); engine.close()


def cmd_server(args):
    """Start HTTP API server (delegates to api/server.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "server", str(Path(__file__).parent / "api" / "server.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    root = argparse.ArgumentParser(
        prog="srhn-assembly",
        description="SRHN Assembly — Real-world AI for medical, legal, devops, sports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  python main.py demo                          # see all domains in action
  python main.py medical --llm ollama          # clinical triage (real LLM)
  python main.py legal   --llm openai --api-key sk-...
  python main.py devops  --llm ollama --model llama3.2
  python main.py server  --llm ollama --apps medical,legal,devops
  python main.py load /models/llama-3.2-1b/ --name "llama-3.2-1b"
        """)
    sub = root.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--store",   default=os.getenv("SRHN_STORE","./srhn_store"))
        p.add_argument("--llm",     default=os.getenv("SRHN_LLM","mock"),
                       choices=["mock","ollama","openai","anthropic"])
        p.add_argument("--model",   default=os.getenv("SRHN_MODEL",""))
        p.add_argument("--api-key", default=os.getenv("LLM_API_KEY",""), dest="api_key")
        p.add_argument("--edge",    action="store_true")
        p.add_argument("--micro",   action="store_true")
        p.add_argument("--srhn-key",default=os.getenv("SRHN_API_KEY",""), dest="srhn_key")

    p = sub.add_parser("demo",    help="Run all domain demos")
    _common(p); p.set_defaults(func=cmd_demo)

    p = sub.add_parser("medical", help="Medical triage assistant")
    _common(p); p.set_defaults(func=cmd_medical)

    p = sub.add_parser("legal",   help="Legal clause analyser")
    _common(p); p.set_defaults(func=cmd_legal)

    p = sub.add_parser("devops",  help="DevOps incident response agent")
    _common(p)
    p.add_argument("--env", default="production")
    p.set_defaults(func=cmd_devops)

    p = sub.add_parser("sports",  help="Sports analytics assistant")
    _common(p); p.set_defaults(func=cmd_sports)

    p = sub.add_parser("load",    help="Load safetensors model")
    _common(p)
    p.add_argument("path"); p.add_argument("--name", default="")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("server",  help="HTTP API server")
    _common(p)
    p.add_argument("--port",  type=int, default=int(os.getenv("PORT","8765")))
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--apps",  default="medical,legal,devops,sports")
    p.add_argument("--safetensors", default=None)
    p.add_argument("--gen-key", action="store_true", dest="gen_key")
    p.set_defaults(func=cmd_server)

    args = root.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [srhn] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
