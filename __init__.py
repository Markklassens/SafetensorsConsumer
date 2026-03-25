"""
SRHN Assembly — Main Entry Point
==================================
One import to get any domain application running.

Usage:
  from srhn_assembly import create_app, create_engine

  # Medical triage
  engine = create_engine(store="./medical_store", llm="ollama", model="phi3.5")
  app    = create_app("medical", engine)
  result = app.triage("65yo chest pain, diaphoresis, jaw radiation")
  print(result.display())

  # Legal analysis
  engine = create_engine(store="./legal_store", llm="openai", api_key="sk-...")
  app    = create_app("legal", engine)
  result = app.analyse_clause("Company may terminate without notice or reason.")
  print(result.display())

  # DevOps incident
  engine = create_engine(store="./ops_store", llm="ollama", model="llama3.2")
  app    = create_app("devops", engine)
  result = app.diagnose("CPU 99%, OOM errors, pod crashlooping — prod-api-03")
  print(result.display())

  # Run HTTP server with all apps
  python main.py server --apps medical,legal,devops --llm ollama --model phi3.5
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from core.config import Config
from core.engine import (SRHNEngine, MockLLM, OllamaLLM,
                          OpenAILLM, AnthropicLLM)


def create_engine(store: str = "./srhn_store",
                  llm: str = "mock",
                  model: str = "",
                  api_key: str = "",
                  base_url: str = None,
                  edge: bool = False,
                  micro: bool = False,
                  srhn_key: str = "") -> SRHNEngine:
    """
    Create and return a ready-to-use SRHNEngine.

    llm:     "mock" | "ollama" | "openai" | "anthropic"
    model:   LLM model name (e.g. "phi3.5", "gpt-4o-mini")
    api_key: API key for openai/anthropic
    edge:    Use edge config (192-dim, 50k memory, lower RAM)
    micro:   Use micro config (64-dim, 5k memory, minimal RAM)
    """
    if micro:
        cfg = Config.micro()
    elif edge:
        cfg = Config.edge()
    else:
        cfg = Config.from_env()

    cfg.store_dir = store
    if srhn_key:
        cfg.api_key = srhn_key

    os.makedirs(store, exist_ok=True)

    if llm == "ollama":
        llm_obj = OllamaLLM(model=model or "phi3.5")
    elif llm == "openai":
        llm_obj = OpenAILLM(api_key=api_key or os.getenv("OPENAI_API_KEY",""),
                             model=model or "gpt-4o-mini",
                             base_url=base_url)
    elif llm == "anthropic":
        llm_obj = AnthropicLLM(api_key=api_key or os.getenv("ANTHROPIC_API_KEY",""),
                                model=model or "claude-3-5-haiku-20241022")
    else:
        llm_obj = MockLLM()

    engine = SRHNEngine(cfg, llm_obj)
    return engine


def create_app(domain: str, engine: SRHNEngine):
    """
    Create a domain-specific application backed by the given engine.

    domain: "medical" | "legal" | "devops" | "sports"

    Returns:
      medical → MedicalAssistant
      legal   → LegalAnalyser
      devops  → RunbookAgent
      sports  → SportsAnalyst
    """
    d = domain.lower()
    if d == "medical":
        from apps.medical.assistant import MedicalAssistant
        return MedicalAssistant(engine)
    elif d == "legal":
        from apps.legal.analyser import LegalAnalyser
        return LegalAnalyser(engine)
    elif d == "devops":
        from apps.devops.agent import RunbookAgent
        return RunbookAgent(engine)
    elif d == "sports":
        from apps.sports.analyst import SportsAnalyst
        return SportsAnalyst(engine)
    else:
        raise ValueError(f"Unknown domain: {d}. Choose: medical, legal, devops, sports")


def create_multi_app(domains: list[str], store: str = "./srhn_store",
                     llm: str = "mock", model: str = "",
                     api_key: str = "", edge: bool = False) -> dict:
    """
    Create multiple domain apps sharing one engine and one memory store.
    Returns dict of {domain_name: app_instance}.

    This is the recommended setup — one store = shared memory across domains.
    Medical facts about a patient can inform legal decisions, etc.
    """
    engine = create_engine(store=store, llm=llm, model=model,
                           api_key=api_key, edge=edge)
    return {d: create_app(d, engine) for d in domains}
