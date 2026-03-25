"""
SRHN v5 — Config
Production-ready. All values tunable via env vars or Config subclass.
"""
from __future__ import annotations
import os, secrets
from dataclasses import dataclass, field
from typing import Optional


def _e(key, default):
    v = os.environ.get(key)
    if v is None: return default
    if isinstance(default, bool): return v.lower() in ("1","true","yes")
    if isinstance(default, int):  return int(v)
    if isinstance(default, float): return float(v)
    return v


@dataclass
class Config:
    # ── Embedding ──────────────────────────────────────────────────────────────
    embed_dim:           int   = 384
    embed_max_chars:     int   = 8192
    embed_idf_max_docs:  int   = 100_000

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory_db:           str   = "srhn_memory.db"
    memory_max_entries:  int   = 500_000
    memory_cache_size:   int   = 50_000
    memory_wal_mode:     bool  = True

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_rank:           int   = 8
    lora_alpha:          float = 16.0
    lora_lr:             float = 0.01
    lora_clip:           float = 3.0
    lora_lr_reset_every: int   = 200   # FIX-5: periodic lr reset prevents B saturation
    lora_max_adapters:   int   = 500
    lora_bank_file:      str   = "lora_bank.json"

    # ── Engine ─────────────────────────────────────────────────────────────────
    store_dir:           str   = field(default_factory=lambda: _e("SRHN_STORE","srhn_store"))
    session_turns:       int   = 64
    episode_cache_size:  int   = 1000
    autosave_interval:   int   = 300

    # ── FIX-1: API authentication ──────────────────────────────────────────────
    api_key:             Optional[str] = field(default_factory=lambda: _e("SRHN_API_KEY", None))
    api_key_header:      str   = "X-SRHN-Key"

    # ── FIX-3: Request concurrency cap (prevents OOM under load) ──────────────
    max_concurrent_llm:  int   = field(default_factory=lambda: _e("SRHN_MAX_CONCURRENT", 8))

    # ── Loader ─────────────────────────────────────────────────────────────────
    loader_registry:         str  = "model_registry.json"
    loader_max_tensor_mb:    int  = 500
    loader_embed_timeout_ms: int  = 3000

    # ── Learning ───────────────────────────────────────────────────────────────
    learn_on_every_query:        bool  = True
    learn_feedback_weight:       float = 1.0
    learn_min_confidence_replay: float = 0.55

    # ── Edge preset ────────────────────────────────────────────────────────────
    @classmethod
    def edge(cls) -> "Config":
        """Raspberry Pi 4 / 2GB RAM / ARM Cortex-A72 target."""
        return cls(
            embed_dim=192,
            embed_max_chars=2048,
            memory_max_entries=50_000,
            memory_cache_size=5_000,
            lora_rank=4,
            lora_max_adapters=50,
            session_turns=16,
            episode_cache_size=200,
            loader_max_tensor_mb=50,
            max_concurrent_llm=2,
            autosave_interval=600,
        )

    @classmethod
    def micro(cls) -> "Config":
        """Microcontroller-class (ESP32-S3, 512KB+ PSRAM)."""
        return cls(
            embed_dim=64,
            embed_max_chars=512,
            memory_max_entries=5_000,
            memory_cache_size=500,
            lora_rank=2,
            lora_max_adapters=10,
            session_turns=4,
            episode_cache_size=50,
            loader_max_tensor_mb=10,
            max_concurrent_llm=1,
            autosave_interval=3600,
        )

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        mapping = {
            "SRHN_EMBED_DIM":       ("embed_dim",            int),
            "SRHN_STORE_DIR":       ("store_dir",            str),
            "SRHN_MEMORY_DB":       ("memory_db",            str),
            "SRHN_LORA_RANK":       ("lora_rank",            int),
            "SRHN_MAX_ENTRIES":     ("memory_max_entries",   int),
            "SRHN_AUTOSAVE":        ("autosave_interval",    int),
            "SRHN_MAX_CONCURRENT":  ("max_concurrent_llm",   int),
            "SRHN_API_KEY":         ("api_key",              str),
        }
        for env_key, (attr, typ) in mapping.items():
            val = os.getenv(env_key)
            if val is not None:
                setattr(c, attr, typ(val))
        return c

    def generate_api_key(self) -> str:
        """Generate a random API key and set it."""
        self.api_key = secrets.token_urlsafe(32)
        return self.api_key
