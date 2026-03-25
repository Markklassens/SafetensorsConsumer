"""
SRHN v5 — Memory
FIX-2: Prompt injection sanitisation on all stored content.
       Content stored via learn_fact / store_episodic is stripped of
       instruction-override patterns before being injected into prompts.
FIX-CACHE: Inherited from v4 (2s throttle, delta-count rebuild).
FIX-STRUCT: struct at top (inherited from v4).
"""
from __future__ import annotations
import hashlib, json, os, re, struct, sqlite3, threading, time, logging
from typing import Optional
import numpy as np

from core.config import Config

log = logging.getLogger("srhn")

# ── FIX-2: Prompt injection guard ─────────────────────────────────────────────
# Patterns that attempt to override system instructions.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|context)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"(disregard|forget|override)\s+(your\s+)?(instructions?|guidelines?|rules?|system)",
    r"jailbreak",
    r"DAN\s+mode",
    r"do\s+anything\s+now",
    r"(system|assistant)\s*:\s*you",
    r"<\s*/?\s*(system|instruction|context)\s*>",
    r"\[INST\]|\[\/INST\]",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitise(text: str, max_len: int = 2000) -> str:
    """Strip prompt injection patterns. Truncate to max_len."""
    if not text: return ""
    text = str(text)[:max_len * 2]
    text = _INJECTION_RE.sub("[filtered]", text)
    return text[:max_len]


def is_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(str(text or "")))


class AgentMemory:
    """Thread-safe persistent agent memory. SQLite WAL + numpy cosine search."""

    def __init__(self, db_path: str = "srhn_memory.db",
                 cfg: Optional[Config] = None):
        self._cfg       = cfg or Config()
        self.dim        = self._cfg.embed_dim
        self._path      = str(db_path)
        self._lock      = threading.RLock()
        self._read_only = False

        try:
            self._conn = sqlite3.connect(
                self._path, check_same_thread=False, timeout=30)
            self._conn.row_factory = sqlite3.Row
            if self._cfg.memory_wal_mode:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=10000")
            self._init_schema()
        except sqlite3.OperationalError as e:
            log.error(f"Memory DB open failed ({e}) — using in-memory fallback")
            self._conn      = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
            self._read_only = True

        self._cache_embs:  Optional[np.ndarray] = None
        self._cache_ids:   list[str]             = []
        self._cache_key:   str                   = ""
        self._cache_count: int                   = 0
        self._cache_built: float                 = 0.0
        self._dirty:       bool                  = True
        self._REBUILD_INTERVAL_S  = 2.0
        self._REBUILD_DELTA_COUNT = 50

    def _init_schema(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            entry_id     TEXT PRIMARY KEY,
            kind         TEXT NOT NULL DEFAULT 'episodic',
            content      TEXT NOT NULL,
            metadata     TEXT NOT NULL DEFAULT '{}',
            embedding    BLOB NOT NULL,
            created_at   REAL NOT NULL,
            accessed_at  REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            importance   REAL NOT NULL DEFAULT 0.5,
            domain       TEXT NOT NULL DEFAULT 'general'
        );
        CREATE INDEX IF NOT EXISTS idx_kind    ON memories(kind);
        CREATE INDEX IF NOT EXISTS idx_domain  ON memories(domain);
        CREATE INDEX IF NOT EXISTS idx_import  ON memories(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_access  ON memories(accessed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC);
        """)
        self._conn.commit()

    def _emb_to_blob(self, emb: np.ndarray) -> bytes:
        return emb.astype(np.float32).tobytes()

    def _blob_to_emb(self, blob: bytes) -> np.ndarray:
        try:
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            if len(arr) == self.dim: return arr
            out = np.zeros(self.dim, dtype=np.float32)
            n   = min(len(arr), self.dim)
            out[:n] = arr[:n]
            return out
        except Exception:
            return np.zeros(self.dim, dtype=np.float32)

    def store(self, content: str, embedding: np.ndarray,
              kind: str = "episodic", domain: str = "general",
              metadata: Optional[dict] = None,
              importance: float = 0.5) -> str:
        # FIX-2: sanitise all stored content
        content  = sanitise(content, 2000)
        ts_bytes = struct.pack(">d", time.time())
        entry_id = hashlib.sha256(
            content.encode() + ts_bytes + os.urandom(4)).hexdigest()[:16]
        now      = time.time()
        emb_blob = self._emb_to_blob(embedding)
        meta_str = json.dumps(metadata or {})
        imp      = float(np.clip(importance, 0.0, 1.0))

        with self._lock:
            try:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM memories").fetchone()[0]
                if total >= self._cfg.memory_max_entries:
                    self._conn.execute("""
                        DELETE FROM memories WHERE entry_id=(
                            SELECT entry_id FROM memories
                            ORDER BY importance ASC, created_at ASC LIMIT 1)""")
                self._conn.execute("""
                    INSERT INTO memories
                    (entry_id,kind,content,metadata,embedding,
                     created_at,accessed_at,access_count,importance,domain)
                    VALUES(?,?,?,?,?,?,?,0,?,?)""",
                    (entry_id, kind, content, meta_str, emb_blob,
                     now, now, imp, domain))
                self._conn.commit()
                self._dirty = True
            except sqlite3.Error as e:
                log.error(f"Memory store error: {e}")
        return entry_id

    def store_episodic(self, query: str, response: str, domain: str,
                        embedding: np.ndarray, confidence: float,
                        metadata: Optional[dict] = None) -> str:
        # FIX-2: sanitise query and response before storing
        query    = sanitise(query or "",    300)
        response = sanitise(response or "", 600)
        content  = f"Q: {query}\nA: {response}"
        meta     = {"query": query[:200], "response": response[:400],
                    "confidence": confidence, **(metadata or {})}
        return self.store(content, embedding, kind="episodic", domain=domain,
                          metadata=meta,
                          importance=float(np.clip(confidence, 0.0, 1.0)))

    def store_fact(self, fact: str, embedding: np.ndarray,
                   domain: str = "general", source: str = "",
                   confidence: float = 0.8) -> str:
        fact = sanitise(fact, 1000)
        return self.store(fact, embedding, kind="semantic", domain=domain,
                          metadata={"source": source[:200], "confidence": confidence},
                          importance=float(np.clip(confidence, 0.0, 1.0)))

    def store_failure(self, query: str, problem: str,
                      embedding: np.ndarray, domain: str = "general") -> str:
        content = f"FAILURE | Query: {sanitise(query,200)} | Problem: {sanitise(problem,300)}"
        return self.store(content, embedding, kind="failure", domain=domain,
                          metadata={"query": query[:200], "problem": problem[:300]},
                          importance=0.9)

    def store_preference(self, pref: str, embedding: np.ndarray,
                          category: str = "style") -> str:
        pref = sanitise(pref, 500)
        return self.store(pref, embedding, kind="preference", domain="user",
                          metadata={"category": category}, importance=0.85)

    def store_workflow(self, name: str, steps: list[str],
                       embedding: np.ndarray, domain: str,
                       success_rate: float = 1.0) -> str:
        safe_steps = [sanitise(s, 80) for s in steps[:20]]
        content    = f"WORKFLOW: {sanitise(name,100)} → " + " → ".join(safe_steps)
        return self.store(content, embedding, kind="procedural", domain=domain,
                          metadata={"name": name[:100], "steps": safe_steps,
                                    "success_rate": success_rate},
                          importance=float(np.clip(success_rate, 0.0, 1.0)))

    def search(self, query_emb: np.ndarray, top_k: int = 10,
               kind: Optional[str] = None, domain: Optional[str] = None,
               min_importance: float = 0.0) -> list[dict]:
        self._maybe_rebuild_cache(kind, domain, min_importance)
        if self._cache_embs is None or len(self._cache_ids) == 0:
            return []
        q = np.asarray(query_emb, dtype=np.float32)
        n = np.linalg.norm(q)
        if n < 1e-9: return []
        q /= n
        scores  = self._cache_embs @ q
        scores  = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        top_k   = min(top_k, len(scores))
        idx     = np.argpartition(-scores, min(top_k - 1, len(scores) - 1))[:top_k]
        idx     = idx[np.argsort(-scores[idx])]
        results = []
        now     = time.time()
        with self._lock:
            for i in idx:
                if i >= len(self._cache_ids): continue
                eid = self._cache_ids[i]
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE entry_id=?", (eid,)).fetchone()
                if not row: continue
                d = dict(row)
                d["similarity"] = round(float(scores[i]), 4)
                d["metadata"]   = json.loads(d.get("metadata", "{}"))
                d.pop("embedding", None)
                results.append(d)
                self._conn.execute(
                    "UPDATE memories SET accessed_at=?,access_count=access_count+1 "
                    "WHERE entry_id=?", (now, eid))
            self._conn.commit()
        return results

    def get_failures(self, q_emb: np.ndarray, top_k: int = 3) -> list[dict]:
        return self.search(q_emb, top_k=top_k, kind="failure")

    def get_preferences(self, top_n: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE kind='preference' "
                "ORDER BY importance DESC LIMIT ?", (top_n,)).fetchall()
        return [{**dict(r), "metadata": json.loads(r["metadata"])} for r in rows]

    def get_recent(self, n: int = 20, kind: Optional[str] = None) -> list[dict]:
        with self._lock:
            q      = ("SELECT entry_id,kind,domain,content,importance,accessed_at "
                      "FROM memories")
            params: list = []
            if kind: q += " WHERE kind=?"; params.append(kind)
            q += " ORDER BY accessed_at DESC LIMIT ?"; params.append(n)
            return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def update_importance(self, entry_id: str, reward: float):
        delta = float(np.clip(reward * 0.1, -0.2, 0.2))
        with self._lock:
            row = self._conn.execute(
                "SELECT importance FROM memories WHERE entry_id=?",
                (entry_id,)).fetchone()
            if row:
                new_imp = float(np.clip(row["importance"] + delta, 0.0, 1.0))
                self._conn.execute(
                    "UPDATE memories SET importance=? WHERE entry_id=?",
                    (new_imp, entry_id))
                self._conn.commit()

    def invalidate_cache(self):
        self._dirty = True

    def _maybe_rebuild_cache(self, kind, domain, min_importance):
        cache_key = f"{kind}|{domain}|{min_importance}"
        now       = time.time()
        if not self._dirty and cache_key == self._cache_key:
            with self._lock:
                cur_count = self._conn.execute(
                    "SELECT COUNT(*) FROM memories").fetchone()[0]
            if abs(cur_count - self._cache_count) < self._REBUILD_DELTA_COUNT:
                return
        if (not self._dirty and
                now - self._cache_built < self._REBUILD_INTERVAL_S):
            return
        self._rebuild_cache(kind, domain, min_importance, cache_key)

    def _rebuild_cache(self, kind, domain, min_importance, cache_key):
        qry    = "SELECT entry_id,embedding FROM memories WHERE importance >= ?"
        params: list = [float(min_importance)]
        if kind:   qry += " AND kind=?";   params.append(kind)
        if domain: qry += " AND domain=?"; params.append(domain)
        qry += f" ORDER BY importance DESC LIMIT {self._cfg.memory_cache_size}"
        with self._lock:
            rows      = self._conn.execute(qry, params).fetchall()
            cur_count = self._conn.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0]
        ids, embs = [], []
        for r in rows:
            try:
                e = self._blob_to_emb(r["embedding"])
                n = np.linalg.norm(e)
                embs.append(e / n if n > 1e-9 else e)
                ids.append(r["entry_id"])
            except Exception:
                pass
        self._cache_embs  = np.stack(embs).astype(np.float32) if embs else None
        self._cache_ids   = ids
        self._cache_key   = cache_key
        self._cache_count = cur_count
        self._cache_built = time.time()
        self._dirty       = False

    def stats(self) -> dict:
        with self._lock:
            total   = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_kind = {r[0]: r[1] for r in self._conn.execute(
                "SELECT kind,COUNT(*) FROM memories GROUP BY kind").fetchall()}
            avg_imp = (self._conn.execute(
                "SELECT AVG(importance) FROM memories").fetchone()[0] or 0.0)
        return {"total": total, "by_kind": by_kind,
                "avg_importance": round(float(avg_imp), 3),
                "read_only": self._read_only}

    def close(self):
        with self._lock:
            self._conn.close()
