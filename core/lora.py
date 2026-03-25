"""
SRHN v5 — LoRA
FIX-5: Periodic lr reset every N steps prevents B matrix saturation.
       Robbins-Monro 1/sqrt(step) decay is mathematically correct for
       convergence but makes effective_lr → 0 after ~200 steps.
       Fix: reset step_count to 0 every lora_lr_reset_every steps,
       giving periodic warm restarts (cosine-annealing style, simpler).
FIX-9 (inherited): eviction cleans both _adapters AND _domain_sigs.
FIX-22 (inherited): clip=3.0.
"""
from __future__ import annotations
import json, math, os, time, threading
from typing import Optional
import numpy as np

from core.config import Config


class LoRAAdapter:
    """Single domain LoRA adapter with warm-restart lr scheduling."""

    def __init__(self, name: str, dim: int, cfg: Optional[Config] = None):
        self.name        = name
        self.dim         = dim
        self._cfg        = cfg or Config()
        rank             = self._cfg.lora_rank
        self.rank        = rank
        self.scale       = self._cfg.lora_alpha / rank
        self.created     = time.time()
        self.updated     = time.time()
        self.use_count   = 0
        self.reward_sum  = 0.0
        self.step_count  = 0
        self._lock       = threading.Lock()

        rng    = np.random.RandomState(hash(name) % (2**31))
        self.A = (rng.randn(rank, dim) / math.sqrt(dim)).astype(np.float32)
        self.B = np.zeros((dim, rank), dtype=np.float32)

    def apply(self, emb: np.ndarray) -> np.ndarray:
        if emb is None or len(emb) == 0: return emb
        emb = np.asarray(emb, dtype=np.float32)
        if not np.isfinite(emb).all():
            emb = np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            delta = self.B @ (self.A @ emb)
            out   = np.clip(emb + self.scale * delta, -5.0, 5.0)
            return out.astype(np.float32)
        except Exception:
            return emb

    def update(self, emb: np.ndarray, reward: float,
               lr: Optional[float] = None):
        if emb is None or not np.isfinite(emb).all(): return
        reward = float(np.clip(reward, -1.0, 1.0))
        lr     = lr if lr is not None else self._cfg.lora_lr
        with self._lock:
            self.step_count += 1
            self.reward_sum += reward

            # FIX-5: warm restart — reset step_count so lr doesn't decay to zero
            reset_every = getattr(self._cfg, "lora_lr_reset_every", 200)
            effective_step = self.step_count % reset_every or 1
            effective_lr   = lr / math.sqrt(effective_step)

            if abs(reward) < 1e-6: return

            emb = np.asarray(emb, dtype=np.float32)
            try:
                h  = self.A @ emb
                dB = reward * np.outer(emb, h)   # (dim, rank) ✓
                dA = reward * np.outer(h, emb)   # (rank, dim) ✓
                self.B += effective_lr * dB
                self.A += effective_lr * dA
                clip   = self._cfg.lora_clip
                self.B = np.clip(self.B, -clip, clip)
                self.A = np.clip(self.A, -clip, clip)
                self.updated = time.time()
            except Exception:
                pass

    @property
    def avg_reward(self) -> float:
        return self.reward_sum / max(self.step_count, 1)

    def size_kb(self) -> float:
        return (self.A.nbytes + self.B.nbytes) / 1024

    def to_dict(self) -> dict:
        with self._lock:
            return {"name": self.name, "dim": self.dim, "rank": self.rank,
                    "scale": self.scale, "created": self.created,
                    "updated": self.updated, "use_count": self.use_count,
                    "reward_sum": self.reward_sum, "step_count": self.step_count,
                    "A": self.A.tolist(), "B": self.B.tolist()}

    @classmethod
    def from_dict(cls, d: dict, cfg: Optional[Config] = None) -> "LoRAAdapter":
        cfg = cfg or Config()
        orig_rank, cfg.lora_rank = cfg.lora_rank, d.get("rank", cfg.lora_rank)
        obj = cls(d["name"], d["dim"], cfg)
        cfg.lora_rank  = orig_rank
        obj.scale      = d["scale"];      obj.created    = d["created"]
        obj.updated    = d["updated"];    obj.use_count  = d["use_count"]
        obj.reward_sum = d["reward_sum"]; obj.step_count = d["step_count"]
        obj.A = np.array(d["A"], dtype=np.float32)
        obj.B = np.array(d["B"], dtype=np.float32)
        return obj


class LoRABank:
    """Collection of domain adapters. Thread-safe."""

    # FIX-4: Added ml, sports domains
    DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "code":     ["python","function","class","def","bug","import","sql",
                     "error","api","algorithm","code","script","variable","loop"],
        "ml":       ["model","train","neural","network","deep","learning",
                     "predict","classification","regression","dataset","accuracy",
                     "transformer","attention","embedding","gradient","epoch"],
        "math":     ["calculate","equation","formula","integral","derivative",
                     "matrix","solve","proof","theorem","vector"],
        "medical":  ["patient","symptom","diagnosis","treatment","drug","dose",
                     "clinical","disease","health","medication","surgery",
                     "fever","pain","prescription","hospital","therapy"],
        "legal":    ["contract","law","regulation","clause","liability",
                     "compliance","statute","court","agreement","rights",
                     "litigation","breach","attorney","judgment"],
        "finance":  ["revenue","profit","investment","portfolio","stock",
                     "balance","budget","tax","forecast","capital","loss"],
        "sports":   ["player","match","game","team","score","goal","win",
                     "tournament","league","championship","coach","athlete",
                     "cricket","football","basketball","tennis","season"],
        "science":  ["hypothesis","experiment","result","data","analysis",
                     "research","study","theory","evidence","observation"],
        "creative": ["write","story","poem","character","plot","creative",
                     "narrative","fiction","dialogue","scene"],
        "devops":   ["docker","kubernetes","deploy","server","cloud","nginx",
                     "pipeline","cicd","infrastructure","monitor","log"],
    }

    def __init__(self, cfg: Optional[Config] = None):
        self._cfg       = cfg or Config()
        self._adapters: dict[str, LoRAAdapter] = {}
        self._sigs:     dict[str, np.ndarray]  = {}
        self._lock      = threading.RLock()

    def get_or_create(self, domain: str,
                      seed_emb: Optional[np.ndarray] = None) -> LoRAAdapter:
        with self._lock:
            if domain not in self._adapters:
                if len(self._adapters) >= self._cfg.lora_max_adapters:
                    self._evict()
                self._adapters[domain] = LoRAAdapter(
                    domain, self._cfg.embed_dim, self._cfg)
                if seed_emb is not None:
                    self._sigs[domain] = seed_emb.copy()
        return self._adapters[domain]

    def select(self, q_emb: np.ndarray, top_k: int = 3,
               threshold: float = 0.35) -> list[LoRAAdapter]:
        with self._lock:
            if not self._sigs: return []
            domains = [d for d in self._sigs if d in self._adapters]
            if not domains: return []
            sigs  = np.stack([self._sigs[d] for d in domains])
            q_n   = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            s_n   = sigs  / (np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-9)
            scores = np.nan_to_num(s_n @ q_n, nan=0.0)
            top_i  = np.argsort(-scores)[:top_k]
            return [self._adapters[domains[i]] for i in top_i
                    if float(scores[i]) > threshold]

    def apply(self, emb: np.ndarray, adapters: list[LoRAAdapter]) -> np.ndarray:
        result = emb.copy()
        for a in adapters:
            result = a.apply(result)
            with self._lock:
                a.use_count += 1
        return result

    def update(self, domain: str, emb: np.ndarray, reward: float):
        adapter = self.get_or_create(domain, emb)
        adapter.update(emb, reward)
        with self._lock:
            if domain in self._sigs:
                sig = 0.9 * self._sigs[domain] + 0.1 * emb
                n   = np.linalg.norm(sig)
                self._sigs[domain] = (sig / n) if n > 1e-9 else emb
            else:
                self._sigs[domain] = emb.copy()

    def infer_domain(self, text: str) -> str:
        """FIX-4: tie-break by score, not first match. Minimum threshold=1."""
        text_lower = text.lower()
        scores = {d: sum(1 for kw in kws if kw in text_lower)
                  for d, kws in self.DOMAIN_KEYWORDS.items()}
        best, val = max(scores.items(), key=lambda x: x[1])
        return best if val >= 1 else "general"

    def _evict(self):
        """Evict worst adapter AND its domain sig."""
        if not self._adapters: return
        worst = min(self._adapters, key=lambda d: self._adapters[d].avg_reward)
        del self._adapters[worst]
        self._sigs.pop(worst, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_adapters": len(self._adapters),
                "total_size_kb":  sum(a.size_kb() for a in self._adapters.values()),
                "domains":        list(self._adapters.keys()),
                "avg_reward":     float(np.mean([a.avg_reward
                                                  for a in self._adapters.values()])
                                        if self._adapters else 0.0),
                # FIX-5: show B_max so saturation is visible in status
                "B_max":          float(max((np.max(np.abs(a.B))
                                             for a in self._adapters.values()),
                                            default=0.0)),
            }

    def save(self, path: str):
        with self._lock:
            data = {"adapters": {k: v.to_dict() for k, v in self._adapters.items()},
                    "sigs":     {k: v.tolist()   for k, v in self._sigs.items()},
                    "embed_dim": self._cfg.embed_dim}
        tmp = path + ".tmp"
        with open(tmp, "w") as f: json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path): return
        try:
            with open(path) as f: data = json.load(f)
            with self._lock:
                self._adapters = {k: LoRAAdapter.from_dict(v, self._cfg)
                                  for k, v in data.get("adapters", {}).items()}
                self._sigs     = {k: np.array(v, dtype=np.float32)
                                  for k, v in data.get("sigs", {}).items()}
        except Exception as e:
            log.warning(f"LoRA load failed: {e} — starting fresh")


import logging
log = logging.getLogger("srhn")
