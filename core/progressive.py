"""
SRHN v5 — Progressive Ingest Session
======================================
Answers the question: "I have 14 shards — can I load them one at a time,
test after each one, and resume if I get interrupted at shard 7?"

Yes. This module adds:

  ProgressiveSession   — load shards one-by-one, checkpoint after each
  ShardBenchmark       — test quality after each shard (before vs after)
  resume_session()     — pick up from the last completed shard
  ProgressiveCLI       — interactive REPL for shard-by-shard exploration

Design:
  - Each shard load is atomic: either it completes and is checkpointed, or
    the session stays at the previous shard.
  - Checkpoint file (JSON) records which shards are done, their stats,
    and the model_id so resume() works across restarts.
  - Benchmark runs a fixed probe query set before and after each shard,
    scoring confidence + memory hits as a quality proxy.
"""
from __future__ import annotations
import json, os, struct, time, logging, hashlib, shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable
import numpy as np

from core.config import Config
from core.engine import SRHNEngine
from core.safetensors_loader import SafetensorsLoader
from core.ingestor import (ingest, IngestResult, ProgressBar,
                            _parse_header, _file_size_mb, _estimate_peak_ram)

log = logging.getLogger("srhn.progressive")


# ─────────────────────────────────────────────────────────────────────────────
# Shard status
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShardStatus:
    index:        int           # 0-based
    filename:     str
    path:         str
    size_mb:      float
    tensors:      int   = 0
    params_M:     float = 0.0
    state:        str   = "pending"   # pending | loading | done | failed | skipped
    elapsed_s:    float = 0.0
    error:        str   = ""
    peak_ram_mb:  float = 0.0
    bench_before: Optional[dict] = None
    bench_after:  Optional[dict] = None

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def quality_delta(self) -> Optional[float]:
        """Change in benchmark score after this shard. None if not benchmarked."""
        if self.bench_before and self.bench_after:
            return round(self.bench_after["score"] - self.bench_before["score"], 4)
        return None


@dataclass
class SessionState:
    session_id:    str
    model_name:    str
    model_id:      str
    store_dir:     str
    shard_dir:     str
    total_shards:  int
    shards:        list[ShardStatus] = field(default_factory=list)
    started_at:    float = field(default_factory=time.time)
    updated_at:    float = field(default_factory=time.time)
    completed:     bool  = False

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.shards if s.done)

    @property
    def pending_count(self) -> int:
        return sum(1 for s in self.shards if s.state == "pending")

    @property
    def total_params_M(self) -> float:
        return sum(s.params_M for s in self.shards if s.done)

    @property
    def total_tensors(self) -> int:
        return sum(s.tensors for s in self.shards if s.done)

    def progress_bar(self, width: int = 30) -> str:
        done  = self.done_count
        total = self.total_shards
        filled = int(width * done / max(total, 1))
        bar   = "█" * filled + "░" * (width - filled)
        pct   = int(100 * done / max(total, 1))
        return f"[{bar}] {pct:3d}% ({done}/{total})"

    def summary_line(self) -> str:
        return (f"{self.model_name} | {self.progress_bar()} | "
                f"{self.total_params_M:.0f}M params | "
                f"{self.total_tensors} tensors")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: probe query set
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PROBES = [
    ("How many transformer layers does this model have?",     "model_structure"),
    ("What is the hidden dimension of this model?",           "model_structure"),
    ("Does this model use grouped query attention?",          "model_structure"),
    ("What vocabulary size does this model use?",             "model_structure"),
    ("Is this model quantized?",                              "model_structure"),
    ("What is the estimated context length?",                 "model_structure"),
]


def _run_benchmark(engine: SRHNEngine,
                   probes: list[tuple[str, str]] = None) -> dict:
    """
    Run probe queries and return a quality snapshot.
    Score = mean(confidence) weighted by memories_used.
    """
    probes = probes or _DEFAULT_PROBES
    results = []
    for query, expected_domain in probes:
        try:
            r = engine.query(query, top_k=6)
            results.append({
                "query":        query,
                "conf":         r.get("confidence", 0.0),
                "mem":          r.get("memories_used", 0),
                "domain":       r.get("domain", "general"),
                "correct_domain": r.get("domain") == expected_domain,
                "ms":           r.get("elapsed_ms", 0),
            })
        except Exception as e:
            results.append({"query": query, "conf": 0.0, "mem": 0,
                            "error": str(e)})

    confs    = [r["conf"] for r in results]
    mems     = [r["mem"]  for r in results]
    dom_ok   = [r.get("correct_domain", False) for r in results]
    score    = float(np.mean(confs)) * 0.6 + float(np.mean(mems)) * 0.04 * 0.4
    return {
        "score":          round(score, 4),
        "mean_conf":      round(float(np.mean(confs)), 4),
        "mean_mem":       round(float(np.mean(mems)), 2),
        "domain_acc":     round(float(np.mean(dom_ok)), 2),
        "mean_ms":        round(float(np.mean([r.get("ms",0) for r in results])), 1),
        "n_probes":       len(probes),
        "details":        results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resolve shard list from directory
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_shards(shard_dir: str) -> list[str]:
    """Return ordered list of shard paths. Reads index.json if present."""
    d = Path(shard_dir)
    idx_file = d / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx = json.load(f)
        ordered = list(dict.fromkeys(idx.get("weight_map", {}).values()))
        files   = [d / s for s in ordered if (d / s).exists()]
        missing = [s for s in ordered if not (d / s).exists()]
        if missing:
            log.warning(f"Missing shards: {missing}")
        return [str(f) for f in files]
    # fallback: glob sorted
    return sorted(str(f) for f in d.glob("*.safetensors"))


# ─────────────────────────────────────────────────────────────────────────────
# Progressive Session
# ─────────────────────────────────────────────────────────────────────────────

class ProgressiveSession:
    """
    Load a sharded model one shard at a time with:
      - Live progress tracking per shard
      - Checkpointing after each shard (resume if interrupted)
      - Optional benchmark before/after each shard
      - Query interface available at any point
      - Full status reporting

    Usage:
        session = ProgressiveSession.create(
            shard_dir="/models/llama-3-70b/",
            engine=engine,
            loader=loader,
            model_name="llama-3-70b"
        )
        session.run()           # load all remaining shards
        # or step-by-step:
        session.next()          # load next pending shard
        result = session.query("How many layers?")
        session.next()
    """

    CHECKPOINT_FILENAME = "progressive_session.json"

    def __init__(self, state: SessionState,
                 engine: SRHNEngine,
                 loader: SafetensorsLoader,
                 benchmark: bool = False,
                 probe_queries: Optional[list[tuple[str, str]]] = None,
                 on_shard_done: Optional[Callable[[ShardStatus], None]] = None):
        self.state        = state
        self.engine       = engine
        self.loader       = loader
        self.benchmark    = benchmark
        self.probe_queries = probe_queries
        self.on_shard_done = on_shard_done
        self._ckpt_path   = str(Path(state.store_dir) / self.CHECKPOINT_FILENAME)

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def create(cls,
               shard_dir: str,
               engine: SRHNEngine,
               loader: SafetensorsLoader,
               model_name: str = "",
               model_id:   str = "",
               benchmark:  bool = False,
               probe_queries: Optional[list[tuple[str, str]]] = None,
               on_shard_done: Optional[Callable] = None,
               ) -> "ProgressiveSession":
        """Create a new session for a shard directory."""
        shard_paths = _resolve_shards(shard_dir)
        if not shard_paths:
            raise ValueError(f"No .safetensors files in {shard_dir}")

        d      = Path(shard_dir)
        name   = model_name or d.name
        mid    = model_id   or hashlib.sha256(str(d).encode()).hexdigest()[:8]
        sid    = hashlib.sha256(f"{mid}{time.time()}".encode()).hexdigest()[:12]

        shards = []
        for i, p in enumerate(shard_paths):
            try:
                header, data_off = _parse_header(p)
                tensors = {k: v for k, v in header.items() if k != "__metadata__"}
                peak_ram = _estimate_peak_ram(tensors)
            except Exception:
                tensors  = {}
                peak_ram = 0.0
            shards.append(ShardStatus(
                index=i, filename=Path(p).name, path=p,
                size_mb=round(_file_size_mb(p), 1),
                peak_ram_mb=round(peak_ram, 1),
            ))

        state = SessionState(
            session_id=sid, model_name=name, model_id=mid,
            store_dir=engine._cfg.store_dir,
            shard_dir=str(d),
            total_shards=len(shards), shards=shards,
        )
        session = cls(state, engine, loader, benchmark, probe_queries, on_shard_done)
        session._save_checkpoint()
        return session

    @classmethod
    def resume(cls,
               engine: SRHNEngine,
               loader: SafetensorsLoader,
               benchmark: bool = False,
               probe_queries: Optional[list[tuple[str, str]]] = None,
               on_shard_done: Optional[Callable] = None,
               ) -> Optional["ProgressiveSession"]:
        """Resume an interrupted session from checkpoint in the store directory."""
        ckpt = Path(engine._cfg.store_dir) / cls.CHECKPOINT_FILENAME
        if not ckpt.exists():
            return None
        with open(ckpt) as f:
            raw = json.load(f)
        shards = [ShardStatus(**s) for s in raw.pop("shards")]
        state  = SessionState(**raw, shards=shards)
        s = cls(state, engine, loader, benchmark, probe_queries, on_shard_done)
        log.info(f"Resumed session {state.session_id}: "
                 f"{state.done_count}/{state.total_shards} shards done")
        return s

    # ── Core operations ──────────────────────────────────────────────────────

    def next(self) -> Optional[ShardStatus]:
        """
        Load the next pending shard. Returns the shard status, or None if all done.
        Checkpoints after each successful load.
        """
        pending = [s for s in self.state.shards if s.state == "pending"]
        if not pending:
            self.state.completed = True
            self._save_checkpoint()
            return None

        shard = pending[0]
        return self._load_shard(shard)

    def load_shard_by_index(self, index: int) -> ShardStatus:
        """Load a specific shard by its index (0-based)."""
        if index < 0 or index >= len(self.state.shards):
            raise IndexError(f"Shard index {index} out of range (0-{len(self.state.shards)-1})")
        shard = self.state.shards[index]
        if shard.done:
            log.info(f"Shard {index} already done")
            return shard
        return self._load_shard(shard)

    def _load_shard(self, shard: ShardStatus) -> ShardStatus:
        """Internal: load one shard with timing, benchmarking, and checkpointing."""
        shard.state = "loading"
        self._save_checkpoint()

        # Benchmark BEFORE this shard (if enabled and there's something to compare)
        if self.benchmark and self.state.done_count > 0:
            print(f"  Benchmarking before shard {shard.index+1}...")
            shard.bench_before = _run_benchmark(self.engine, self.probe_queries)

        t0 = time.time()
        try:
            print(f"  Loading shard {shard.index+1}/{self.state.total_shards}: "
                  f"{shard.filename}  ({shard.size_mb:.1f} MB)...")
            pb = ProgressBar(100, f"  Shard {shard.index+1}", width=32)
            processed = [0]

            def _cb(i, total, key):
                pct = int(100 * i / max(total, 1))
                processed[0] = pct
                pb.update(pct, suffix=key.split(".")[-1][:18])

            ms = self.loader.add_shard(
                shard.path,
                model_id=self.state.model_id,
                model_name=self.state.model_name,
                progress_cb=_cb)

            pb.done(f"{ms.tensors[list(ms.tensors.keys())[-1]].n_params//1000}k params last tensor")

            # Inject architecture facts after each shard
            af = self.loader.get_arch_facts(self.state.model_id)
            lp = self.loader.get_layer_profile(self.state.model_id)
            self.engine.inject_model_knowledge(af, lp)

            # Update shard status
            shard.tensors  = len(ms.tensors)
            shard.params_M = round(ms.total_params / 1e6, 1)
            shard.elapsed_s = round(time.time() - t0, 2)
            shard.state    = "done"

            # Benchmark AFTER
            if self.benchmark:
                print(f"  Benchmarking after shard {shard.index+1}...")
                shard.bench_after = _run_benchmark(self.engine, self.probe_queries)
                delta = shard.quality_delta
                if delta is not None:
                    arrow = "↑" if delta >= 0 else "↓"
                    print(f"  Quality delta: {arrow} {abs(delta):.4f}  "
                          f"(score: {shard.bench_before['score']:.3f} → "
                          f"{shard.bench_after['score']:.3f})")

        except Exception as e:
            shard.state   = "failed"
            shard.error   = str(e)
            shard.elapsed_s = round(time.time() - t0, 2)
            log.error(f"Shard {shard.index} failed: {e}")

        self.state.updated_at = time.time()
        self._save_checkpoint()

        if self.on_shard_done:
            try: self.on_shard_done(shard)
            except Exception: pass

        return shard

    def run(self, stop_after: Optional[int] = None,
            on_progress: Optional[Callable[[SessionState], None]] = None) -> SessionState:
        """
        Load all remaining shards sequentially.
        stop_after: stop after this many shards (None = all)
        """
        loaded = 0
        while True:
            if stop_after is not None and loaded >= stop_after:
                break
            shard = self.next()
            if shard is None:
                print(f"\n  All {self.state.total_shards} shards loaded.")
                break
            loaded += 1
            print(f"  {self.state.summary_line()}")
            if on_progress:
                try: on_progress(self.state)
                except Exception: pass
        return self.state

    def skip(self, index: int):
        """Mark a shard as skipped (e.g. corrupt file)."""
        shard = self.state.shards[index]
        shard.state = "skipped"
        self._save_checkpoint()
        print(f"  Shard {index} skipped.")

    # ── Query interface ──────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 6) -> dict:
        """Query the engine with whatever shards are loaded so far."""
        result = self.engine.query(text, top_k=top_k)
        return result

    def feedback(self, query_id: int, reward: float, note: str = ""):
        """Give feedback on a query result."""
        return self.engine.feedback(query_id, reward, note)

    def benchmark_now(self,
                      probes: Optional[list[tuple[str,str]]] = None) -> dict:
        """Run a benchmark snapshot right now."""
        return _run_benchmark(self.engine, probes or self.probe_queries)

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        s = self.state
        done   = [sh for sh in s.shards if sh.done]
        failed = [sh for sh in s.shards if sh.state == "failed"]

        result = {
            "session_id":    s.session_id,
            "model":         s.model_name,
            "model_id":      s.model_id,
            "progress":      f"{s.done_count}/{s.total_shards}",
            "completed":     s.completed,
            "total_params_M": round(s.total_params_M, 1),
            "total_tensors": s.total_tensors,
            "done_shards":   [{"index":sh.index,"file":sh.filename,
                               "tensors":sh.tensors,"params_M":sh.params_M,
                               "elapsed_s":sh.elapsed_s,
                               "quality_delta":sh.quality_delta}
                              for sh in done],
            "pending_shards": [{"index":sh.index,"file":sh.filename,
                                "size_mb":sh.size_mb,"peak_ram_mb":sh.peak_ram_mb}
                               for sh in s.shards if sh.state=="pending"],
            "failed_shards":  [{"index":sh.index,"file":sh.filename,
                                "error":sh.error} for sh in failed],
            "engine":         self.engine.status(),
        }
        return result

    def print_status(self):
        s = self.state
        print(f"\n  Session: {s.session_id}  Model: {s.model_name}")
        print(f"  Progress: {s.progress_bar(40)}")
        print(f"  Params loaded: {s.total_params_M:.0f}M  |  Tensors: {s.total_tensors}")
        print()
        print(f"  {'#':>3}  {'Filename':<40}  {'State':<8}  "
              f"{'Tensors':>7}  {'Params_M':>8}  {'Time_s':>6}  {'Δ Quality':>9}")
        print(f"  {'─'*3}  {'─'*40}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*9}")
        for sh in s.shards:
            delta = f"{sh.quality_delta:+.4f}" if sh.quality_delta is not None else "       "
            icon  = {"done":"✓","pending":"·","loading":"→",
                     "failed":"✗","skipped":"○"}.get(sh.state,"?")
            print(f"  {sh.index+1:>3}  {sh.filename:<40}  "
                  f"{icon} {sh.state:<6}  {sh.tensors:>7}  "
                  f"{sh.params_M:>8.1f}  {sh.elapsed_s:>6.1f}  {delta:>9}")
        print()
        mem = self.engine.memory.stats()
        lora = self.engine.lora.stats()
        print(f"  Memory: {mem['total']} entries  |  "
              f"LoRA: {lora['total_adapters']} adapters  |  "
              f"B_max: {lora.get('B_max',0):.4f}")

    def print_benchmark_report(self):
        """Print quality change across all benchmarked shards."""
        benchmarked = [sh for sh in self.state.shards
                       if sh.bench_after is not None]
        if not benchmarked:
            print("  No benchmark data yet. Use benchmark=True when creating session.")
            return
        print(f"\n  Benchmark report — {self.state.model_name}")
        print(f"  {'Shard':>5}  {'Before':>7}  {'After':>7}  {'Delta':>7}  "
              f"{'Mem':>5}  {'DomAcc':>7}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*7}")
        for sh in benchmarked:
            b4  = sh.bench_before
            b4s = b4["score"] if b4 else 0
            af  = sh.bench_after
            afs = af["score"]
            delta = afs - b4s
            print(f"  {sh.index+1:>5}  "
                  f"{b4s:>7.4f}  {afs:>7.4f}  "
                  f"{delta:>+7.4f}  {af['mean_mem']:>5.1f}  {af['domain_acc']:>7.2f}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_checkpoint(self):
        """Save session state to JSON checkpoint file."""
        s     = self.state
        data  = {
            "session_id":   s.session_id,
            "model_name":   s.model_name,
            "model_id":     s.model_id,
            "store_dir":    s.store_dir,
            "shard_dir":    s.shard_dir,
            "total_shards": s.total_shards,
            "started_at":   s.started_at,
            "updated_at":   s.updated_at,
            "completed":    s.completed,
            "shards":       [asdict(sh) for sh in s.shards],
        }
        tmp = self._ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._ckpt_path)

    def save_engine(self):
        """Persist engine state (LoRA, memory, query count)."""
        self.engine.save()
        log.info("Engine state saved")

    def export_report(self, path: str):
        """Export full session report as JSON."""
        report = {
            "session":    asdict(self.state),
            "engine":     self.engine.status(),
            "benchmark":  self.print_benchmark_report.__doc__,
            "shards":     [asdict(sh) for sh in self.state.shards],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report exported: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# resume_session() convenience function
# ─────────────────────────────────────────────────────────────────────────────

def resume_session(engine: SRHNEngine,
                   loader: SafetensorsLoader,
                   benchmark: bool = False) -> Optional[ProgressiveSession]:
    """
    Resume a previously interrupted session. Returns None if no checkpoint found.
    """
    s = ProgressiveSession.resume(engine, loader, benchmark=benchmark)
    if s is None:
        print("  No session checkpoint found in store directory.")
        print(f"  Expected: {engine._cfg.store_dir}/{ProgressiveSession.CHECKPOINT_FILENAME}")
        return None
    print(f"\n  Resumed session: {s.state.session_id}")
    print(f"  {s.state.summary_line()}")
    pending = [sh for sh in s.state.shards if sh.state == "pending"]
    print(f"  Pending shards: {len(pending)}")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def interactive_session(session: ProgressiveSession):
    """
    Interactive REPL for progressive shard loading.
    Load shards, query after each one, give feedback, check quality.

    Commands:
      next      — load next pending shard
      load N    — load shard N (0-based)
      run N     — load next N shards
      run all   — load all remaining
      query     — enter a query
      bench     — run benchmark now
      status    — show full status table
      feedback  — give +/-1 on last result
      skip N    — skip shard N
      save      — save engine state
      report F  — export JSON report to file F
      quit      — save and exit
    """
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  SRHN Progressive Session — Interactive              ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Model: {session.state.model_name:<45}║")
    print(f"  ║  Shards: {session.state.total_shards:<44}║")
    print(f"  ║  Store:  {session.state.store_dir:<44}║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"\n  Commands: next  load N  run [N|all]  query  bench")
    print(f"            status  feedback  skip N  save  report F  quit\n")

    last_qid = None

    while True:
        try:
            pending = session.state.pending_count
            done    = session.state.done_count
            total   = session.state.total_shards
            prompt  = f"  [{done}/{total} shards] > "
            cmd = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving and exiting...")
            session.save_engine()
            break

        if not cmd:
            continue

        parts = cmd.split()
        verb  = parts[0].lower()

        if verb == "quit":
            session.save_engine()
            break

        elif verb == "next":
            shard = session.next()
            if shard is None:
                print("  All shards loaded.")
            else:
                icon = "✓" if shard.done else "✗"
                print(f"  {icon} Shard {shard.index+1}: "
                      f"{shard.tensors} tensors, "
                      f"{shard.params_M:.0f}M params, "
                      f"{shard.elapsed_s:.1f}s")
                if shard.error:
                    print(f"  Error: {shard.error}")

        elif verb == "load" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1   # user gives 1-based
                shard = session.load_shard_by_index(idx)
                icon = "✓" if shard.done else "✗"
                print(f"  {icon} Shard {shard.index+1}: "
                      f"{shard.tensors} tensors, {shard.params_M:.0f}M params")
                if shard.error:
                    print(f"  Error: {shard.error}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")

        elif verb == "run":
            n_arg = parts[1] if len(parts) >= 2 else "1"
            n = None if n_arg.lower() == "all" else int(n_arg)
            state = session.run(stop_after=n)
            print(f"  {state.summary_line()}")

        elif verb == "query":
            user = input("  Query: ").strip()
            if not user:
                continue
            r = session.query(user)
            last_qid = r["query_id"]
            if r.get("error") == "invalid_query":
                print("  [BLOCKED] Query contains disallowed patterns.")
            else:
                print(f"\n  [{r['domain']} | conf={r['confidence']:.2f} | "
                      f"mem={r['memories_used']} | {r['elapsed_ms']}ms]\n")
                print(f"  {r['response']}\n")

        elif verb == "q":   # shorthand
            r = session.query(" ".join(parts[1:]) if len(parts)>1 else
                              input("  Query: ").strip())
            last_qid = r["query_id"]
            if not r.get("error"):
                print(f"\n  [{r['domain']} | conf={r['confidence']:.2f} | "
                      f"mem={r['memories_used']}]\n  {r['response']}\n")

        elif verb == "bench":
            print("  Running benchmark...")
            b = session.benchmark_now()
            print(f"  Score: {b['score']:.4f} | "
                  f"Conf: {b['mean_conf']:.3f} | "
                  f"Mem: {b['mean_mem']:.1f} | "
                  f"DomAcc: {b['domain_acc']:.2f}")

        elif verb == "status":
            session.print_status()

        elif verb == "feedback":
            if not last_qid:
                print("  No query to give feedback on yet.")
                continue
            fb = input("  +1 (good) or -1 (bad)? ").strip()
            try:
                reward = float(fb)
                note   = input("  Note (optional): ").strip()
                session.feedback(last_qid, reward, note)
                print(f"  Feedback recorded: {reward:+.1f}")
            except ValueError:
                print("  Enter +1 or -1")

        elif verb == "skip" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
                session.skip(idx)
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")

        elif verb == "save":
            session.save_engine()
            print("  Saved.")

        elif verb == "report" and len(parts) >= 2:
            session.export_report(parts[1])

        elif verb in ("help","?"):
            print("  Commands:")
            print("    next          Load next pending shard")
            print("    load N        Load shard N (1-based)")
            print("    run [N|all]   Load N or all remaining shards")
            print("    q <text>      Quick query")
            print("    query         Interactive query prompt")
            print("    bench         Run quality benchmark")
            print("    status        Full status table")
            print("    feedback      +/-1 on last result")
            print("    skip N        Skip shard N")
            print("    save          Save engine state")
            print("    report FILE   Export JSON report")
            print("    quit          Save and exit")
        else:
            print(f"  Unknown command: {cmd}  (type 'help' for list)")


# ─────────────────────────────────────────────────────────────────────────────
# Layer coverage map
# ─────────────────────────────────────────────────────────────────────────────

def layer_coverage_map(loader: SafetensorsLoader, model_id: str,
                       width: int = 60) -> str:
    """
    Returns a visual map of which transformer layers are loaded.

      Layers 0-31  (32 total)
      ████████████████░░░░░░░░░░░░░░░░
      Loaded: 0-15 (16)  |  Pending: 16-31 (16)

    Each character = one layer. █ = loaded, ░ = not yet loaded.
    """
    ms = loader.get_model(model_id)
    if not ms:
        return "  No model loaded"

    # Extract layer numbers from tensor keys
    loaded_layers: set[int] = set()
    for key in ms.tensors:
        parts = key.split(".")
        for p in parts:
            if p.isdigit():
                loaded_layers.add(int(p))

    if not loaded_layers:
        return "  No layer structure detected"

    total = ms.architecture.get("n_layers", max(loaded_layers) + 1)
    total = max(total, max(loaded_layers) + 1)

    # Build character map
    step  = max(1, total // width)
    chars = []
    for i in range(0, total, step):
        bucket = set(range(i, min(i + step, total)))
        if bucket & loaded_layers:
            chars.append("█")
        else:
            chars.append("░")

    pct = int(100 * len(loaded_layers) / total)
    loaded_range = (f"{min(loaded_layers)}-{max(loaded_layers)}"
                    if loaded_layers else "none")
    pending = [i for i in range(total) if i not in loaded_layers]
    pending_range = (f"{min(pending)}-{max(pending)}"
                     if pending else "all loaded")

    lines = [
        f"  Layers 0-{total-1}  ({total} total, {pct}% loaded)",
        f"  {''.join(chars)}",
        f"  Loaded:  {loaded_range} ({len(loaded_layers)} layers)",
        f"  Pending: {pending_range} ({len(pending)} layers)",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor inspector
# ─────────────────────────────────────────────────────────────────────────────

def tensor_inspector(loader: SafetensorsLoader, model_id: str,
                     filter_type: str = "") -> str:
    """
    Show breakdown of loaded tensors by type (attention/mlp/embed/norm/head).
    filter_type: '' = all, 'attn', 'mlp', 'embed', 'norm', 'head'
    """
    ms = loader.get_model(model_id)
    if not ms:
        return "  No model loaded"

    buckets: dict[str, list] = {
        "embed": [], "attn": [], "mlp": [], "norm": [], "head": [], "other": []
    }
    for key, ti in ms.tensors.items():
        k = key.lower()
        if "embed" in k:               buckets["embed"].append((key, ti))
        elif "self_attn" in k or "attention" in k: buckets["attn"].append((key, ti))
        elif "mlp" in k or "ffn" in k: buckets["mlp"].append((key, ti))
        elif "norm" in k or "ln" in k: buckets["norm"].append((key, ti))
        elif "lm_head" in k or "head" in k: buckets["head"].append((key, ti))
        else:                          buckets["other"].append((key, ti))

    if filter_type and filter_type in buckets:
        buckets = {filter_type: buckets[filter_type]}

    lines = [f"  Tensor breakdown — {ms.name}  ({len(ms.tensors)} total)\n"]
    for btype, items in buckets.items():
        if not items: continue
        total_p = sum(ti.n_params for _, ti in items)
        lines.append(f"  {btype.upper():<8} {len(items):>4} tensors  "
                     f"{total_p/1e6:>8.1f}M params")
        if len(items) <= 6:   # show details for small groups
            for key, ti in items:
                lines.append(f"           {key[-50:]:<50}  "
                             f"{str(ti.shape):<20}  {ti.dtype}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Compare: query before vs after a shard
# ─────────────────────────────────────────────────────────────────────────────

def compare_query(session: ProgressiveSession,
                  query: str,
                  shard_index: int) -> dict:
    """
    Load a shard, compare query result before and after.
    Returns: {before, after, delta_conf, delta_mem}
    """
    # Query before
    before = session.engine.query(query, top_k=6)

    # Load the shard
    shard = session.load_shard_by_index(shard_index)
    if not shard.done:
        return {"error": f"Shard {shard_index} failed: {shard.error}",
                "before": before, "after": None}

    # Query after
    after = session.engine.query(query, top_k=6)

    delta_conf = round(after["confidence"] - before["confidence"], 4)
    delta_mem  = round(after["memories_used"] - before["memories_used"], 1)

    return {
        "query":       query,
        "shard":       shard.index + 1,
        "shard_file":  shard.filename,
        "before": {
            "response":   before["response"],
            "confidence": before["confidence"],
            "memories":   before["memories_used"],
            "domain":     before["domain"],
        },
        "after": {
            "response":   after["response"],
            "confidence": after["confidence"],
            "memories":   after["memories_used"],
            "domain":     after["domain"],
        },
        "delta_conf":  delta_conf,
        "delta_mem":   delta_mem,
        "improved":    delta_conf > 0 or delta_mem > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom probe loader
# ─────────────────────────────────────────────────────────────────────────────

def load_probes_from_file(path: str) -> list[tuple[str, str]]:
    """
    Load custom benchmark probes from a JSON or plain-text file.

    JSON format:
      [{"query": "...", "domain": "medical"}, ...]
      or [["query text", "domain"], ...]
      or ["query1", "query2", ...]   (domain defaults to "general")

    Plain text (one query per line):
      What are the symptoms of appendicitis?
      How many layers does this model have?
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Probe file not found: {path}")

    content = p.read_text(encoding="utf-8").strip()

    if p.suffix.lower() == ".json":
        data = json.loads(content)
        probes = []
        for item in data:
            if isinstance(item, dict):
                probes.append((item.get("query",""), item.get("domain","general")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                probes.append((str(item[0]), str(item[1])))
            elif isinstance(item, str):
                probes.append((item, "general"))
        return probes
    else:
        # Plain text: one query per line, optional |domain suffix
        probes = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                q, d = line.rsplit("|", 1)
                probes.append((q.strip(), d.strip()))
            else:
                probes.append((line, "general"))
        return probes


# ─────────────────────────────────────────────────────────────────────────────
# URL shard downloader (edge devices often pull shards from remote storage)
# ─────────────────────────────────────────────────────────────────────────────

def download_shard(url: str, dest_dir: str,
                   filename: str = "",
                   on_progress: Optional[Callable[[int, int], None]] = None,
                   timeout: int = 300) -> str:
    """
    Download a .safetensors shard from a URL to dest_dir.
    Shows progress. Returns local file path.

    Usage:
        path = download_shard(
            "https://huggingface.co/.../model-00001-of-00014.safetensors",
            dest_dir="/models/llama-70b/")
        session = ProgressiveSession.create(dest_dir, engine, loader)
    """
    import urllib.request, urllib.error

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or Path(url.split("?")[0]).name
    dest  = dest_dir / fname

    if dest.exists():
        log.info(f"Already downloaded: {dest}")
        return str(dest)

    log.info(f"Downloading: {url} → {dest}")
    tmp = str(dest) + ".download"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SRHN/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks (edge-friendly)

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        try: on_progress(downloaded, total)
                        except Exception: pass
                    if total > 0:
                        pct = int(100 * downloaded / total)
                        mb  = downloaded / (1024 * 1024)
                        tot_mb = total / (1024 * 1024)
                        print(f"\r  Downloading {fname}: {mb:.0f}/{tot_mb:.0f} MB  ({pct}%)",
                              end="", flush=True)

        os.replace(tmp, str(dest))
        print(f"\r  Downloaded: {fname}  ({os.path.getsize(str(dest))//1024//1024} MB)    ")
        return str(dest)

    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed: {e}")


def download_shards_from_index(index_url: str, dest_dir: str,
                                shards: Optional[list[int]] = None,
                                timeout: int = 300) -> list[str]:
    """
    Download shards from a HuggingFace-style index.json URL.
    shards: list of 1-based shard numbers to download (None = all).

    Usage:
        paths = download_shards_from_index(
            "https://huggingface.co/.../model.safetensors.index.json",
            dest_dir="/models/llama-70b/",
            shards=[1, 2, 3])  # first 3 shards only
    """
    import urllib.request
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download and parse the index
    print(f"  Fetching index: {index_url}")
    with urllib.request.urlopen(index_url, timeout=30) as resp:
        index = json.loads(resp.read())

    weight_map = index.get("weight_map", {})
    all_shards = list(dict.fromkeys(weight_map.values()))

    if shards:
        selected = [all_shards[i-1] for i in shards
                    if 0 < i <= len(all_shards)]
    else:
        selected = all_shards

    # Save the index.json locally
    idx_path = dest_dir / "model.safetensors.index.json"
    with open(idx_path, "w") as f:
        json.dump(index, f)
    print(f"  Index saved: {idx_path}")
    print(f"  Downloading {len(selected)}/{len(all_shards)} shards...")

    # Build base URL from index URL
    base_url = index_url.rsplit("/", 1)[0]
    downloaded = []
    for shard_name in selected:
        shard_url  = f"{base_url}/{shard_name}"
        local_path = download_shard(shard_url, str(dest_dir),
                                     filename=shard_name, timeout=timeout)
        downloaded.append(local_path)

    print(f"  Done: {len(downloaded)} shards in {dest_dir}")
    return downloaded
