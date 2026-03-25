"""
SRHN v5 — Progressive Safetensors Ingestor
===========================================
Converts any .safetensors file (single shard or sharded model directory)
into a RAM-bounded, edge-ready SRHN memory store.

What "progressive" means here:
  - Reads header only first (kilobytes, instant)
  - Streams tensor data one layer at a time
  - Peak RAM = max(one_tensor) not model_total
  - Writes a compact index (JSON, ~50KB per model)
  - Injects architecture facts into agent memory
  - Works on 512MB RAM devices

Usage:
  from core.ingestor import ingest
  result = ingest("model.safetensors", engine, loader)

  from core.ingestor import ingest_directory
  result = ingest_directory("/models/llama-3.2-1b/", engine, loader)
"""
from __future__ import annotations
import json, os, struct, time, logging, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import numpy as np

from core.config import Config
from core.engine import SRHNEngine
from core.safetensors_loader import SafetensorsLoader, DTYPE_BYTES

log = logging.getLogger("srhn.ingestor")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    model_id:      str
    model_name:    str
    shards:        int
    tensors:       int
    params_M:      float
    params_B:      float
    size_mb:       float
    arch_facts:    list[str]
    layer_profile: list[str]
    facts_injected: int
    elapsed_s:     float
    peak_ram_mb:   float
    warnings:      list[str]
    architecture:  dict

    def summary(self) -> str:
        lines = [
            f"Model:     {self.model_name} ({self.model_id})",
            f"Params:    {self.params_B:.2f}B  ({self.params_M:.0f}M)",
            f"Tensors:   {self.tensors} across {self.shards} shard(s)",
            f"Size:      {self.size_mb:.1f} MB on disk",
            f"Peak RAM:  {self.peak_ram_mb:.1f} MB during ingest",
            f"Injected:  {self.facts_injected} facts into agent memory",
            f"Time:      {self.elapsed_s:.1f}s",
        ]
        if self.architecture:
            arch = self.architecture
            lines.append(
                f"Arch:      {arch.get('n_layers','?')} layers, "
                f"hidden={arch.get('hidden_dim','?')}, "
                f"heads={arch.get('n_heads','?')}, "
                f"vocab={arch.get('vocab_size','?'):,}" if arch.get('vocab_size') else
                f"Arch:      {arch.get('n_layers','?')} layers, hidden={arch.get('hidden_dim','?')}"
            )
        if self.warnings:
            lines.append(f"Warnings:  {len(self.warnings)}")
            for w in self.warnings[:3]:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ─────────────────────────────────────────────────────────────────────────────

class ProgressBar:
    """Minimal progress bar that works on any terminal including SSH."""
    def __init__(self, total: int, label: str = "", width: int = 40,
                 silent: bool = False):
        self.total   = max(total, 1)
        self.label   = label
        self.width   = width
        self.silent  = silent
        self._start  = time.time()
        self._last_n = -1

    def update(self, n: int, suffix: str = ""):
        if self.silent or n == self._last_n:
            return
        self._last_n = n
        frac  = min(n / self.total, 1.0)
        filled = int(self.width * frac)
        bar   = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self._start
        eta   = (elapsed / frac - elapsed) if frac > 0.01 else 0
        pct   = int(frac * 100)
        line  = f"\r  {self.label} [{bar}] {pct:3d}%"
        if suffix: line += f"  {suffix}"
        if eta > 0: line += f"  ETA {eta:.0f}s"
        print(line, end="", flush=True)

    def done(self, msg: str = ""):
        if not self.silent:
            elapsed = time.time() - self._start
            print(f"\r  {self.label} [{'█'*self.width}] 100%  {msg}  ({elapsed:.1f}s)",
                  flush=True)
            print()


# ─────────────────────────────────────────────────────────────────────────────
# Core ingest logic
# ─────────────────────────────────────────────────────────────────────────────

def _parse_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"File too small: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size == 0 or header_size > 500_000_000:
            raise ValueError(f"Invalid header size {header_size}")
        header_raw = f.read(header_size)
    return json.loads(header_raw), 8 + header_size


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def _estimate_peak_ram(tensors: dict) -> float:
    """Largest single tensor in MB — that's peak RAM during streaming."""
    max_bytes = 0
    for k, v in tensors.items():
        if k == "__metadata__":
            continue
        shape  = v.get("shape", [])
        dtype  = v.get("dtype", "F32").upper()
        if not shape: continue
        nbytes = math.prod(shape) * DTYPE_BYTES.get(dtype, 4)
        max_bytes = max(max_bytes, nbytes)
    return max_bytes / (1024 * 1024)


def ingest(
    path: str,
    engine: SRHNEngine,
    loader: SafetensorsLoader,
    model_name: str = "",
    model_id: Optional[str] = None,
    silent: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> IngestResult:
    """
    Ingest a single .safetensors file into the SRHN engine.

    Steps:
      1. Parse header (metadata only, no tensor data)
      2. Index all tensor locations
      3. Infer architecture from tensor shapes
      4. Inject human-readable architecture facts into agent memory
      5. Save index to registry

    RAM usage: O(header_size) — typically <5MB even for 70B models.
    The actual tensor data stays on disk until explicitly requested.
    """
    t0       = time.time()
    path     = str(Path(path).resolve())
    warnings = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    size_mb = _file_size_mb(path)
    if not silent:
        print(f"\n  Ingesting: {Path(path).name}  ({size_mb:.1f} MB)")

    # Step 1: parse header
    try:
        header, data_offset = _parse_header(path)
    except Exception as e:
        raise ValueError(f"Cannot parse {path}: {e}")

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("No tensors found in file")

    peak_ram = _estimate_peak_ram(tensors)
    if not silent:
        print(f"  Tensors in file: {len(tensors)}")
        print(f"  Peak RAM needed: {peak_ram:.1f} MB (streaming one tensor at a time)")

    # Step 2+3: index via loader (fast — header only)
    total = len(tensors)
    pb    = ProgressBar(total, "Indexing", silent=silent)

    def _cb(i: int, n: int, key: str):
        pb.update(i, suffix=key.split(".")[-1][:20])
        if on_progress:
            try: on_progress(i, n, key)
            except Exception: pass

    ms = loader.add_shard(path, model_id=model_id,
                          model_name=model_name or Path(path).parent.name,
                          progress_cb=_cb)
    pb.done(f"{len(ms.tensors)} tensors indexed")

    # Step 4: inject architecture facts
    arch_facts   = loader.get_arch_facts(ms.model_id)
    layer_profile = loader.get_layer_profile(ms.model_id)

    if not silent:
        print(f"  Architecture facts extracted: {len(arch_facts)}")
        for fact in arch_facts:
            print(f"    {fact}")

    n_injected = engine.inject_model_knowledge(arch_facts, layer_profile)

    # Warn if tensors are on a path that may not persist
    if "/tmp/" in path or "\\temp\\" in path.lower():
        warnings.append("Safetensors file is in a temp directory — registry will lose it on reboot")

    elapsed = time.time() - t0
    result  = IngestResult(
        model_id=ms.model_id,
        model_name=ms.name,
        shards=len(ms.shards),
        tensors=len(ms.tensors),
        params_M=round(ms.total_params / 1e6, 1),
        params_B=round(ms.total_params / 1e9, 2),
        size_mb=size_mb,
        arch_facts=arch_facts,
        layer_profile=layer_profile,
        facts_injected=n_injected,
        elapsed_s=round(elapsed, 2),
        peak_ram_mb=round(peak_ram, 1),
        warnings=warnings,
        architecture=ms.architecture,
    )

    if not silent:
        print(f"\n  Done in {elapsed:.1f}s — {n_injected} facts in agent memory\n")

    return result


def ingest_directory(
    dir_path: str,
    engine: SRHNEngine,
    loader: SafetensorsLoader,
    model_name: str = "",
    model_id: Optional[str] = None,
    silent: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> IngestResult:
    """
    Ingest a sharded model directory (multiple .safetensors files).

    Handles:
      - model.safetensors.index.json (HuggingFace sharding format)
      - Plain directory of *.safetensors files (sorted order)
      - Mixed dtypes across shards
      - Partial downloads (warns about missing shards)
    """
    d    = Path(dir_path)
    t0   = time.time()
    warnings = []

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Resolve shard order
    idx_file = d / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_files = list(dict.fromkeys(weight_map.values()))
        files = [d / s for s in shard_files]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            warnings.append(f"Missing shards: {missing}")
            files = [f for f in files if f.exists()]
        if not silent:
            print(f"\n  Sharded model: {d.name}")
            print(f"  Shards (from index): {len(files)} of {len(shard_files)}")
    else:
        files = sorted(d.glob("*.safetensors"))
        if not files:
            raise ValueError(f"No .safetensors files in {dir_path}")
        if not silent:
            print(f"\n  Model directory: {d.name}")
            print(f"  Shards found: {len(files)} (no index.json, using glob order)")
            warnings.append("No model.safetensors.index.json — shard order may be wrong")

    total_size = sum(_file_size_mb(str(f)) for f in files)
    if not silent:
        print(f"  Total size: {total_size:.1f} MB across {len(files)} shards")

    # Ingest shards one at a time
    mid  = model_id or None
    name = model_name or d.name
    last_result: Optional[IngestResult] = None

    for i, shard in enumerate(files):
        if not silent:
            print(f"\n  [{i+1}/{len(files)}] {shard.name}  ({_file_size_mb(str(shard)):.1f} MB)")
        r = ingest(
            str(shard), engine, loader,
            model_name=name, model_id=mid,
            silent=silent, on_progress=on_progress,
        )
        mid         = r.model_id   # keep same ID across shards
        last_result = r

    if last_result is None:
        raise RuntimeError("No shards ingested")

    # Re-fetch final state after all shards
    ms           = loader.get_model(mid)
    arch_facts   = loader.get_arch_facts(mid) if ms else []
    layer_profile = loader.get_layer_profile(mid) if ms else []
    n_injected   = engine.inject_model_knowledge(arch_facts, layer_profile)

    elapsed = time.time() - t0
    result  = IngestResult(
        model_id=mid,
        model_name=name,
        shards=len(files),
        tensors=len(ms.tensors) if ms else 0,
        params_M=round(ms.total_params / 1e6, 1) if ms else 0,
        params_B=round(ms.total_params / 1e9, 2) if ms else 0,
        size_mb=round(total_size, 1),
        arch_facts=arch_facts,
        layer_profile=layer_profile,
        facts_injected=n_injected,
        elapsed_s=round(elapsed, 2),
        peak_ram_mb=round(last_result.peak_ram_mb, 1),
        warnings=warnings + last_result.warnings,
        architecture=ms.architecture if ms else {},
    )

    if not silent:
        print(f"\n  All shards ingested in {elapsed:.1f}s")
        print(f"  Total tensors: {result.tensors}")
        print(f"  Total params:  {result.params_B:.2f}B")
        print(f"  Facts in memory: {n_injected}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: ingest anything (auto-detect file vs directory)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_any(
    path: str,
    engine: SRHNEngine,
    loader: SafetensorsLoader,
    model_name: str = "",
    silent: bool = False,
) -> IngestResult:
    """
    Auto-detect whether path is a file or directory and ingest accordingly.
    This is the recommended entry point.
    """
    p = Path(path)
    if p.is_dir():
        return ingest_directory(str(p), engine, loader,
                                model_name=model_name, silent=silent)
    elif p.is_file() and p.suffix == ".safetensors":
        return ingest(str(p), engine, loader,
                      model_name=model_name, silent=silent)
    else:
        # Try finding safetensors in the given path's parent
        candidates = list(p.parent.glob("*.safetensors"))
        if candidates:
            log.warning(f"Path {path} not found, using directory {p.parent}")
            return ingest_directory(str(p.parent), engine, loader,
                                    model_name=model_name, silent=silent)
        raise FileNotFoundError(
            f"No .safetensors file or directory at: {path}\n"
            "  Expected: path/to/model.safetensors  OR  path/to/model_dir/")
