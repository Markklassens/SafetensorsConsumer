"""
safetensors_loader.py v4 — What actually helps the LLM vs v3.

ROOT CAUSE OF v3 FAILURE:
  Storing "Tensor: model.layers.0.q_proj.weight from llama" as a fact
  gives the LLM zero useful information. Weight matrices are not readable
  knowledge — they're floating point numbers that only make sense through
  the forward pass.

WHAT ACTUALLY HELPS:
  1. Architecture knowledge graph — extracted from tensor names:
     - Layer count, attention heads, hidden dim, vocab size
     - These ARE human-interpretable and useful for the LLM to understand
       the model it's running alongside
  2. Model capability profile — derived from architecture:
     - Context length estimate, parameter count, known model family
  3. Fast tensor index — for on-demand lazy loading of specific weights
  4. Shard manifest — so sharded models load in correct order
  5. Performance profile — what layers are slow, what's quantized

The loader now stores ARCHITECTURE FACTS that an LLM can actually use,
not raw tensor descriptions.
"""
from __future__ import annotations
import hashlib, json, math, os, struct, threading, time, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Callable
import numpy as np

from core.config import Config
from core.embeddings import EmbeddingEngine

log = logging.getLogger("srhn")

DTYPE_BYTES: dict[str, float] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "I4": 0.5, "U8": 1, "BOOL": 0.125,
}


# ── Architecture extraction ──────────────────────────────────────────────────

def _infer_architecture(tensors: dict) -> dict:
    """
    Derive human-readable architecture facts from tensor shapes.
    These are what the LLM can actually USE in its context.
    """
    keys  = list(tensors.keys())
    facts = {}

    # Vocab / embedding size
    for k in keys:
        if "embed" in k.lower() and "weight" in k.lower():
            shape = tensors[k].get("shape", [])
            if len(shape) == 2:
                facts["vocab_size"]  = max(shape)
                facts["hidden_dim"]  = min(shape)
                break

    # Layer count
    layer_nums = set()
    for k in keys:
        parts = k.split(".")
        for p in parts:
            if p.isdigit():
                layer_nums.add(int(p))
    if layer_nums:
        facts["n_layers"] = max(layer_nums) + 1

    # Attention heads (from q_proj shape)
    for k in keys:
        if ("q_proj" in k or "query" in k.lower()) and "weight" in k:
            shape = tensors[k].get("shape", [])
            if len(shape) == 2 and "hidden_dim" in facts:
                out_dim = shape[0]
                head_dim = 64  # typical
                facts["n_heads"]   = max(1, out_dim // head_dim)
                facts["head_dim"]  = head_dim
            break

    # KV heads (GQA)
    for k in keys:
        if ("k_proj" in k or "key" in k.lower()) and "weight" in k:
            shape = tensors[k].get("shape", [])
            if len(shape) == 2 and "n_heads" in facts:
                kv_heads = max(1, shape[0] // facts.get("head_dim", 64))
                if kv_heads != facts.get("n_heads"):
                    facts["n_kv_heads"] = kv_heads  # GQA
                    facts["uses_gqa"]   = True
            break

    # FFN dim
    for k in keys:
        if ("mlp" in k.lower() or "ffn" in k.lower()) and ("gate" in k or "up" in k):
            shape = tensors[k].get("shape", [])
            if len(shape) == 2:
                facts["ffn_dim"] = max(shape)
                break

    # Total params
    total = sum(math.prod(v.get("shape", [0])) for v in tensors.values()
                if isinstance(v, dict) and v.get("shape"))
    facts["total_params"] = total
    facts["params_M"]     = round(total / 1e6, 1)
    facts["params_B"]     = round(total / 1e9, 2)

    # Estimate context length from RoPE/positional info (rough heuristic)
    if "n_layers" in facts and "hidden_dim" in facts:
        h = facts["hidden_dim"]
        facts["estimated_context"] = (
            4096  if h <= 2048 else
            8192  if h <= 4096 else
            32768 if h <= 8192 else
            131072
        )

    # Quantization
    dtypes_used = set()
    for v in tensors.values():
        if isinstance(v, dict):
            dtypes_used.add(v.get("dtype", "F32"))
    facts["dtypes"]        = list(dtypes_used)
    facts["is_quantized"]  = any(d in {"I8","I4","NF4"} for d in dtypes_used)
    facts["is_fp16"]       = "F16" in dtypes_used or "BF16" in dtypes_used

    # Model family heuristics
    if total < 1e9:
        facts["size_class"] = "tiny (<1B)"
    elif total < 4e9:
        facts["size_class"] = "small (1-4B)"
    elif total < 10e9:
        facts["size_class"] = "medium (4-10B)"
    elif total < 30e9:
        facts["size_class"] = "large (10-30B)"
    else:
        facts["size_class"] = "very large (>30B)"

    return facts


def _architecture_to_facts(arch: dict, model_name: str) -> list[str]:
    """Convert architecture dict into natural language facts for memory injection."""
    facts = []
    if arch.get("params_B", 0) > 0:
        facts.append(
            f"{model_name} has {arch['params_B']}B parameters "
            f"({arch.get('size_class','unknown size')})")
    if "n_layers" in arch:
        facts.append(f"{model_name} has {arch['n_layers']} transformer layers")
    if "hidden_dim" in arch:
        facts.append(f"{model_name} hidden dimension is {arch['hidden_dim']}")
    if "n_heads" in arch:
        gqa = f", {arch['n_kv_heads']} KV heads (GQA)" if arch.get("uses_gqa") else ""
        facts.append(f"{model_name} has {arch['n_heads']} attention heads{gqa}")
    if "vocab_size" in arch:
        facts.append(f"{model_name} vocabulary size is {arch['vocab_size']:,} tokens")
    if "ffn_dim" in arch:
        facts.append(f"{model_name} FFN intermediate dimension is {arch['ffn_dim']}")
    if "estimated_context" in arch:
        facts.append(
            f"{model_name} estimated context length is {arch['estimated_context']:,} tokens")
    if arch.get("is_quantized"):
        facts.append(f"{model_name} is quantized (dtypes: {arch.get('dtypes',[])})")
    if arch.get("is_fp16"):
        facts.append(f"{model_name} uses half-precision (FP16/BF16) weights")
    return facts


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TensorInfo:
    key:        str
    dtype:      str
    shape:      list[int]
    n_params:   int
    file_path:  str
    data_start: int
    data_end:   int
    size_bytes: int = 0

    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class ModelState:
    model_id:     str
    name:         str
    total_params: int                 = 0
    tensors:      dict[str, TensorInfo] = field(default_factory=dict)
    shards:       list[str]           = field(default_factory=list)
    architecture: dict                = field(default_factory=dict)
    arch_facts:   list[str]           = field(default_factory=list)
    created_at:   float               = field(default_factory=time.time)
    updated_at:   float               = field(default_factory=time.time)
    metadata:     dict                = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "model_id":    self.model_id,
            "name":        self.name,
            "tensors":     len(self.tensors),
            "params_M":    round(self.total_params / 1e6, 2),
            "n_layers":    self.architecture.get("n_layers", 0),
            "hidden_dim":  self.architecture.get("hidden_dim", 0),
            "n_heads":     self.architecture.get("n_heads", 0),
            "size_class":  self.architecture.get("size_class", "unknown"),
            "quantized":   self.architecture.get("is_quantized", False),
            "shards":      len(self.shards),
            "arch_facts":  len(self.arch_facts),
            "updated_at":  self.updated_at,
        }


# ── Loader ───────────────────────────────────────────────────────────────────

class SafetensorsLoader:
    """
    Production safetensors loader.

    Key improvements over v3:
    - Extracts real architecture knowledge (layer count, heads, dim, vocab)
    - Converts architecture to LLM-usable natural language facts
    - Fast indexing: reads header only, no tensor data loaded during scan
    - Lazy tensor loading: weights loaded only when explicitly requested
    - Timeout-free: no background threads per tensor, just header parsing
    """

    def __init__(self, cfg: Optional[Config] = None,
                 embed_engine: Optional[EmbeddingEngine] = None):
        self._cfg    = cfg or Config()
        self._embed  = embed_engine or EmbeddingEngine(self._cfg)
        self._models: dict[str, ModelState] = {}
        self._lock   = threading.RLock()
        self._reg    = str(Path(self._cfg.store_dir) / self._cfg.loader_registry)
        os.makedirs(self._cfg.store_dir, exist_ok=True)
        self._load_registry()

    # ── Registry ─────────────────────────────────────────────────────────────

    def _load_registry(self):
        if not os.path.exists(self._reg):
            return
        try:
            with open(self._reg) as f:
                raw = json.load(f)
            for mid, d in raw.items():
                ms = ModelState(
                    model_id=d["model_id"], name=d["name"],
                    total_params=d.get("total_params", 0),
                    shards=d.get("shards", []),
                    architecture=d.get("architecture", {}),
                    arch_facts=d.get("arch_facts", []),
                    created_at=d.get("created_at", time.time()),
                    updated_at=d.get("updated_at", time.time()),
                    metadata=d.get("metadata", {}),
                )
                for k, v in d.get("tensors", {}).items():
                    ms.tensors[k] = TensorInfo(**v)
                self._models[mid] = ms
            log.info(f"Registry: {len(self._models)} models")
        except Exception as e:
            log.warning(f"Registry load failed: {e}")

    def _save_registry(self):
        data = {}
        for mid, ms in self._models.items():
            data[mid] = {
                "model_id": ms.model_id, "name": ms.name,
                "total_params": ms.total_params, "shards": ms.shards,
                "architecture": ms.architecture, "arch_facts": ms.arch_facts,
                "created_at": ms.created_at, "updated_at": ms.updated_at,
                "metadata": ms.metadata,
                "tensors": {k: t.__dict__ for k, t in ms.tensors.items()},
            }
        tmp = self._reg + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._reg)

    # ── Header parsing ────────────────────────────────────────────────────────

    def _parse_header(self, path: str) -> tuple[dict, int]:
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        file_size = os.path.getsize(path)
        if file_size < 8:
            raise ValueError(f"Too small ({file_size}B): {path}")
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size == 0 or header_size > min(file_size - 8, 500_000_000):
                raise ValueError(f"Invalid header size {header_size}: {path}")
            raw = f.read(header_size)
        try:
            return json.loads(raw), 8 + header_size
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSON header: {e}")

    # ── Add shards ────────────────────────────────────────────────────────────

    def add_shard(self, path: str, model_id: Optional[str] = None,
                  model_name: str = "",
                  progress_cb: Optional[Callable[[int, int, str], None]] = None
                  ) -> ModelState:
        """
        Add one shard. Fast: only reads header, not tensor data.
        Call get_arch_facts() after all shards are loaded to get LLM-usable facts.
        """
        path     = str(Path(path).resolve())
        model_id = model_id or hashlib.sha256(path.encode()).hexdigest()[:8]
        name     = model_name or Path(path).parent.name or model_id

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        header, data_offset = self._parse_header(path)
        tensor_keys = [k for k in header if k != "__metadata__"]
        metadata    = header.get("__metadata__", {})

        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = ModelState(
                    model_id=model_id, name=name)
            ms = self._models[model_id]
            ms.metadata.update(metadata)
            if path not in ms.shards:
                ms.shards.append(path)

        total   = len(tensor_keys)
        added   = 0
        skipped = 0

        for i, key in enumerate(tensor_keys):
            if progress_cb:
                try: progress_cb(i, total, key)
                except Exception: pass

            info    = header[key]
            shape   = info.get("shape", [])
            dtype   = info.get("dtype", "F32").upper()
            offsets = info.get("data_offsets")

            if not shape or any(d == 0 for d in shape):
                skipped += 1; continue
            if offsets is None or len(offsets) < 2:
                skipped += 1; continue

            n_params   = math.prod(shape)
            if n_params < 4:
                skipped += 1; continue

            start, end = int(offsets[0]), int(offsets[1])
            if start >= end:
                skipped += 1; continue

            size_bytes = int(n_params * DTYPE_BYTES.get(dtype, 4))
            ti = TensorInfo(
                key=key, dtype=dtype, shape=shape, n_params=n_params,
                file_path=path, data_start=data_offset + start,
                data_end=data_offset + end, size_bytes=size_bytes,
            )
            with self._lock:
                if key not in ms.tensors:
                    ms.total_params += n_params
                ms.tensors[key] = ti
                ms.updated_at   = time.time()
            added += 1

        # Extract architecture from header (fast, no data loading)
        with self._lock:
            arch = _infer_architecture(
                {k: v for k, v in header.items() if k != "__metadata__"})
            ms.architecture.update(arch)
            ms.arch_facts = _architecture_to_facts(ms.architecture, ms.name)

        self._save_registry()
        log.info(f"Shard: {Path(path).name} → {added} tensors indexed, "
                 f"{ms.total_params/1e6:.1f}M params, {skipped} skipped")
        return ms

    def add_directory(self, dir_path: str, model_id: Optional[str] = None,
                       model_name: str = "",
                       progress_cb: Optional[Callable] = None) -> ModelState:
        d    = Path(dir_path)
        mid  = model_id or hashlib.sha256(str(d).encode()).hexdigest()[:8]
        name = model_name or d.name

        idx_file = d / "model.safetensors.index.json"
        if idx_file.exists():
            with open(idx_file) as f:
                idx = json.load(f)
            shards = list(dict.fromkeys(idx.get("weight_map", {}).values()))
            files  = [d / s for s in shards
                      if (d / s).exists() and s.endswith(".safetensors")]
        else:
            files = sorted(d.glob("*.safetensors"))

        if not files:
            raise ValueError(f"No .safetensors files found in {dir_path}")

        ms = None
        for i, f in enumerate(files):
            log.info(f"Shard {i+1}/{len(files)}: {f.name}")
            ms = self.add_shard(str(f), model_id=mid, model_name=name,
                                progress_cb=progress_cb)
        return ms or self._models.get(mid, ModelState(model_id=mid, name=name))

    # ── Knowledge extraction ─────────────────────────────────────────────────

    def get_arch_facts(self, model_id: str) -> list[str]:
        """
        Get natural language architecture facts.
        THESE are what you inject into agent memory — not tensor descriptions.
        """
        ms = self._models.get(model_id)
        return ms.arch_facts if ms else []

    def get_arch_embeddings(self, model_id: str) -> tuple[list[str], np.ndarray]:
        """
        Get architecture facts with their embeddings for memory injection.
        Returns (facts, embedding_matrix).
        """
        ms = self._models.get(model_id)
        if not ms or not ms.arch_facts:
            return [], np.empty((0, self._embed.dim), dtype=np.float32)
        embs = self._embed.embed_batch(ms.arch_facts)
        return ms.arch_facts, embs

    def get_layer_profile(self, model_id: str) -> list[str]:
        """
        Generate layer-level profile facts (what each layer type does).
        Useful as system context for the LLM.
        """
        ms = self._models.get(model_id)
        if not ms:
            return []
        layer_types: dict[str, int] = {}
        for key in ms.tensors:
            if "embed" in key.lower():       layer_types["embedding"]  = layer_types.get("embedding",0)+1
            elif "attn" in key.lower() or "attention" in key.lower():
                                             layer_types["attention"]  = layer_types.get("attention",0)+1
            elif "mlp" in key.lower() or "ffn" in key.lower():
                                             layer_types["feedforward"] = layer_types.get("feedforward",0)+1
            elif "norm" in key.lower():      layer_types["norm"]       = layer_types.get("norm",0)+1
            elif "head" in key.lower():      layer_types["output_head"] = layer_types.get("output_head",0)+1

        name = ms.name
        facts = [
            f"{name} architecture: {', '.join(f'{v} {k} layers' for k,v in layer_types.items())}",
            f"{name} has {ms.architecture.get('n_layers',0)} transformer blocks "
            f"with hidden_dim={ms.architecture.get('hidden_dim','?')}",
        ]
        if ms.architecture.get("is_quantized"):
            facts.append(f"{name} weights are quantized — faster inference, some quality tradeoff")
        if ms.architecture.get("uses_gqa"):
            facts.append(f"{name} uses grouped query attention (GQA) for efficient KV caching")
        return facts

    # ── Lazy tensor access ───────────────────────────────────────────────────

    def load_tensor(self, model_id: str, key: str) -> Optional[np.ndarray]:
        """On-demand lazy load of one tensor. RAM bounded to that one tensor."""
        ms = self._models.get(model_id)
        if not ms or key not in ms.tensors:
            return None
        ti = ms.tensors[key]
        try:
            dtype_str = ti.dtype.upper()
            is_bf16   = dtype_str == "BF16"
            is_i4     = dtype_str == "I4"
            n         = ti.n_params

            with open(ti.file_path, "rb") as f:
                f.seek(ti.data_start)
                raw = f.read(ti.data_end - ti.data_start)

            if not raw:
                return None

            if is_bf16:
                need = n * 2
                if len(raw) < need: return None
                u32  = np.frombuffer(raw[:need], dtype=np.uint16).astype(np.uint32) << 16
                arr  = u32.view(np.float32).copy()
            elif is_i4:
                need   = math.ceil(n / 2)
                if len(raw) < need: return None
                packed = np.frombuffer(raw[:need], dtype=np.uint8)
                lo     = (packed & 0x0F).astype(np.int8)
                hi     = (packed >> 4).astype(np.int8)
                arr    = np.stack([lo, hi], axis=1).flatten()[:n].astype(np.float32) / 8.0
            else:
                dtype_map = {
                    "F32": np.float32, "F64": np.float64, "F16": np.float16,
                    "I64": np.int64, "I32": np.int32, "I16": np.int16,
                    "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
                }
                np_dt = dtype_map.get(dtype_str, np.float32)
                need  = n * np.dtype(np_dt).itemsize
                if len(raw) < need: return None
                arr   = np.frombuffer(raw[:need], dtype=np_dt).astype(np.float32)

            if arr.size != n:
                arr = arr[:n] if arr.size > n else np.pad(arr, (0, n - arr.size))
            try:
                return arr.reshape(ti.shape)
            except ValueError:
                return arr
        except Exception as e:
            log.debug(f"load_tensor {key}: {e}")
            return None

    def iter_tensors(self, model_id: str,
                     filter_keys: Optional[list[str]] = None
                     ) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate tensors one at a time. No full model in RAM."""
        ms = self._models.get(model_id)
        if not ms: return
        for key, ti in ms.tensors.items():
            if filter_keys and key not in filter_keys: continue
            tensor = self.load_tensor(model_id, key)
            if tensor is not None:
                yield key, tensor
                del tensor

    # ── Info ─────────────────────────────────────────────────────────────────

    def get_model(self, model_id: str) -> Optional[ModelState]:
        return self._models.get(model_id)

    def list_models(self) -> list[dict]:
        with self._lock:
            return [ms.summary() for ms in self._models.values()]

    def delete_model(self, model_id: str) -> bool:
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                self._save_registry()
                return True
        return False
