"""Small shared helpers: atomic writes, distributed helpers, RNG state, time parsing."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------- atomic files
def atomic_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".partial",
                                     delete=False, encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    os.replace(tmp, path)


def append_jsonl(path, row) -> None:
    with Path(path).open("a", encoding="utf8") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def atomic_torch_save(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Not JSON serialisable: {type(x)}")


def file_sha256(path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        read = 0
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()


# ----------------------------------------------------------------------------- distributed
def ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if ddp_active() else 0


def world_size() -> int:
    return dist.get_world_size() if ddp_active() else 1


def is_main() -> bool:
    return rank() == 0


def setup_device(device_name: str | None = None) -> torch.device:
    """DDP when launched by torchrun (LOCAL_RANK set), otherwise a single device."""
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("LOCAL_RANK is set (torchrun) but CUDA is unavailable")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(minutes=30))
        return torch.device("cuda", local)
    name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def barrier() -> None:
    if ddp_active():
        dist.barrier()


# ----------------------------------------------------------------------------- RNG
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError) as exc:  # different GPU count after resume
            print(f"[resume] CUDA RNG state not restored ({exc}); continuing with fresh CUDA RNG", flush=True)


# ----------------------------------------------------------------------------- time
def parse_deadline(text: str) -> float:
    """ISO-8601 with an explicit UTC offset, e.g. 2026-10-17T09:30:00+05:30 -> POSIX seconds."""
    try:
        when = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"--deadline must be ISO-8601 (e.g. 2026-10-17T09:30:00+05:30), got {text!r}") from exc
    if when.tzinfo is None:
        raise ValueError(f"--deadline must include a UTC offset (e.g. +05:30), got {text!r}")
    return when.timestamp()


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
