"""Torch datasets over SynthMixer.

Every item: wav float32 [S], length int (valid samples), target float32 [24, T] (common.targets.frame_targets),
clip_cats float32 [7], tier "synthetic", weight float, boundary_weight 1.0 (labels are sample-exact).
S = crop_seconds or max_seconds (default 10 s, zero padded); T = ceil(S / FRAME). Longer clips get a random
window; events cut by the window lose the cut boundary (frame_targets ignores boundaries at clip edges).

SynthIterableDataset  endless (or epoch_size) stream, RNG per (seed, epoch, rank, worker); call set_epoch(e)
SynthDataset          index-based: item i of epoch e is always the same mixture
OfflineSynthDataset   reads shards written by synthgen.write_offline_set (memory-mapped, no decoding)
"""
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from common.contract import FRAME, SR
from common.targets import clip_categories, frame_targets

HOP = int(round(FRAME * SR))


def to_item(wav: np.ndarray, events, rng: np.random.Generator, max_seconds: float = 10.0,
            crop_seconds: float | None = None, weight: float = 1.0, sigma_frames: float = 1.0,
            clip_edge_boundaries: bool = False) -> dict:
    """Window/pad one rendered clip and build its training dict."""
    win = int(round((crop_seconds or max_seconds) * SR))
    start = int(rng.integers(0, len(wav) - win + 1)) if len(wav) > win else 0
    x = np.asarray(wav[start:start + win], dtype=np.float32)
    valid, t0 = len(x), start / SR
    dur = valid / SR
    ev = [(max(0.0, a - t0), min(dur, b - t0), cat, tag) for a, b, cat, tag in events if b - t0 > 0 and a - t0 < dur]
    y = frame_targets(ev, dur, sigma_frames=sigma_frames, n=-(-win // HOP), clip_edge_boundaries=clip_edge_boundaries)
    out = np.zeros(win, dtype=np.float32)
    out[:valid] = x
    return dict(wav=torch.from_numpy(out), length=valid, target=torch.from_numpy(y),
                clip_cats=torch.from_numpy(clip_categories(ev)), tier="synthetic", weight=float(weight),
                boundary_weight=1.0)


def _shard_info():
    """(seed components, shard index, number of shards) for the current DDP rank and DataLoader worker."""
    info = get_worker_info()
    wid, nw = (info.id, info.num_workers) if info else (0, 1)
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)
    return [rank, wid], rank * nw + wid, nw * world


class _Base:
    def __init__(self, seed=0, max_seconds=10.0, crop_seconds=None, weight=1.0, **target_kw):
        self.seed, self.epoch = int(seed), 0
        self.item_kw = dict(max_seconds=max_seconds, crop_seconds=crop_seconds, weight=weight, **target_kw)

    def set_epoch(self, epoch: int):
        """Call before each epoch (with persistent_workers=False) so epochs see different mixtures."""
        self.epoch = int(epoch)


class SynthIterableDataset(_Base, IterableDataset):
    def __init__(self, mixer, epoch_size: int | None = None, **kw):
        _Base.__init__(self, **kw)
        self.mixer, self.epoch_size = mixer, epoch_size

    def __iter__(self):
        ids, shard, nshard = _shard_info()
        rng = np.random.default_rng([self.seed, self.epoch, *ids])
        count = None if self.epoch_size is None else len(range(shard, self.epoch_size, nshard))
        i = 0
        while count is None or i < count:
            r = self.mixer.sample(rng)
            yield to_item(r["wav"], r["events"], rng, **self.item_kw)
            i += 1


class SynthDataset(_Base, Dataset):
    def __init__(self, mixer, size: int, **kw):
        _Base.__init__(self, **kw)
        self.mixer, self.size = mixer, int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        r = self.mixer.sample(rng)
        return to_item(r["wav"], r["events"], rng, **self.item_kw)


def npz_memmap(path, key: str) -> np.ndarray:
    """Memory-map one array stored uncompressed (np.savez) inside an .npz; falls back to a full load."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path, allow_pickle=False) as z:
            return z[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


class OfflineSynthDataset(_Base, Dataset):
    """Pre-rendered shards (shard_*.npz: wav float16 [total], offsets int64 [K+1], events JSON).
    Crops/pads with an RNG seeded by (seed, epoch, index)."""

    def __init__(self, root, **kw):
        _Base.__init__(self, **kw)
        self.shards = sorted(Path(root).glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no shard_*.npz under {root}")
        self.index, self.events, self.offsets = [], [], []
        for s, p in enumerate(self.shards):
            with np.load(p, allow_pickle=False) as z:
                ev, off = json.loads(str(z["events"])), z["offsets"].astype(np.int64)
            self.events += [[tuple(e) for e in clip] for clip in ev]
            self.offsets.append(off)
            self.index += [(s, j) for j in range(len(off) - 1)]
        self._wav = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, j = self.index[i]
        if s not in self._wav:
            self._wav[s] = npz_memmap(self.shards[s], "wav")
        off = self.offsets[s]
        wav = np.asarray(self._wav[s][off[j]:off[j + 1]], dtype=np.float32)
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        return to_item(wav, self.events[i], rng, **self.item_kw)

    def __getstate__(self):  # memmaps are reopened in each worker
        return {**self.__dict__, "_wav": {}}
