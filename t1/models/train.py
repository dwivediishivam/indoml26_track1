"""Train one SED candidate against a wall-clock deadline.

  python -m models.train --config models/configs/m1_atst.yaml --out ~/indoml-work/runs_t1f/m1_atst \
      --deadline 2026-10-17T09:30:00+05:30 [--resume] [--staged-dir D] [--val-dir D] [--work-dir D] \
      [--set optim.batch_size=12 ...]

Stage 1: encoder frozen for optim.freeze_steps (new params only, head LR). Stage 2: encoder unfrozen with
layer-wise LR decay. Cosine schedules are fitted to a step budget derived from measured throughput and
the deadline (re-planned after a benchmark in each stage and after the first monitor).
DDP when launched by torchrun (LOCAL_RANK set); batch_size is per GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import nn  # noqa: E402

from models.config import load_config  # noqa: E402
from models.data import build_sources, filter_augment, freq_mixstyle, make_loader, set_data_roots  # noqa: E402
from models.losses import compute_losses  # noqa: E402
from models.model import build_model  # noqa: E402
from models.monitor import evaluate, load_monitor_audio, monitor_ids  # noqa: E402
from models.utils import (append_jsonl, atomic_json, atomic_torch_save, barrier, ddp_active, is_main, iso,  # noqa: E402
                          parse_deadline, rank, seed_everything, setup_device, world_size)

FORMAT = "t1f_sed_v1"


# ============================================================================ parameter groups
def _params_of(group):
    if isinstance(group, nn.Parameter):
        return [group]
    if isinstance(group, nn.Module):
        return list(group.parameters())
    out = []
    for g in group:
        out.extend(_params_of(g))
    return out


def build_param_groups(model, ocfg):
    """Head/new params at head_lr; encoder params per layer group at encoder_lr * decay^(depth from top)."""
    enc = list(dict.fromkeys(model.encoder_parameters()))
    enc_ids = {id(p) for p in enc}
    bb_new_ids = {id(p) for p in model.backbone.new_parameters()}
    if enc_ids & bb_new_ids:
        raise ValueError("backbone encoder_parameters() and new_parameters() overlap")
    for n, p in model.backbone.named_parameters():
        if p.requires_grad and id(p) not in enc_ids | bb_new_ids:
            raise ValueError(f"backbone parameter {n} is in neither encoder_parameters() nor new_parameters()")
    new = [p for p in dict.fromkeys(model.new_parameters()) if p.requires_grad]
    enc_train = [p for p in enc if p.requires_grad]     # params the backbone froze itself stay frozen
    layer_of = {}
    groups = model.layer_groups()
    for gi, g in enumerate(groups):
        for p in _params_of(g):
            if id(p) not in enc_ids:
                raise ValueError(f"layer_groups()[{gi}] contains a parameter outside encoder_parameters()")
            layer_of.setdefault(id(p), gi)
    n_layers = max(1, len(groups))
    leftover = [p for p in enc_train if id(p) not in layer_of]
    if leftover:
        print(f"[params] {len(leftover)} encoder tensors are not in layer_groups(); using the shallowest-layer LR",
              flush=True)
    by_layer = {}
    for p in enc_train:
        by_layer.setdefault(layer_of.get(id(p), 0), []).append(p)

    def split(params, kind, lr, name):
        out = []
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        for ps, wd, suffix in ((decay, ocfg["weight_decay"], ""), (no_decay, 0.0, "_nd")):
            if ps:
                out.append({"params": ps, "lr": 0.0, "base_lr": lr, "kind": kind, "weight_decay": wd, "name": name + suffix})
        return out

    pg = split(new, "head", ocfg["head_lr"], "head")
    for gi in sorted(by_layer):
        lr = ocfg["encoder_lr"] * ocfg["layer_decay"] ** (n_layers - 1 - gi)
        pg += split(by_layer[gi], "encoder", lr, f"enc{gi:02d}")
    n_new, n_enc = sum(p.numel() for p in new), sum(p.numel() for p in enc_train)
    print(f"[params] new/head {n_new / 1e6:.2f}M, encoder {n_enc / 1e6:.2f}M in {len(by_layer)} LR groups", flush=True)
    return pg, enc_train


def lr_multiplier(step, total, warmup, min_ratio):
    warm = min(1.0, (step + 1) / max(1, warmup))
    progress = min(1.0, max(0.0, step / max(1, total)))
    return warm * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ============================================================================ planning
class Planner:
    """Step budget = what fits before the deadline given measured s/step and monitor cost."""

    def __init__(self, cfg, deadline):
        o, t = cfg["optim"], cfg["train"]
        self.max_steps, self.freeze = o["max_steps"], o["freeze_steps"]
        self.deadline, self.reserve = deadline, t["reserve_seconds"]
        self.slow, self.every = t["stage2_slowdown_guess"], t["monitor_every"]
        self.rate = {"1": None, "2": None}
        self.monitor_s, self.monitor_measured = float(t["monitor_seconds_guess"]), False
        self.save_s = 30.0
        self.planned = self.max_steps

    def state_dict(self):
        return dict(rate=self.rate, monitor_s=self.monitor_s, monitor_measured=self.monitor_measured,
                    save_s=self.save_s, planned=self.planned)

    def load_state_dict(self, s):
        self.rate, self.monitor_s = dict(s["rate"]), s["monitor_s"]
        self.monitor_measured, self.save_s, self.planned = s["monitor_measured"], s["save_s"], s["planned"]

    def final_reserve(self):
        return max(self.reserve, 2.2 * self.monitor_s + 2 * self.save_s + 30)

    def plan(self, step, now):
        r1, r2 = self.rate["1"], self.rate["2"]
        if r2 is None and r1 is not None:
            r2 = r1 * self.slow
        if r2 is None:
            return self.planned
        if r1 is None:
            r1 = r2 / self.slow
        left = self.deadline - now - self.final_reserve()
        per_mon = self.monitor_s / self.every
        if step < self.freeze:
            t1 = (self.freeze - step) * (r1 + per_mon)
            steps = self.freeze + max(0.0, left - t1) / (r2 + per_mon)
        else:
            steps = step + max(0.0, left) / (r2 + per_mon)
        self.planned = int(max(step + 1, min(self.max_steps, steps)))
        return self.planned


# ============================================================================ checkpoint retention
class Retainer:
    """Keeps the best `keep` monitor-scored weights (student and EMA compete) + checkpoints/best.json."""

    def __init__(self, directory: Path, keep: int):
        self.dir, self.keep = Path(directory), keep
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "best.json"
        self.rows = json.loads(self.index.read_text())["checkpoints"] if self.index.exists() else []

    def offer(self, kind, step, metrics, state_dict, cfg):
        sc = float(metrics["combined"])
        if not math.isfinite(sc):
            raise ValueError(f"non-finite monitor score {sc}")
        name = f"{kind}_step{step:07d}.pt"
        rows = [r for r in self.rows if r["file"] != name]
        if len(rows) >= self.keep and sc <= min(r["combined"] for r in rows):
            return None
        atomic_torch_save(self.dir / name, {"format": FORMAT, "config": cfg, "weights": kind, "step": step,
                                            "metrics": metrics, "model": state_dict})
        rows.append({"file": name, "weights": kind, "step": step, "combined": sc, "metrics": metrics})
        rows.sort(key=lambda r: r["combined"], reverse=True)
        kept, dropped = rows[:self.keep], rows[self.keep:]
        atomic_json(self.index, {"checkpoints": kept})       # commit index before deleting losers
        for r in dropped:
            (self.dir / r["file"]).unlink(missing_ok=True)
        self.rows = kept
        return name


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", required=True, help="ISO-8601 with offset, e.g. 2026-10-17T09:30:00+05:30")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="start a fresh run from a retained checkpoint's weights "
                    "(same architecture); optimizer/schedule start fresh")
    ap.add_argument("--staged-dir", default=None, help="override data.staged_dir (READY/manifest.jsonl/audio.f16)")
    ap.add_argument("--val-dir", default=None, help="override data.val_dir (contains validationMetadata.json)")
    ap.add_argument("--work-dir", default=None, help="override data.work_dir (contains folds.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (YAML value)")
    return ap.parse_args(argv)


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p.detach(), 1.0 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def _config_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _config_diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [f"{path}: {a!r} -> {b!r}"]


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    set_data_roots(val_dir=args.val_dir or cfg["data"]["val_dir"], work_dir=args.work_dir or cfg["data"]["work_dir"])
    staged_dir = args.staged_dir or cfg["data"]["staged_dir"]
    deadline = parse_deadline(args.deadline)
    if deadline - time.time() < 600:
        raise ValueError(f"deadline {args.deadline} is less than 10 minutes away")
    device = setup_device(args.device)
    main_proc = is_main()
    out = Path(args.out).expanduser()
    latest = out / "latest.pt"
    if args.resume and not latest.exists():
        raise FileNotFoundError(f"--resume given but {latest} does not exist")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; pass --resume or choose a fresh --out")
    barrier()
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    o, t, a = cfg["optim"], cfg["train"], cfg["augment"]
    seed = int(cfg["seed"])
    seed_everything(seed + 1000 * rank())
    sources = build_sources(cfg, staged_dir, train=True)
    sizes = {k: len(v) for k, v in sources.items()}

    def log(state, **kw):
        if not main_proc:
            return
        row = dict(state=state, time=iso(time.time()), step=step, planned=planner.planned, stage=stage, **kw)
        atomic_json(out / "status.json", dict(row, deadline=iso(deadline), rates=planner.rate,
                                              monitor_seconds=planner.monitor_s, best=retainer.rows if retainer else []))
        append_jsonl(out / "history.jsonl", row)
        print(json.dumps(row, default=str), flush=True)

    mon_wavs = mon_val = None
    retainer = None
    if main_proc:
        ids, mon_val = monitor_ids(cfg["monitor"]["folds"], cfg["monitor"]["max_clips"])
        if "val_natural" in sources:
            leak = set(ids) & set(sources["val_natural"].ids)
            if leak:
                raise RuntimeError(f"{len(leak)} monitor clips are also val_natural training clips")
        mon_wavs = load_monitor_audio(ids, mon_val)
        retainer = Retainer(out / "checkpoints", t["keep_best"])

    model = build_model(cfg).to(device)
    groups, enc_train = build_param_groups(model, o)
    optimizer = torch.optim.AdamW(groups, lr=o["head_lr"], betas=(0.9, 0.999), eps=1e-8)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if main_proc else None
    planner = Planner(cfg, deadline)
    step, ema_updates, last_monitor, bad = 0, 0, -1, 0

    if args.init_from and not args.resume:
        s = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if s.get("format") != FORMAT:
            raise ValueError(f"{args.init_from} is not a {FORMAT} checkpoint")
        model.load_state_dict(s["model"])
        if main_proc:
            ema.load_state_dict(s["model"])
        print(f"[init] weights from {args.init_from} (step {s.get('step')}, {s.get('weights')})", flush=True)
        del s
    if args.resume:
        s = torch.load(latest, map_location="cpu", weights_only=False)
        if s.get("format") != FORMAT:
            raise ValueError(f"{latest} is not a {FORMAT} checkpoint")
        diff = _config_diff(s["config"], cfg)
        if diff:
            raise ValueError("resume config mismatch:\n  " + "\n  ".join(diff))
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        if main_proc:
            ema.load_state_dict(s["ema"])
        planner.load_state_dict(s["planner"])
        planner.deadline = deadline
        step, ema_updates, last_monitor = s["step"], s["ema_updates"], s["last_monitor"]
        del s
        planner.planned = planner.max_steps
        planner.plan(step, time.time())       # new deadline -> new budget (re-benchmarked below)
        seed_everything(seed + 1000 * rank() + step)
    if main_proc:
        atomic_json(out / "config.resolved.json", dict(config=cfg, staged_dir=str(staged_dir), sources=sizes,
                                                       world_size=world_size(), deadline=iso(deadline),
                                                       monitor_clips=len(mon_wavs)))
        excluded = {k: getattr(v, "excluded", 0) for k, v in sources.items()}
        print(f"[data] sources {sizes}; bronze excluded (too long) {excluded.get('bronze', 0)}; "
              f"monitor clips {len(mon_wavs)}", flush=True)

    stage = 1 if step < o["freeze_steps"] else 2
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def set_stage(st):
        if hasattr(model.backbone, "freeze_encoder"):     # also keeps frozen encoder modules in eval()
            model.backbone.freeze_encoder(st == 1)
        for p in enc_train:
            p.requires_grad_(st == 2)
        if ddp_active():
            return nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                       find_unused_parameters=t["ddp_find_unused"])
        return model

    net = set_stage(stage)
    loader = make_loader(sources, cfg, seed=seed + 7919 * rank() + step, pin_memory=device.type == "cuda")
    batches = iter(loader)

    stopping = {"signal": None}

    def on_signal(signum, _frame):
        stopping["signal"] = signal.Signals(signum).name

    old_handlers = {sg: signal.signal(sg, on_signal) for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    bench = {"stage": stage, "start": step + t["benchmark_warmup"], "t0": None, "excluded": 0.0, "done": False}
    last_save = time.time()
    t_start = time.time()

    def save_latest():
        t0 = time.time()
        atomic_torch_save(latest, {"format": FORMAT, "config": cfg, "step": step, "model": model.state_dict(),
                                   "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                                   "planner": planner.state_dict(), "ema_updates": ema_updates,
                                   "last_monitor": last_monitor, "saved": iso(time.time())})
        planner.save_s = max(5.0, time.time() - t0)
        atomic_json(out / "latest.json", {"step": step, "saved": iso(time.time()), "seconds": planner.save_s})

    def run_monitor():
        nonlocal last_monitor
        t0 = time.time()
        for kind, m in (("student", model), ("ema", ema)):
            r = evaluate(m, mon_wavs, mon_val, device, cfg["monitor"])
            kept = retainer.offer(kind, step, r, m.state_dict(), cfg)
            log("monitor", weights=kind, metrics=r, retained=kept)
        last_monitor = step
        took = time.time() - t0
        first = not planner.monitor_measured
        planner.monitor_s, planner.monitor_measured = took, True
        if first:
            planner.plan(step, time.time())
            log("replan", reason="first monitor", monitor_seconds=took)
        return took

    reason = None
    try:
        while True:
            now = time.time()
            # rank 0 owns the deadline / budget decision (it measures monitor + save costs); any rank may signal
            local_stop = stopping["signal"] is not None or (main_proc and (now >= deadline - planner.final_reserve()
                                                                           or step >= planner.planned))
            if ddp_active():
                flag = torch.tensor([float(local_stop), float(planner.planned if main_proc else 0)], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                local_stop, planner.planned = bool(flag[0] > 0), int(flag[1])
            if local_stop:
                reason = stopping["signal"] or ("step_budget" if step >= planner.planned else "deadline")
                break
            if stage == 1 and step >= o["freeze_steps"]:
                stage = 2
                net = set_stage(2)
                bench = {"stage": 2, "start": step + t["benchmark_warmup"], "t0": None, "excluded": 0.0, "done": False}
                log("unfreeze")

            hm = lr_multiplier(step, planner.planned, o["warmup_steps"], o["min_lr_ratio"])
            em = 0.0 if stage == 1 else lr_multiplier(step - o["freeze_steps"], planner.planned - o["freeze_steps"],
                                                      o["encoder_warmup_steps"], o["min_lr_ratio"])
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * (hm if g["kind"] == "head" else em)

            batch = _to_device(next(batches), device)
            wav = batch["wav"]
            with torch.no_grad():
                wav = filter_augment(wav, a["filter_aug_prob"], a["filter_aug_db"], a["filter_aug_bands"])
                wav = freq_mixstyle(wav, batch["length"], a["freq_mixstyle_prob"], a["freq_mixstyle_alpha"])
            net.train()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = net(wav, batch["length"])
            loss, parts = compute_losses(outputs, batch, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(trainable, o["grad_clip"])
            if not torch.isfinite(norm):
                bad += 1
                optimizer.zero_grad(set_to_none=True)
                log("skip_nonfinite", loss=float(loss.detach()), consecutive=bad)
                if bad >= o["max_bad_steps"]:
                    raise FloatingPointError(f"{bad} consecutive non-finite gradient steps")
                continue
            optimizer.step()
            bad = 0
            step += 1
            if main_proc:
                ema_updates += 1
                ema_update(ema, model, min(o["ema_decay"], (1 + ema_updates) / (10 + ema_updates)))

            # ---- throughput benchmark (per stage) -> re-plan the cosine budget
            if not bench["done"]:
                if step == bench["start"]:
                    bench["t0"] = time.time()
                elif bench["t0"] is not None and step == bench["start"] + t["benchmark_steps"]:
                    rate = (time.time() - bench["t0"] - bench["excluded"]) / t["benchmark_steps"]
                    planner.rate[str(bench["stage"])] = rate
                    bench["done"] = True
                    planner.plan(step, time.time())
                    log("replan", reason=f"stage{bench['stage']} benchmark", seconds_per_step=rate,
                        samples_per_second=o["batch_size"] * world_size() / rate)

            if step % t["log_every"] == 0 or step == 1:
                log("train", loss=float(loss.detach()), parts={k: float(v.detach()) for k, v in parts.items()},
                    grad_norm=float(norm), lr_head=hm * o["head_lr"], lr_enc=em * o["encoder_lr"],
                    tiers=torch.bincount(batch["tier"], minlength=5).tolist(), elapsed=time.time() - t_start)

            if step == t["first_monitor"] or step % t["monitor_every"] == 0:
                if main_proc:
                    took = run_monitor()
                    if bench["t0"] is not None and not bench["done"]:
                        bench["excluded"] += took
                barrier()
            if main_proc and time.time() - last_save >= t["save_minutes"] * 60:
                save_latest()
                last_save = time.time()
                if bench["t0"] is not None and not bench["done"]:
                    bench["excluded"] += planner.save_s
        # ---- clean finish
        if main_proc:
            if last_monitor != step and step > 0 and time.time() + 1.2 * planner.monitor_s + planner.save_s < deadline:
                run_monitor()
            save_latest()
            log("finished", reason=reason)
        barrier()
    except BaseException as exc:
        if main_proc:
            try:
                save_latest()
            except Exception as save_exc:  # noqa: BLE001 - report both failures
                print(f"[error] could not save latest.pt after failure: {save_exc}", flush=True)
            log("interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sg, h in old_handlers.items():
            signal.signal(sg, h)
        if ddp_active():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
