"""Track 1 SED trainer: WavLM + weighted layer sum + BiGRU, strong + weak supervision.

Reads the staged float16 blob written by stage_data.py. Fully resumable:
checkpoints (model + optimizer + scaler + step + RNG) land every --ckpt-every
steps and the newest one is reloaded automatically on restart.

Losses
  frame BCE (tier-weighted)        - gold 1.0, silver `silver_w`
  soft Tversky on the noise track  - directly shapes segment Dice
  onset/offset heatmap BCE         - sharpens boundaries, which is what the
                                     tolerance-based event F1 actually rewards
  category BCE (7 aux tracks)
  clip-level MIL BCE for bronze    - attention pooling over frames
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/content/ai2")
DATA = ROOT / "data"
CATEGORIES = ("animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech")
CIDX = {c: i for i, c in enumerate(CATEGORIES)}
N_TAG = 64


def tag_vocab():
    """Top-N_TAG noise tags, cached. Adjacent natural events always carry
    different tags, so a per-tag track lets the decoder split blobs that a
    single binary track would merge into one event."""
    path = DATA / "tags.json"
    if path.exists():
        return json.loads(path.read_text())
    from collections import Counter
    cnt = Counter()
    with (DATA / "manifest.jsonl").open() as fh:
        for line in fh:
            try:
                it = json.loads(line)
            except Exception:
                break
            for sp in it["s"]:
                if len(sp) > 3 and sp[3]:
                    cnt[sp[3]] += 1
    vocab = {t: i for i, (t, _) in enumerate(cnt.most_common(N_TAG))}
    path.write_text(json.dumps(vocab, ensure_ascii=False))
    return vocab
FRAME = 0.02
SR = 16000


# ------------------------------------------------------------------ data
class Blob:
    def __init__(self):
        self.mm = np.memmap(DATA / "audio.f16", dtype=np.float16, mode="r")
        self.items = []
        with (DATA / "manifest.jsonl").open() as fh:
            for line in fh:
                try:
                    self.items.append(json.loads(line))
                except Exception:
                    break
        print(f"manifest {len(self.items)} clips, blob {len(self.mm)/SR/3600:.1f} h", flush=True)

    def wave(self, it):
        return np.asarray(self.mm[it["o"]: it["o"] + it["n"]], dtype=np.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, blob, tags, seconds=6.0, train=True):
        self.items, self.blob, self.train, self.tags = items, blob, train, tags
        self.n_out = 1 + len(CATEGORIES) + N_TAG
        self.L = int(seconds * SR)
        self.T = int(seconds / FRAME)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav = self.blob.wave(it)
        spans = it["s"]
        if len(wav) > self.L:
            if self.train:
                # bias crops toward windows that contain an event
                if spans and random.random() < 0.8:
                    sp = random.choice(spans)
                    s, e = sp[0], sp[1]
                    centre = int((s + e) / 2 * SR)
                    start = centre - random.randint(0, self.L)
                    start = max(0, min(len(wav) - self.L, start))
                else:
                    start = random.randint(0, len(wav) - self.L)
            else:
                start = 0
        else:
            start = 0
        chunk = wav[start: start + self.L]
        pad = self.L - len(chunk)
        if pad > 0:
            chunk = np.pad(chunk, (0, pad))
        valid = min(self.T, int(math.ceil((len(wav) - start) / SR / FRAME)))

        y = np.zeros((self.n_out, self.T), np.float32)
        on = np.zeros(self.T, np.float32)
        off = np.zeros(self.T, np.float32)
        t0 = start / SR
        for sp in spans:
            s, e, cat = sp[0], sp[1], sp[2]
            tag = sp[3] if len(sp) > 3 else ""
            a, b = s - t0, e - t0
            if b <= 0 or a >= self.T * FRAME:
                continue
            lo = max(0, int(math.floor(a / FRAME)))
            hi = min(self.T, int(math.ceil(b / FRAME)))
            if hi <= lo:
                continue
            y[0, lo:hi] = 1.0
            ci = CIDX.get(cat)
            if ci is not None:
                y[1 + ci, lo:hi] = 1.0
            ti = self.tags.get(tag)
            if ti is not None:
                y[1 + len(CATEGORIES) + ti, lo:hi] = 1.0
            if 0 <= lo < self.T:
                on[max(0, lo - 1): lo + 2] = np.maximum(on[max(0, lo - 1): lo + 2], [0.5, 1.0, 0.5][: len(on[max(0, lo - 1): lo + 2])])
            if 0 <= hi - 1 < self.T:
                j = hi - 1
                off[max(0, j - 1): j + 2] = np.maximum(off[max(0, j - 1): j + 2], [0.5, 1.0, 0.5][: len(off[max(0, j - 1): j + 2])])

        clip = np.zeros(self.n_out, np.float32)
        if it["t"] == "bronze":
            for c in it.get("c") or []:
                ci = CIDX.get(c)
                if ci is not None:
                    clip[1 + ci] = 1.0
            clip[0] = 1.0 if (it.get("c") or []) else 0.0
        else:
            clip = y.max(axis=1)

        mask = np.zeros(self.T, np.float32)
        mask[:valid] = 1.0
        tier = {"gold": 0, "silver": 1, "bronze": 2}[it["t"]]
        return (torch.from_numpy(chunk), torch.from_numpy(y), torch.from_numpy(on),
                torch.from_numpy(off), torch.from_numpy(clip), torch.from_numpy(mask),
                torch.tensor(tier))


# ----------------------------------------------------------------- model
class SED(nn.Module):
    def __init__(self, backbone="microsoft/wavlm-base-plus", hidden=256, n_out=72):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(backbone)
        self.encoder.feature_extractor._freeze_parameters()
        self.encoder.config.output_hidden_states = True
        nl = self.encoder.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(nl))
        d = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 384), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(384, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.15)
        h = 2 * hidden
        self.norm = nn.LayerNorm(h)
        self.frame = nn.Linear(h, n_out)
        self.bound = nn.Linear(h, 2)
        self.att = nn.Linear(h, n_out)

    def forward(self, wav, spec_mask=False):
        out = self.encoder(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)                  # L,B,T,D
        # transformers versions differ on whether the embedding output is
        # included, so trim the learnable weights to the layers we actually got
        w = torch.softmax(self.layer_w[: hs.shape[0]], 0).view(-1, 1, 1, 1)
        x = (hs * w).sum(0)
        if spec_mask and self.training:
            B, T, D = x.shape
            for _ in range(2):
                t = random.randint(0, max(1, T // 12))
                s = random.randint(0, max(0, T - t))
                x[:, s:s + t, :] = 0
            c = random.randint(0, max(1, D // 12))
            s = random.randint(0, max(0, D - c))
            x[:, :, s:s + c] = 0
        x = self.proj(x)
        x, _ = self.gru(x)
        x = self.norm(x)
        return self.frame(x).transpose(1, 2), self.bound(x).transpose(1, 2), self.att(x).transpose(1, 2)


def align(t, T):
    return F.interpolate(t, size=T, mode="linear", align_corners=False)


def tversky(prob, target, mask, alpha=0.5, beta=0.5, eps=1.0):
    p, t, m = prob * mask, target * mask, mask
    tp = (p * t).sum(-1)
    fp = (p * (1 - t) * m).sum(-1)
    fn = ((1 - p) * t).sum(-1)
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--enc-lr", type=float, default=2e-5)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--silver-w", type=float, default=0.55)
    ap.add_argument("--bronze-w", type=float, default=0.25)
    ap.add_argument("--tversky", type=float, default=0.6)
    ap.add_argument("--bound-w", type=float, default=1.0)
    ap.add_argument("--cat-w", type=float, default=0.3)
    ap.add_argument("--mixup", type=float, default=0.35)
    ap.add_argument("--pos-weight", type=float, default=1.5)
    ap.add_argument("--ckpt-every", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run", default="run1")
    ap.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", default="gold,silver,bronze",
                    help="comma list of tiers to train on; 'gold' alone gives a "
                         "boundary-sharpening fine-tune on verified timestamps")
    ap.add_argument("--init", default=None,
                    help="weights_*.pt to start from (model only, fresh optimiser)")
    args = ap.parse_args()

    outdir = ROOT / "runs" / args.run
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2))
    logf = (outdir / "train.log").open("a")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    blob = Blob()
    keep = set(args.tiers.split(","))
    items = [it for it in blob.items if it["t"] in keep]
    random.Random(7).shuffle(items)
    log(f"training on tiers {sorted(keep)}: {len(items)} clips")
    tags = tag_vocab()
    log(f"tag vocab {len(tags)}")
    ds = Crops(items, blob, tags, args.seconds, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True,
                                     pin_memory=True, persistent_workers=args.workers > 0,
                                     prefetch_factor=4 if args.workers else None)

    model = SED(args.backbone, n_out=1 + len(CATEGORIES) + N_TAG).to(dev)
    decay, no_decay, enc = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder."):
            enc.append(p)
        elif p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([
        {"params": enc, "lr": args.enc_lr, "weight_decay": 0.01},
        {"params": decay, "lr": args.lr, "weight_decay": 0.01},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
    ])
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, args.steps - warm)))) * 0.95 + 0.05))
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    if args.init and not (outdir / "last.pt").exists():
        ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
        sd0 = {k: (v.float() if v.is_floating_point() else v) for k, v in ck0["model"].items()}
        model.load_state_dict(sd0, strict=True)
        log(f"initialised from {args.init} (step {ck0.get('step')})")
    last = outdir / "last.pt"
    if last.exists():
        try:
            ck = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            step = ck["step"]
            log(f"resumed from step {step}")
        except Exception as err:
            log(f"resume failed ({err}); starting fresh")

    tier_w = torch.tensor([1.0, args.silver_w, args.bronze_w], device=dev)
    posw = torch.tensor(args.pos_weight, device=dev)
    t0 = time.time()
    running = []
    model.train()
    while step < args.steps:
        for wav, y, on, off, clip, mask, tier in dl:
            if step >= args.steps:
                break
            wav = wav.to(dev, non_blocking=True); y = y.to(dev); on = on.to(dev)
            off = off.to(dev); clip = clip.to(dev); mask = mask.to(dev); tier = tier.to(dev)

            if args.mixup > 0 and random.random() < args.mixup:
                perm = torch.randperm(wav.size(0), device=dev)
                lam = float(np.random.beta(2.0, 2.0))
                wav = lam * wav + (1 - lam) * wav[perm]
                y = torch.maximum(y, y[perm]); on = torch.maximum(on, on[perm])
                off = torch.maximum(off, off[perm]); clip = torch.maximum(clip, clip[perm])
                mask = torch.maximum(mask, mask[perm])
                tier = torch.maximum(tier, tier[perm])
            if random.random() < 0.5:
                wav = wav * (10 ** (torch.empty(wav.size(0), 1, device=dev).uniform_(-6, 6) / 20))

            with torch.autocast("cuda", dtype=torch.bfloat16):
                fr, bd, at = model(wav, spec_mask=True)
                T = y.shape[-1]
                fr, bd, at = align(fr, T), align(bd, T), align(at, T)
                w = tier_w[tier].view(-1, 1, 1)
                m = mask.unsqueeze(1)
                strong = (tier < 2).float().view(-1, 1, 1)

                bce = F.binary_cross_entropy_with_logits(fr, y, reduction="none", pos_weight=posw)
                cw = torch.ones_like(bce); cw[:, 1:, :] = args.cat_w
                l_frame = (bce * m * w * strong * cw).sum() / (m * w * strong * cw).sum().clamp(min=1)

                p0 = torch.sigmoid(fr[:, 0]) * mask
                l_tv = (tversky(p0, y[:, 0], mask, 0.5, 0.5) *
                        (w[:, 0, 0] * strong[:, 0, 0])).sum() / (w[:, 0, 0] * strong[:, 0, 0]).sum().clamp(min=1)

                bt = torch.stack([on, off], 1)
                bb = F.binary_cross_entropy_with_logits(bd, bt, reduction="none")
                l_bd = (bb * m * w * strong).sum() / (m * w * strong).sum().clamp(min=1)

                aw = torch.softmax(at.masked_fill(m < 0.5, -1e4), dim=-1)
                pooled = (torch.sigmoid(fr.float()) * aw.float()).sum(-1).clamp(1e-6, 1 - 1e-6)
                mil = -(clip * torch.log(pooled) + (1 - clip) * torch.log1p(-pooled))
                l_mil = (mil.mean(1) * tier_w[tier]).mean()

                loss = (l_frame + args.tversky * l_tv + args.bound_w * l_bd +
                        0.5 * l_mil) / args.accum

            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            running.append(float(loss) * args.accum)

            if step % 50 == 0:
                log(f"step {step}/{args.steps} loss {np.mean(running[-50:]):.4f} "
                    f"frame {float(l_frame):.4f} tv {float(l_tv):.4f} bd {float(l_bd):.4f} "
                    f"mil {float(l_mil):.4f} lr {sched.get_last_lr()[1]:.2e} "
                    f"{(time.time()-t0)/60:.1f}min")
            if step % args.ckpt_every == 0 or step == args.steps:
                payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                           "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                           "step": step, "args": vars(args), "backbone": args.backbone}
                tmp = outdir / "last.tmp"
                torch.save(payload, tmp); os.replace(tmp, last)
                half = {k: (v.half() if v.is_floating_point() else v)
                        for k, v in model.state_dict().items()}
                tmp2 = outdir / "w.tmp"
                torch.save({"model": half, "step": step, "backbone": args.backbone,
                            "args": vars(args), "tags": tags}, tmp2)
                os.replace(tmp2, outdir / f"weights_{step}.pt")
                log(f"checkpoint {step}")
    log("training complete")


if __name__ == "__main__":
    main()
