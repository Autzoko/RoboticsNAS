"""Supernet (sandwich + in-place distillation) or fixed-architecture training of elastic SmolVLA on LIBERO.

Single GPU: python -m rnas.train ...
Multi GPU : torchrun --nproc_per_node=N -m rnas.train ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .build import build_elastic, load_trainable, make_processors, setup_env_vars, trainable_state_dict
from .data import make_dataset
from .space import DEFAULT, SMALLEST_NET, Arch, sample_net

ACT_DIM = 7


def masked_mse(a, b, is_pad):
    err = (a[:, :, :ACT_DIM] - b[:, :, :ACT_DIM]) ** 2
    valid = (~is_pad).unsqueeze(-1).to(err.dtype)
    return (err * valid).sum() / (valid.sum() * ACT_DIM).clamp_min(1)


def flow_losses(model, batch, archs: list[Arch], kd_weight: float, noise=None, t=None):
    """Shared prefix (per vtok) + shared noise/time; first arch is the distillation teacher if kd_weight>0."""
    images, img_masks, lt, lm, state = model.prepare(batch)
    actions = model.policy.prepare_action(batch)
    is_pad = batch["action_is_pad"]
    if noise is None:
        noise = model.m.sample_noise(actions.shape, actions.device)
    if t is None:
        t = model.m.sample_time(actions.shape[0], actions.device)
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    u_t = noise - actions
    prefixes = {}
    with model.image_cache():
        for vt in sorted({a.vtok for a in archs}):
            n = max(a.vlm_layers_needed() for a in archs if a.vtok == vt)
            prefixes[vt] = model.prefix_kv(images, img_masks, lt, lm, state, vt, n)
    out, teacher = [], None
    for i, a in enumerate(archs):
        kvs, pad = prefixes[a.vtok]
        v = model.velocity(kvs, pad, x_t, t, a)
        fm = masked_mse(v.float(), u_t, is_pad)
        kd = None
        if i == 0:
            teacher = v.detach()
        elif kd_weight > 0:
            kd = masked_mse(v.float(), teacher.float(), is_pad)
        out.append((a, fm, kd))
    return out


def lr_at(step, peak, warmup, total, final):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["supernet", "fixed"], default="supernet")
    ap.add_argument("--arch", default=DEFAULT.key(), help="fixed mode: architecture key")
    ap.add_argument("--init", default=None, help="init trainable weights from checkpoint (e.g. supernet)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64, help="global batch size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-final", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="configs/split.json")
    ap.add_argument("--stats", default="configs/train_stats.json")
    args = ap.parse_args()
    setup_env_vars()

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.manual_seed(args.seed + rank)
    rng = random.Random(args.seed)  # identical arch sampling on every rank
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    model = build_elastic("cuda")
    pre, _ = make_processors(model, args.stats)
    if args.init:
        load_trainable(model, args.init)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    step = 0
    if (out / "last.pt").exists():
        ck = load_trainable(model, str(out / "last.pt"))
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        rng.setstate(ck["rng"])
        print(f"[rank{rank}] resumed at step {step}")

    split = json.loads(Path(args.split).read_text())
    ds = make_dataset(split["train"], chunk_size=model.chunk)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, seed=args.seed) if ddp else None
    dl = DataLoader(
        ds, batch_size=args.batch // world, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True,
    )
    val_ds = make_dataset(split["val"], chunk_size=model.chunk)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4,
                        generator=torch.Generator().manual_seed(123))
    val_batches = []
    for i, b in enumerate(val_dl):
        if i == 8:
            break
        val_batches.append(b)

    fixed = Arch.from_key(args.arch)
    log_f = open(out / "train_log.jsonl", "a") if rank == 0 else None
    t0, epoch = time.time(), 0
    model.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            if step >= args.steps:
                break
            batch = pre(batch)
            if args.mode == "supernet":
                archs = [DEFAULT, SMALLEST_NET] + [sample_net(rng) for _ in range(args.n_random)]
            else:
                archs = [fixed]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps, args.lr_final)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = flow_losses(model, batch, archs, args.kd_weight if args.mode == "supernet" else 0.0)
            loss = sum(fm + (args.kd_weight * kd if kd is not None else 0.0) for _, fm, kd in res)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ddp:  # data-parallel by hand: identical init on all ranks, one flat all-reduce of grads
                gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                flat = torch.cat([g.reshape(-1) for g in gs])
                dist.all_reduce(flat)
                flat /= world
                off = 0
                for p, g in zip(params, gs):
                    p.grad = flat[off : off + g.numel()].view_as(g)
                    off += g.numel()
            gn = torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            step += 1
            if rank == 0 and step % args.log_every == 0:
                rec = {"step": step, "lr": opt.param_groups[0]["lr"], "gn": float(gn),
                       "sps": args.log_every / (time.time() - t0)}
                for i, (a, fm, kd) in enumerate(res):
                    tag = ["anchor", "smallest", "rand", "rand"][min(i, 3)] if args.mode == "supernet" else "fixed"
                    rec[f"fm_{tag}{i}"] = float(fm)
                    if kd is not None:
                        rec[f"kd_{tag}{i}"] = float(kd)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(rec, flush=True)
                t0 = time.time()
            if rank == 0 and step % args.val_every == 0:
                model.eval()
                g = torch.Generator(device="cuda").manual_seed(0)
                vals = {}
                eval_archs = [DEFAULT, SMALLEST_NET] if args.mode == "supernet" else [fixed]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sums = [0.0] * len(eval_archs)
                    for vb in val_batches:
                        vb = pre(dict(vb))
                        act = model.policy.prepare_action(vb)
                        nz = torch.randn(act.shape, device="cuda", generator=g)
                        tt = torch.rand(act.shape[0], device="cuda", generator=g) * 0.999 + 0.001
                        r = flow_losses(model, vb, eval_archs, 0.0, noise=nz, t=tt)
                        for i, (_, fm, _) in enumerate(r):
                            sums[i] += float(fm) / len(val_batches)
                for a, s in zip(eval_archs, sums):
                    vals[f"val_fm_{a.key()}"] = s
                log_f.write(json.dumps({"step": step, **vals}) + "\n")
                log_f.flush()
                print(vals, flush=True)
                model.train()
            if rank == 0 and (step % args.save_every == 0 or step == args.steps):
                ck = {"model": trainable_state_dict(model), "opt": opt.state_dict(), "step": step,
                      "rng": rng.getstate(), "args": vars(args)}
                torch.save(ck, out / "last.tmp")
                os.replace(out / "last.tmp", out / "last.pt")
                if step % 10000 == 0:  # keep sparse snapshots (offline-loss vs closed-loop over training)
                    torch.save({"model": ck["model"], "step": step}, out / f"step{step // 1000}k.pt")
        epoch += 1
    if rank == 0:
        torch.save({"model": trainable_state_dict(model), "step": step, "args": vars(args)}, out / "final.pt")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
