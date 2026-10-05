"""Cheap architecture-ranking signals (no simulation), computed with the shared supernet weights.

Offline (expert / off-policy states, offline-val demos):
  fm_loss      flow-matching loss with fixed (noise, t) draws               (standard NAS val loss)
  act_l1       L1 of the denoised chunk (arch.steps Euler steps) vs GT, all chunk steps
  act_l1_exec  same, restricted to the executed prefix [0, arch.horizon)
  grip_err     gripper-sign error rate of executed prefix around gripper transitions
  kd_offline   L1 between arch and anchor denoised executed actions on offline-val states
On-policy (states visited by the anchor policy in search-val rollouts, recorded by rollout.py):
  kd_onpolicy  L1 between arch and anchor executed actions on anchor-visited states
  self_cons    L1 between two denoising draws of the arch (sampling stochasticity) on visited states
Cost proxies: params of the active expert, prefix/expert FLOPs estimate, measured latency (latency.py).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from .build import build_elastic, load_trainable, make_processors, setup_env_vars
from .data import make_dataset
from .space import DEFAULT, Arch
from .train import flow_losses

ACT = 7


def offline_batches(split_path, n_batches, bs, chunk):
    split = json.loads(Path(split_path).read_text())
    ds = make_dataset(split["val"], chunk_size=chunk)
    g = torch.Generator().manual_seed(2024)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=8)
    out = []
    for i, b in enumerate(dl):
        if i == n_batches:
            break
        out.append(b)
    return out


def visited_batches(record_dir, per_task, bs):
    """Anchor-visited states, subsampled per task, grouped into batches of a single task."""
    rng = np.random.default_rng(0)
    batches = []
    for fpath in sorted(glob.glob(f"{record_dir}/*.npz")):
        z = np.load(fpath, allow_pickle=True)
        idx = rng.choice(len(z["state"]), size=min(per_task, len(z["state"])), replace=False)
        for s in range(0, len(idx), bs):
            j = idx[s : s + bs]
            batches.append({
                "observation.images.image": torch.from_numpy(z["img"][j]).float() / 255,
                "observation.images.image2": torch.from_numpy(z["img2"][j]).float() / 255,
                "observation.state": torch.from_numpy(z["state"][j]).float(),
                "task": [str(z["desc"])] * len(j),
            })
    return batches


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--archs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--visited-per-task", type=int, default=32)
    ap.add_argument("--split", default="configs/split.json")
    ap.add_argument("--stats", default="configs/train_stats.json")
    args = ap.parse_args()
    setup_env_vars()
    keys = json.loads(Path(args.archs).read_text()) if args.archs.endswith(".json") else args.archs.split(",")
    archs = [Arch.from_key(k) for k in keys]
    model = build_elastic("cuda", sort_neurons=False).eval()
    load_trainable(model, args.ckpt)
    pre, post = make_processors(model, args.stats)

    off = [pre(b) for b in offline_batches(args.split, args.n_batches, args.bs, model.chunk)]
    vis = [pre(b) for b in visited_batches(args.record_dir, args.visited_per_task, args.bs)] if args.record_dir else []
    gen = torch.Generator(device="cuda")
    st = json.loads(Path(args.stats).read_text())["action"]
    g_thr = -st["mean"][6] / st["std"][6]  # normalized value of raw gripper command 0

    def fixed_noise(b, k):
        gen.manual_seed(1000 + k)
        n = b["observation.state"].shape[0]
        return torch.randn(n, model.chunk, 32, device="cuda", generator=gen)

    def denoise(b, a, k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.sample_actions(b, a, noise=fixed_noise(b, k)).float()

    # anchor references (shared by all archs)
    anchor_off = [denoise(b, DEFAULT, i) for i, b in enumerate(off)]
    anchor_vis = [denoise(b, DEFAULT, 10_000 + i) for i, b in enumerate(vis)]

    rows = []
    for a in archs:
        r = {"arch": a.key()}
        fm = []
        for i, b in enumerate(off):
            act = model.policy.prepare_action(b)
            gen.manual_seed(i)
            nz = torch.randn(act.shape, device="cuda", generator=gen)
            tt = torch.rand(act.shape[0], device="cuda", generator=gen) * 0.999 + 0.001
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm.append(float(flow_losses(model, b, [a], 0.0, noise=nz, t=tt)[0][1]))
        r["fm_loss"] = float(np.mean(fm))
        l1, l1e, kd, ge = [], [], [], []
        for i, b in enumerate(off):
            pred = denoise(b, a, i)
            gt = b["action"][:, :, :ACT].float()
            valid = (~b["action_is_pad"]).float()[..., None]
            h = a.horizon
            l1.append(float(((pred - gt).abs() * valid).sum() / (valid.sum() * ACT)))
            l1e.append(float(((pred[:, :h] - gt[:, :h]).abs() * valid[:, :h]).sum() / (valid[:, :h].sum() * ACT)))
            kd.append(float((pred[:, :h] - anchor_off[i][:, :h]).abs().mean()))
            # gripper open/close disagreement on executed chunks where the GT gripper command flips
            g_gt, g_pr = gt[:, :h, 6] > g_thr, pred[:, :h, 6] > g_thr
            flip = (g_gt != g_gt[:, :1]).any(1)
            if flip.any():
                ge.append(float((g_gt[flip] != g_pr[flip]).float().mean()))
        r.update(act_l1=float(np.mean(l1)), act_l1_exec=float(np.mean(l1e)), kd_offline=float(np.mean(kd)),
                 grip_err=float(np.mean(ge)) if ge else float("nan"))
        if vis:
            kdv, sc = [], []
            for i, b in enumerate(vis):
                h = a.horizon
                p1 = denoise(b, a, 10_000 + i)
                p2 = denoise(b, a, 20_000 + i)
                kdv.append(float((p1[:, :h] - anchor_vis[i][:, :h]).abs().mean()))
                sc.append(float((p1[:, :h] - p2[:, :h]).abs().mean()))
            r.update(kd_onpolicy=float(np.mean(kdv)), self_cons=float(np.mean(sc)))
        rows.append(r)
        print(r, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
