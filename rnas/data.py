"""LIBERO data split (train / offline-val), train-only normalization stats, dataset construction."""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("RNAS_LIBERO_ROOT", "/scratch/ll5582/LIBERO/lerobot_libero"))
REPO_ID = "lerobot/libero"
N_VAL_PER_TASK = 3
SPLIT_SEED = 0


def _frames(cols: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_ROOT / "data" / "*" / "*.parquet")))
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)


def build_split(out_path: str) -> dict:
    """Per task, hold out N_VAL_PER_TASK demos (fixed seed) as offline-val; the rest is train."""
    df = _frames(["episode_index", "task_index"])
    ep_task = df.groupby("episode_index")["task_index"].first()
    tasks = pq.read_table(DATA_ROOT / "meta" / "tasks.parquet").to_pandas()
    # tasks.parquet: index = task string, column task_index
    idx2task = {int(r.task_index): str(t) for t, r in tasks.iterrows()}
    rng = random.Random(SPLIT_SEED)
    train, val = [], []
    for t in sorted(ep_task.unique()):
        eps = sorted(int(e) for e in ep_task[ep_task == t].index)
        rng.shuffle(eps)
        val += eps[:N_VAL_PER_TASK]
        train += eps[N_VAL_PER_TASK:]
    split = {
        "repo_id": REPO_ID,
        "data_root": str(DATA_ROOT),
        "seed": SPLIT_SEED,
        "n_val_per_task": N_VAL_PER_TASK,
        "train": sorted(train),
        "val": sorted(val),
        "episode_task": {int(e): int(t) for e, t in ep_task.items()},
        "tasks": idx2task,
    }
    assert not set(train) & set(val)
    with open(out_path, "w") as f:
        json.dump(split, f)
    return split


def build_stats(split: dict, out_path: str) -> dict:
    """mean/std/min/max/q01/q99 of observation.state and action over TRAIN episodes only."""
    df = _frames(["episode_index", "observation.state", "action"])
    df = df[df["episode_index"].isin(set(split["train"]))]
    stats = {}
    for key in ("observation.state", "action"):
        x = np.stack(df[key].to_numpy()).astype(np.float64)
        stats[key] = {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "min": x.min(0).tolist(),
            "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
            "count": [int(x.shape[0])],
        }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def load_stats_tensors(path: str) -> dict:
    import torch

    with open(path) as f:
        stats = json.load(f)
    return {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}


def make_dataset(episodes: list[int], chunk_size: int = 50, video_backend: str = "pyav"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(DATA_ROOT / "meta" / "info.json") as f:
        fps = json.load(f)["fps"]
    return LeRobotDataset(
        REPO_ID,
        root=DATA_ROOT,
        episodes=episodes,
        delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
        video_backend=video_backend,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="configs")
    a = ap.parse_args()
    sp = build_split(f"{a.out_dir}/split.json")
    build_stats(sp, f"{a.out_dir}/train_stats.json")
    print(f"train={len(sp['train'])} val={len(sp['val'])} tasks={len(sp['tasks'])}")
