"""Closed-loop LIBERO evaluation of many sub-networks of one (super)net checkpoint.

Modes
- search: `init_states=False`; layouts sampled by LIBERO's placement samplers from reset seeds
          SEARCH_SEED_BASE + 1000 * global_task + episode. Official init-state files are never loaded.
- test:   official LIBERO init states (episode i -> init state i). Only for frozen final candidates.

All archs in a job are evaluated on the same env workers with the same seeds (common random numbers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .build import build_elastic, load_trainable, make_processors, setup_env_vars
from .space import Arch

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEARCH_SEED_BASE = 1_000_000
TEST_SEED = 7


def _make_env(suite_name: str, task_id: int, episode_index: int, init_states: bool):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    class RnasLiberoEnv(LiberoEnv):
        def sim_hash(self) -> str:
            st = self._env.sim.get_state().flatten()
            return hashlib.md5(np.round(st, 5).tobytes()).hexdigest()[:12]

    return RnasLiberoEnv(
        task_suite=_get_suite(suite_name), task_id=task_id, task_suite_name=suite_name,
        camera_name="agentview_image,robot0_eye_in_hand_image", obs_type="pixels_agent_pos",
        observation_width=256, observation_height=256, init_states=init_states,
        episode_index=episode_index, n_envs=1, control_mode="relative", hard_reset=True,
    )


def make_venv(suite: str, task_id: int, episodes: list[int], mode: str):
    import gymnasium as gym

    from lerobot.envs.utils import freeze_after_episode_end

    init_states = mode == "test"
    fns = [freeze_after_episode_end(partial(_make_env, suite, task_id, e, init_states)) for e in episodes]
    return gym.vector.AsyncVectorEnv(fns, context="forkserver", shared_memory=True,
                                     autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)


@torch.no_grad()
def run_episodes(venv, model, arch: Arch, pre, post, env_pre, seeds, max_steps, task_desc, record=None):
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation

    n = venv.num_envs
    obs, _ = venv.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
    hashes = list(venv.call("sim_hash"))
    done = np.zeros(n, bool)
    succ = np.zeros(n, bool)
    length = np.full(n, max_steps)
    queue, qi = None, 0
    n_calls, t_policy = 0, 0.0
    for step in range(max_steps):
        if queue is None or qi >= arch.horizon:
            o = preprocess_observation(obs)
            o["task"] = [task_desc] * n
            o = env_pre(o)
            if record is not None:
                record(step, o, done)
            b = pre(o)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = model.sample_actions(b, arch)
            torch.cuda.synchronize()
            t_policy += time.perf_counter() - t0
            n_calls += 1
            queue = post(chunk.float()).cpu().numpy()  # [n, T, 7] unnormalized
            qi = 0
        act = queue[:, qi]
        qi += 1
        obs, _, term, trunc, info = venv.step(act)
        new = (term | trunc) & ~done
        if new.any():
            s = np.asarray(info.get("is_success", np.zeros(n, bool)), bool)
            succ[new] = s[new]
            length[new] = step + 1
            done |= new
        if done.all():
            break
    return {"success": succ.tolist(), "length": length.tolist(), "init_hash": hashes,
            "policy_calls": n_calls, "policy_s": t_policy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--archs", required=True, help="json list of arch keys, or comma-separated keys")
    ap.add_argument("--mode", choices=["search", "test"], default="search")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--tasks", default=None, help="restrict task ids, e.g. 0,1,2")
    ap.add_argument("--n-eps", type=int, default=10)
    ap.add_argument("--ep-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-dir", default=None, help="save visited observations (first arch only)")
    ap.add_argument("--record-every", type=int, default=10)
    ap.add_argument("--stats", default="configs/train_stats.json")
    args = ap.parse_args()
    setup_env_vars()
    if args.mode == "test":
        assert os.environ.get("RNAS_ALLOW_TEST") == "1", "test mode requires RNAS_ALLOW_TEST=1 (final candidates only)"

    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep

    if args.archs.endswith(".json"):
        keys = json.loads(Path(args.archs).read_text())
    else:
        keys = args.archs.split(",")
    archs = [Arch.from_key(k) for k in keys]
    model = build_elastic("cuda", sort_neurons=False).eval()
    load_trainable(model, args.ckpt)
    pre, post = make_processors(model, args.stats)
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["arch"], r["suite"], r["task"]))
    f = open(out, "a")
    suites = args.suites.split(",")
    for si, suite in enumerate(suites):
        n_tasks = len(_get_suite(suite).tasks)
        task_ids = [int(x) for x in args.tasks.split(",")] if args.tasks else range(n_tasks)
        for tid in task_ids:
            todo = [a for a in archs if (a.key(), suite, tid) not in done_keys]
            if not todo:
                continue
            gtask = SUITES.index(suite) * 10 + tid
            eps = list(range(args.ep_offset, args.ep_offset + args.n_eps))
            seeds = ([SEARCH_SEED_BASE + 1000 * gtask + e for e in eps] if args.mode == "search"
                     else [TEST_SEED] * len(eps))
            t0 = time.time()
            venv = make_venv(suite, tid, eps, args.mode)
            desc = venv.call("task_description")[0]
            t_env = time.time() - t0
            for ai, a in enumerate(todo):
                rec = None
                if args.record_dir and ai == 0:
                    rec = Recorder(Path(args.record_dir) / f"{suite}_t{tid}.npz", args.record_every)
                t1 = time.time()
                r = run_episodes(venv, model, a, pre, post, env_pre, seeds,
                                 TASK_SUITE_MAX_STEPS[suite], desc, record=rec)
                if rec is not None:
                    rec.save()
                row = {"arch": a.key(), "suite": suite, "task": tid, "mode": args.mode, "episodes": eps,
                       "seeds": seeds, "desc": desc, "wall_s": time.time() - t1, "env_init_s": t_env, **r}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{suite} t{tid} {a.key()} SR={np.mean(r['success']):.2f} "
                      f"wall={row['wall_s']:.0f}s calls={r['policy_calls']}", flush=True)
            venv.close()


class Recorder:
    """Stores policy inputs (post env-processing, pre-normalization) at visited states."""

    def __init__(self, path: Path, every: int):
        self.path, self.every = path, every
        self.items = {"img": [], "img2": [], "state": [], "step": [], "env": []}
        self.desc = None

    def __call__(self, step, o, done):
        if step % self.every:
            return
        for i in np.where(~done)[0]:
            self.items["img"].append((o["observation.images.image"][i] * 255).round().byte().numpy())
            self.items["img2"].append((o["observation.images.image2"][i] * 255).round().byte().numpy())
            self.items["state"].append(o["observation.state"][i].numpy())
            self.items["step"].append(step)
            self.items["env"].append(int(i))
        self.desc = o["task"][0]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, desc=self.desc, **{k: np.stack(v) for k, v in self.items.items()})


if __name__ == "__main__":
    main()
