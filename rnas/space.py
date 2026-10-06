"""Search space for elastic SmolVLA (architecture + inference-schedule axes)."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass

N_VLM = (8, 12, 16, 20, 24)
N_EXP = (4, 8, 12, 16)
BRIDGE = ("stretch", "top")
FFN = (0.5, 0.75, 1.0)
VTOK = (16, 64)
STEPS = (2, 4, 10)
HORIZON = (5, 10, 25, 50)

MAX_VLM = max(N_VLM)
MAX_EXP = max(N_EXP)


@dataclass(frozen=True)
class Arch:
    n_vlm: int = 16
    n_exp: int = 16
    bridge: str = "stretch"
    ffn: float = 1.0
    vtok: int = 64
    steps: int = 10
    horizon: int = 50

    # ---- derived ----
    def bridge_map(self) -> list[int]:
        """VLM layer index read by expert layer j (j = 0..n_exp-1)."""
        if self.bridge == "stretch":
            return [((j + 1) * self.n_vlm) // self.n_exp - 1 for j in range(self.n_exp)]
        if self.bridge == "top":
            assert self.n_exp <= self.n_vlm
            return [self.n_vlm - self.n_exp + j for j in range(self.n_exp)]
        raise ValueError(self.bridge)

    def vlm_layers_needed(self) -> int:
        return max(self.bridge_map()) + 1

    def is_valid(self) -> bool:
        if self.bridge == "top" and self.n_exp > self.n_vlm:
            return False
        # `top` == `stretch` when n_exp == n_vlm; keep only the stretch spelling.
        if self.bridge == "top" and self.n_exp == self.n_vlm:
            return False
        return True

    @property
    def net(self) -> tuple:
        """Architecture part only (weights-relevant); inference knobs excluded."""
        return (self.n_vlm, self.n_exp, self.bridge, self.ffn, self.vtok)

    def key(self) -> str:
        return (
            f"v{self.n_vlm}-e{self.n_exp}-{self.bridge}-f{self.ffn:g}-t{self.vtok}"
            f"-s{self.steps}-h{self.horizon}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_key(k: str) -> Arch:
        v, e, b, f, t, s, h = k.split("-")
        return Arch(int(v[1:]), int(e[1:]), b, float(f[1:]), int(t[1:]), int(s[1:]), int(h[1:]))


DEFAULT = Arch()
SMALLEST_NET = Arch(n_vlm=8, n_exp=4, bridge="stretch", ffn=0.5, vtok=16)


def all_archs(include_inference: bool = True) -> list[Arch]:
    out = []
    steps = STEPS if include_inference else (10,)
    hor = HORIZON if include_inference else (50,)
    for v, e, b, f, t, s, h in itertools.product(N_VLM, N_EXP, BRIDGE, FFN, VTOK, steps, hor):
        a = Arch(v, e, b, f, t, s, h)
        if a.is_valid():
            out.append(a)
    return out


def sample_net(rng: random.Random) -> Arch:
    """Uniform over valid network configs (inference knobs at default)."""
    while True:
        a = Arch(
            rng.choice(N_VLM), rng.choice(N_EXP), rng.choice(BRIDGE), rng.choice(FFN), rng.choice(VTOK)
        )
        if a.is_valid():
            return a


def sample_net_depth_balanced(rng: random.Random) -> Arch:
    """Uniform over expert depth first, then uniform over the remaining valid axes."""
    e = rng.choice(N_EXP)
    while True:
        a = Arch(rng.choice(N_VLM), e, rng.choice(BRIDGE), rng.choice(FFN), rng.choice(VTOK))
        if a.is_valid():
            return a


def sample_arch(rng: random.Random) -> Arch:
    a = sample_net(rng)
    return Arch(*a.net, steps=rng.choice(STEPS), horizon=rng.choice(HORIZON))


def save_archs(archs: list[Arch], path: str) -> None:
    with open(path, "w") as f:
        json.dump([a.key() for a in archs], f, indent=1)


def load_archs(path: str) -> list[Arch]:
    with open(path) as f:
        return [Arch.from_key(k) for k in json.load(f)]
