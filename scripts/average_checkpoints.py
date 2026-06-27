#!/usr/bin/env python3
"""Post-hoc weight averaging (SWA / uniform model soup) for SymmCD RL checkpoints.

Averages the `state_dict` tensors of several saved checkpoint directories into a
new checkpoint directory that is loadable by the normal SymmCD loader
(`models/suite/symmcd.py::_load_symmcd_model`) and samplable by the existing
sample_*.sh scripts.

Safe for SymmCD: the network uses LayerNorm only (no BatchNorm / running stats),
so no post-averaging recalibration pass is required.

Examples
--------
# 5 loop checkpoints nearest the best-reward step:
python scripts/average_checkpoints.py exp_res/<run> --around-best 5

# explicit members:
python scripts/average_checkpoints.py exp_res/<run> \
    --checkpoints loop_0069 loop_0079 loop_0089 loop_0099 loop_0109
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

CKPT_GLOB = "*.ckpt"
AUX_FILES = ("hparams.yaml", "lattice_scaler.pt", "prop_scaler.pt")
LOOP_RE = re.compile(r"loop_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="run directory containing models/")
    p.add_argument(
        "--around-best",
        type=int,
        default=None,
        metavar="K",
        help="average the K loop_* checkpoints whose step is nearest "
        "models/best_reward.json's step (default 5 if --checkpoints not given)",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="explicit checkpoint dir names (under models/) or absolute paths",
    )
    p.add_argument(
        "--include-best",
        action="store_true",
        help="also include models/best_reward in the average",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="name of the output checkpoint dir under models/ "
        "(default swa_best_k<K>)",
    )
    return p.parse_args()


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def ckpt_file(ckpt_dir: Path) -> Path:
    files = sorted(ckpt_dir.glob(CKPT_GLOB))
    if not files:
        raise FileNotFoundError(f"No {CKPT_GLOB} in {ckpt_dir}")
    # prefer a non-"last" checkpoint, matching the loader's preference
    non_last = [f for f in files if "last" not in f.name.lower()]
    return (non_last or files)[0]


def loop_step(ckpt_dir: Path) -> int | None:
    m = LOOP_RE.search(ckpt_dir.name)
    return int(m.group(1)) if m else None


def select_around_best(models_dir: Path, k: int) -> list[Path]:
    meta = require(models_dir / "best_reward.json", "best_reward.json")
    best_step = int(json.loads(meta.read_text())["step"])
    loops = [d for d in models_dir.iterdir() if d.is_dir() and loop_step(d) is not None]
    if not loops:
        raise FileNotFoundError(f"No loop_* checkpoints under {models_dir}")
    loops.sort(key=lambda d: abs(loop_step(d) - best_step))  # type: ignore[operator]
    chosen = sorted(loops[:k], key=lambda d: loop_step(d))  # type: ignore[arg-type]
    print(f"best_reward step={best_step}; nearest {len(chosen)} loop checkpoints:")
    for d in chosen:
        print(f"  {d.name} (step {loop_step(d)})")
    return chosen


def resolve_members(args: argparse.Namespace, models_dir: Path) -> list[Path]:
    members: list[Path] = []
    if args.checkpoints:
        for name in args.checkpoints:
            path = Path(name)
            if not path.is_absolute():
                path = models_dir / name
            members.append(require(path, f"checkpoint '{name}'"))
    else:
        k = args.around_best if args.around_best is not None else 5
        members = select_around_best(models_dir, k)
    if args.include_best:
        best = require(models_dir / "best_reward", "best_reward dir")
        if best not in members:
            members.append(best)
    if len(members) < 2:
        raise ValueError("Need at least 2 checkpoints to average")
    return members


def average_state_dicts(sds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set(sds[0].keys())
    for i, sd in enumerate(sds[1:], 1):
        if set(sd.keys()) != keys:
            raise ValueError(f"checkpoint {i} has mismatched state_dict keys")
    out: dict[str, torch.Tensor] = {}
    n = len(sds)
    for key in sds[0]:
        ref = sds[0][key]
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for sd in sds:
                acc += sd[key].to(torch.float64)
            out[key] = (acc / n).to(ref.dtype)
        else:
            # integer buffers (none in SymmCD today) must be identical; copy ref
            for sd in sds[1:]:
                if not torch.equal(sd[key], ref):
                    raise ValueError(
                        f"non-float tensor '{key}' differs across checkpoints; "
                        "cannot average safely"
                    )
            out[key] = ref.clone()
    return out


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    models_dir = require(run_dir / "models", "models directory")
    members = resolve_members(args, models_dir)

    sds, ckpt_paths = [], []
    base_ckpt = None
    for d in members:
        cf = ckpt_file(d)
        ckpt_paths.append(cf)
        blob = torch.load(cf, map_location="cpu", weights_only=False)
        if base_ckpt is None:
            base_ckpt = blob
        sds.append(blob["state_dict"])

    print(f"\nAveraging {len(sds)} checkpoints ({len(sds[0])} tensors each)...")
    avg_sd = average_state_dicts(sds)

    k = len(members)
    out_name = args.output_name or f"swa_best_k{k}"
    out_dir = models_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt["state_dict"] = avg_sd  # type: ignore[index]
    base_ckpt["epoch"] = 0  # type: ignore[index]
    base_ckpt["global_step"] = 0  # type: ignore[index]
    torch.save(base_ckpt, out_dir / "epoch=0-step=0.ckpt")

    src_aux = members[0]
    for fname in AUX_FILES:
        src = src_aux / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    manifest = {
        "method": "uniform_soup",
        "num_members": k,
        "members": [
            {"dir": str(d), "step": loop_step(d), "ckpt": str(cf)}
            for d, cf in zip(members, ckpt_paths)
        ],
        "weights": [1.0 / k] * k,
        "output_dir": str(out_dir),
    }
    (out_dir / "average_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote averaged checkpoint -> {out_dir}")
    print(f"  members: {[d.name for d in members]}")
    print(f"  aux copied from: {src_aux.name}")
    print(f"  manifest: {out_dir / 'average_manifest.json'}")


if __name__ == "__main__":
    main()
