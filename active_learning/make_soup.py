#!/usr/bin/env python3
"""
Average the fine-tuned ensemble seeds into a single "model soup" checkpoint,
a drop-in replacement for data/surrogates/TSENN_bandgap.torch.

Valid here because every seed was fine-tuned from the SAME base checkpoint on the
SAME data (seeds differ only by init/shuffle), so the weights live in one loss
basin -> averaging them ~= the ensemble mean (which had std ~0.01 eV anyway),
usable via eg_model_path with NO change to the reward/pipeline code.

Usage: python make_soup.py [--glob ...] [--out ...]
"""
import os
import sys
import glob
import argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SURR = os.path.join(os.path.dirname(HERE), "data", "surrogates")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(SURR, "TSENN_bandgap_ft_seed*.torch"))
    ap.add_argument("--out", default=os.path.join(SURR, "TSENN_bandgap_ft_soup.torch"))
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no checkpoints match {args.glob}", file=sys.stderr)
        return 1
    print(f"averaging {len(paths)} checkpoints:")
    states = [torch.load(p, map_location="cpu")["state"] for p in paths]
    keys = states[0].keys()
    soup = {}
    for k in keys:
        stacked = torch.stack([s[k].double() for s in states], dim=0)
        soup[k] = stacked.mean(dim=0).to(states[0][k].dtype)
    torch.save({"state": soup, "note": f"model-soup of {len(paths)} seeds",
                "members": [os.path.basename(p) for p in paths]}, args.out)
    for p in paths:
        print(f"  {os.path.basename(p)}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
