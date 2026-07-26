#!/usr/bin/env python3
"""Warm third-party HuggingFace caches for offline / air-gapped compute nodes.

Besides the SPARC-specific assets (see scripts/download_assets.py), SPARC pulls a few
third-party checkpoints from the public HuggingFace Hub on first use:
  - the MatterSim potential and a MatterGen reference dataset (via pipeline.filters.OptFilter)

For normal (online) runs these download automatically. If your GPU nodes have no internet
but share ~/.cache/huggingface with an internet-connected login node, run this once on the
login node, then submit jobs with HF_HUB_OFFLINE=1 (see README).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.environ.setdefault("PROJECT_ROOT", _ROOT)


def main() -> int:
    import hydra.utils
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(_ROOT, "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(config_name="base", overrides=["device=cpu"])

    print("Warming MatterSim potential (5M) ...", flush=True)
    from mattersim.forcefield.potential import Potential

    Potential.from_checkpoint(load_path="MatterSim-v1.0.0-5M.pth", device="cpu")

    print("Warming OptFilter (MatterGen reference dataset) ...", flush=True)
    hydra.utils.instantiate(cfg.sample_cfg.filter)

    print("Done. Third-party weights cached under ~/.cache/huggingface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
