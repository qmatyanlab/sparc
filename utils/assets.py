"""Auto-download of large SPARC assets from a (private) HuggingFace repo.

Large model checkpoints and datasets are NOT stored in git. They live in a single
combined HuggingFace repo and are downloaded on first use to the exact repo-relative
paths the Hydra configs / pretrained-checkpoint hparams already expect, so **no config
changes are needed**.

Per-asset resolution order:
  1. If the file already exists under the project root -> use it (no network).
  2. Else if it is a known asset (in ``REGISTRY``) -> download it from the HF repo into
     the same repo-relative path (``hf_hub_download(..., local_dir=PROJECT_ROOT)``).
  3. Else -> return the path unchanged and let the caller handle a missing file.

Environment variables:
  PROJECT_ROOT               project root used to place files (default: this repo root)
  SPARC_HF_REPO              HF repo id (default: ``AngusHsuPhys/sparc-assets``)
  SPARC_HF_REPO_TYPE         ``model`` (default) or ``dataset``
  SPARC_HF_REVISION          optional git revision / branch / tag of the asset repo
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN   token for a private repo (or ``huggingface-cli login``)
  SPARC_SKIP_ASSET_DOWNLOAD  if set, never download (use only local files; for air-gapped nodes)
  HF_HUB_OFFLINE / SPARC_OFFLINE      offline mode (HF cache only)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

DEFAULT_REPO_ID = "AngusHsuPhys/sparc-assets"
DEFAULT_REPO_TYPE = "model"


def repo_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def repo_id() -> str:
    return os.environ.get("SPARC_HF_REPO", DEFAULT_REPO_ID)


def repo_type() -> str:
    return os.environ.get("SPARC_HF_REPO_TYPE", DEFAULT_REPO_TYPE)


def _revision():
    return os.environ.get("SPARC_HF_REVISION") or None


def _token():
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(key)
        if tok:
            return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001
        return None


def _offline() -> bool:
    return bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("SPARC_OFFLINE"))


def _skip() -> bool:
    return bool(os.environ.get("SPARC_SKIP_ASSET_DOWNLOAD"))


# ---------------------------------------------------------------------------
# Registry: repo-relative path == HF filename. Group in {core, cached, optional}.
#   core     -> needed to run the published RL experiments (~4.2 GB)
#   cached   -> only needed to RE-TRAIN the surrogate models (~8.6 GB)
#   optional -> also shipped in git; download is a redundant safety net
# ---------------------------------------------------------------------------
REGISTRY: dict[str, str] = {}


def _reg(group: str, *paths: str) -> None:
    for p in paths:
        REGISTRY[p] = group


# SymmCD pretrained base checkpoint (required for any SymmCD run)
_reg(
    "core",
    "data/symmcd_pretrained/mp_20/hparams.yaml",
    "data/symmcd_pretrained/mp_20/epoch=0-step=0.ckpt",
    "data/symmcd_pretrained/mp_20/lattice_scaler.pt",
    "data/symmcd_pretrained/mp_20/prop_scaler.pt",
)
# MP-20 sampling statistics + datasets
_reg(
    "core",
    "data/mp_20/sg_info.pt",
    "data/mp_20/train_ori.pt",
    "data/mp_20/val_ori.pt",
    "data/mp_20/train.csv",
    "data/mp_20/val.csv",
    "data/mp_20/test.csv",
    "data/mp_20/train_atom_types_marginals.pt",
    "data/mp_20/train_atom_types_marginals_per_sg.pt",
    "data/mp_20/train_site_symm_marginals_per_sg.pt",
    "data/mp_20/test_mp20_formation_energy_per_atom.pt",
)
# Reward surrogate models
_reg(
    "core",
    "data/surrogates/TSENN_dielectric_spectra.torch",
    "data/surrogates/TSENN_static_dielectric_tensor.torch",
    "data/surrogates/TSENN_bandgap.torch",
    "data/surrogates/TSENN_dielectric_tensor_spectra.torch",
)
# Preprocessed training caches (only to RE-TRAIN the surrogates)
_reg(
    "cached",
    "data/dielectric/cached_band_gap_preprocessed_data.pt",
    "data/dielectric/cached_dielectric_preprocessed_data.pt",
    "data/dielectric/cached_dielectric_spectra_preprocessed_data.pt",
)
# Small files also shipped in git (redundant safety net)
_reg(
    "optional",
    "data/onehot/type_onehot.torch",
    "data/onehot/mass_onehot.torch",
    "data/onehot/dipole_onehot.torch",
    "data/onehot/radius_onehot.torch",
    "data/onehot/type_encoding.torch",
)


def _to_relpath(path) -> str | None:
    """Normalize an absolute-or-relative path to a repo-relative posix string."""
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace(os.sep, "/")
    try:
        rel = os.path.relpath(p, repo_root())
    except ValueError:  # e.g. different drive on Windows
        return None
    if rel.startswith(".."):
        return None
    return rel.replace(os.sep, "/")


def _download(relpath: str) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        got = hf_hub_download(
            repo_id=repo_id(),
            repo_type=repo_type(),
            filename=relpath,
            revision=_revision(),
            local_dir=str(repo_root()),
            token=_token(),
            local_files_only=_offline(),
        )
        return Path(got)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Required SPARC asset '{relpath}' is missing locally and could not be "
            f"downloaded from HuggingFace repo '{repo_id()}' (type={repo_type()}). "
            f"If the repo is private, set HF_TOKEN or run `huggingface-cli login` and "
            f"make sure your account has access. Original error: {exc}"
        ) from exc


def resolve_path(path, *, optional: bool = False) -> Path:
    """Return an absolute path to ``path``, downloading it from HF if missing.

    Accepts an absolute or repo-relative path. No-op (no network) when the file
    already exists. Unknown / user-supplied paths are returned unchanged so the
    caller's own existence check still applies.
    """
    root = repo_root()
    relpath = _to_relpath(path)
    target = (root / relpath) if relpath is not None else Path(path)
    if target.exists():
        return target
    if relpath is None or relpath not in REGISTRY:
        return target
    if _skip():
        return target
    is_optional = optional or REGISTRY[relpath] == "optional"
    try:
        return _download(relpath)
    except FileNotFoundError:
        if is_optional:
            return target
        raise


def ensure_assets(groups: Iterable[str] = ("core",)) -> list[Path]:
    """Eagerly resolve every registered asset in the given group(s)."""
    wanted = set(groups)
    resolved: list[Path] = []
    for relpath, group in REGISTRY.items():
        if group in wanted:
            resolved.append(resolve_path(relpath, optional=(group == "optional")))
    return resolved


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Prefetch SPARC assets from HuggingFace.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="also fetch the large preprocessed training caches (~8.6 GB)",
    )
    ap.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="explicit groups to fetch (core / cached / optional)",
    )
    args = ap.parse_args(argv)
    if args.groups:
        groups = tuple(args.groups)
    else:
        groups = ("core", "optional") + (("cached",) if args.all else ())
    print(f"PROJECT_ROOT = {repo_root()}")
    print(f"HF repo = {repo_id()} (type={repo_type()})  groups={groups}")
    resolved = ensure_assets(groups)
    print(f"Resolved {len(resolved)} assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
