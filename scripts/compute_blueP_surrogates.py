#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # surrogates are local; MatterSim ckpt + MP2020 ref are cached

from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402
from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric  # noqa: E402
from rewards.calculators.e3nn_bandgap import E3NNBandGap  # noqa: E402

SCRATCH = Path(os.environ.get("TMPDIR", "/tmp")) / "blueP_surrogate_root"
SCRATCH.mkdir(parents=True, exist_ok=True)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def main() -> None:
    cif = ROOT / "cifs_iso" / "P_SG144.cif"
    struct = Structure.from_file(cif)

    # --- dielectric tensor + in-plane-isotropy scalar (reward config exactly) ---
    diel = TSENNStaticDielectric(
        root_dir=str(SCRATCH / "diel"),
        task="static_dielectric",
        model_path=str(ROOT / "data/surrogates/TSENN_static_dielectric_tensor.torch"),
        config_path=str(ROOT / "data/surrogates/TSENN_static_dielectric_tensor.yaml"),
        device="cpu",
        batch_size=16,
        scalar_mode="inplane_isotropy",
        standardize_structure="refined",
        standardize_symprec=0.01,
        z_anisotropy_floor=0.01,
    )
    tensors, valid = diel.predict_static_tensor([struct])
    if not bool(valid[0]):
        raise SystemExit("dielectric surrogate returned invalid for blue-P")
    T = tensors[0]
    eps_xx, eps_yy, eps_zz = float(T[0, 0]), float(T[1, 1]), float(T[2, 2])
    iso_scalar = float(diel._reduce_tensor_to_scalar(tensors, valid)[0])

    # --- band gap ---
    bg = E3NNBandGap(
        root_dir=str(SCRATCH / "bandgap"),
        task="band_gap",
        model_path=str(ROOT / "data/surrogates/TSENN_bandgap.torch"),
        config_path=str(ROOT / "data/surrogates/TSENN_bandgap.yaml"),
        device="cpu",
        batch_size=256,
        silent=True,
    )
    gap = float(bg.predict([struct])[0])

    # --- reward r_uni (reduce=min) ---
    diel_term = clip01((iso_scalar - 0.90) / (1.00 - 0.90))
    gap_term = clip01((gap - 0.3) / (0.8 - 0.3))
    r_uni = min(diel_term, gap_term)

    # --- realized space group (symprec 0.01, matches reward standardize) ---
    sga = SpacegroupAnalyzer(struct, symprec=0.01)
    sgno = int(sga.get_space_group_number())
    sgsym = str(sga.get_space_group_symbol())

    # --- E_hull: MatterSim relax + MP-2020 hull (same as S.U.N. filter) ---
    ehull = None
    try:
        from pipeline.filters.opt_filter import (  # noqa: E402
            _relax_structures_with_progress,
            _get_reference,
            get_device,
        )
        from mattergen.evaluation.metrics.evaluator import MetricsEvaluator  # noqa: E402

        device = get_device("cpu")
        relaxed, energies = _relax_structures_with_progress(
            [struct], device=device,
            potential_load_path="MatterSim-v1.0.0-5M.pth", fmax=0.05,
        )
        ev = MetricsEvaluator.from_structures_and_energies(
            structures=relaxed,
            energies=[float(e) for e in energies],
            original_structures=[struct],
            reference=_get_reference(),
        )
        ehull = float(ev.energy_capability.energy_above_hull[0])
    except Exception as exc:  # noqa: BLE001
        print(f"[e_hull FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)

    # --- report ---
    print(f"space group        : {sgsym} (#{sgno})")
    print(f"eps (xx,yy,zz)     : {eps_xx:.3f}, {eps_yy:.3f}, {eps_zz:.3f}"
          f"   (in-plane mismatch |xx-yy|/(xx+yy) = {abs(eps_xx-eps_yy)/(abs(eps_xx)+abs(eps_yy)+1e-8):.4f})")
    print(f"inplane-iso scalar : {iso_scalar:.4f}  -> diel reward term {diel_term:.4f}")
    print(f"band gap (E3NN)    : {gap:.4f} eV      -> gap reward term  {gap_term:.4f}")
    print(f"r_uni (min)        : {r_uni:.4f}")
    print(f"E_hull (MatterSim) : {'%.4f eV/atom (%.1f meV)' % (ehull, ehull*1000) if ehull is not None else 'FAILED'}")

    meta_blob = {
        "META.P": {"name": "blue-P", "formula": "P", "sg": sgsym, "sgno": sgno, "accent": "#2a8f5a"},
        "PROPS.P": {"ehull": (f"{ehull*1000:.1f}" if ehull is not None else "NA"), "eg": f"{gap:.2f}"},
        "FULL_EPS.P": [[round(eps_xx, 3), 0.0, 0.0], [0.0, round(eps_yy, 3), 0.0], [0.0, 0.0, round(eps_zz, 3)]],
        "STARS.P": ["P", round(gap, 3), round(r_uni, 4), "#2a8f5a"],
        "CIF_PATHS.P": "cifs_iso/P_SG144.cif",
    }
    print(json.dumps(meta_blob, indent=2))


if __name__ == "__main__":
    main()
