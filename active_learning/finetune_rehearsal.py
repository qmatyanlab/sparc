#!/usr/bin/env python3
"""
Rehearsal fine-tune: retrain the base band-gap surrogate on MP-train UNION the
AL (dielectric_DFPT PBE) labels, so the model corrects OOD without forgetting MP.
Single seed. Then evaluate base / naive-soup / rehearsal on the SAME MP test set
(4260) and on the OOD test, and dump a 3-way parity npz for the figure.

MP structures live in the cache df['structure'] as ASE Atoms; AL labels are
pymatgen dicts from consolidate_gap_labels.py. Graphs are disk-cached.

Usage: python finetune_rehearsal.py [--mp-sample 6000] [--al-upsample 5]
       [--epochs 60] [--lr 1e-4]
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from rewards.calculators.e3nn_bandgap import E3NNBandGap
from rewards.calculators.tsenn.calc import build_graph

DATA = os.path.join(HERE, "data")
ART = os.path.join(HERE, "artifacts")
SURR = os.path.join(ROOT, "data", "surrogates")
CACHE = "/global/cfs/cdirs/m2663/angush/sparc/data/dielectric/cached_band_gap_preprocessed_data.pt"


def _mk(struct, base):
    g = build_graph(struct, base.type_onehot, base.mass_onehot, base.dipole_onehot,
                    base.radius_onehot, base.type_encoding, base.r_max, dtype=base.dtype)
    return Data(pos=g.pos.double(), x=g.x.double(), z=g.z.double(),
                edge_index=g.edge_index.long(), edge_vec=g.edge_vec.double())


def _mp_graphs(base, cache, rows, tag):
    """Build (disk-cached) MP graphs for the given df rows -> (data_list, y)."""
    cpath = os.path.join(ART, f"mpgraphs_{tag}.pt")
    if os.path.exists(cpath):
        blob = torch.load(cpath)
        return blob["data"], np.array(blob["y"])
    dl, y = [], []
    for atoms, gap in rows:
        try:
            s = AseAtomsAdaptor.get_structure(atoms) if isinstance(atoms, Atoms) else atoms
            dl.append(_mk(s, base)); y.append(float(gap))
        except Exception:
            continue
    torch.save({"data": dl, "y": y}, cpath)
    return dl, np.array(y)


def _al_graphs(base, split):
    recs = json.load(open(os.path.join(DATA, f"gap_labels_{split}.json")))
    dl, y = [], []
    for r in recs:
        try:
            dl.append(_mk(Structure.from_dict(r["structure"]), base)); y.append(float(r["gap"]))
        except Exception:
            continue
    return dl, np.array(y)


def _eval(model, dl, dev, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for b in DataLoader(dl, batch_size=bs, shuffle=False):
            out.append(model(b.to(dev)).view(-1).cpu().numpy())
    return np.clip(np.concatenate(out), 0.0, None)


def _mae(p, y): return float(np.mean(np.abs(p - y)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-sample", type=int, default=6000)
    ap.add_argument("--mp-val", type=int, default=1000)
    ap.add_argument("--al-upsample", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(ART, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading MP cache ...", flush=True)
    c = torch.load(CACHE, map_location="cpu")
    df = c["df"]; tgt = c["target_col"]
    itr, iva, ite = list(c["idx_train"]), list(c["idx_valid"]), list(c["idx_test"])
    rng = np.random.RandomState(args.seed)
    itr_s = list(rng.choice(itr, min(args.mp_sample, len(itr)), replace=False))
    iva_s = list(rng.choice(iva, min(args.mp_val, len(iva)), replace=False))

    base = E3NNBandGap(root_dir=os.path.join(ART, "_rh_base"), device=dev)
    print("building graphs (disk-cached) ...", flush=True)
    def rows(idx): return list(zip(df.iloc[idx]["structure"].values, df.iloc[idx][tgt].values))
    mp_tr, mp_tr_y = _mp_graphs(base, c, rows(itr_s), f"train{args.mp_sample}_s{args.seed}")
    mp_va, mp_va_y = _mp_graphs(base, c, rows(iva_s), f"val{args.mp_val}_s{args.seed}")
    mp_te, mp_te_y = _mp_graphs(base, c, rows(ite), "test_all")
    al_tr, al_tr_y = _al_graphs(base, "train")
    al_va, al_va_y = _al_graphs(base, "val")
    al_oo, al_oo_y = _al_graphs(base, "ood_test")
    print(f"MP: train {len(mp_tr)} val {len(mp_va)} test {len(mp_te)} | "
          f"AL: train {len(al_tr)} val {len(al_va)} ood {len(al_oo)}", flush=True)

    # attach targets, build mixed training set (upsample AL so it isn't drowned)
    for d, g in zip(mp_tr, mp_tr_y): d.y = torch.tensor([g], dtype=torch.float64)
    for d, g in zip(al_tr, al_tr_y): d.y = torch.tensor([g], dtype=torch.float64)
    mixed = list(mp_tr) + list(al_tr) * args.al_upsample
    print(f"mixed train set: {len(mixed)} (MP {len(mp_tr)} + AL {len(al_tr)}x{args.al_upsample})",
          flush=True)

    rh = E3NNBandGap(root_dir=os.path.join(ART, "_rh_ft"), device=dev)  # starts from base
    opt = torch.optim.Adam(rh.model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()
    best, best_state, since = 1e9, None, 0
    for ep in range(1, args.epochs + 1):
        rh.model.train()
        for b in DataLoader(mixed, batch_size=args.batch_size, shuffle=True):
            opt.zero_grad()
            out = rh.model(b.to(dev)).view(-1)
            lossf(out, b.y.view(-1).to(dev)).backward(); opt.step()
        # early stop on combined MP-val + AL-val
        cv = 0.5 * _mae(_eval(rh.model, mp_va, dev), mp_va_y) + \
             0.5 * _mae(_eval(rh.model, al_va, dev), al_va_y)
        if cv < best - 1e-4:
            best, since = cv, 0
            best_state = {k: v.detach().cpu().clone() for k, v in rh.model.state_dict().items()}
        else:
            since += 1
        if ep % 5 == 0 or since == 0:
            print(f"  ep{ep:3d} comb-val {cv:.3f} (best {best:.3f})", flush=True)
        if since >= args.patience:
            print(f"  early stop @ep{ep}", flush=True); break
    if best_state: rh.model.load_state_dict(best_state)

    soup = E3NNBandGap(root_dir=os.path.join(ART, "_rh_soup"),
                       model_path=os.path.join(SURR, "TSENN_bandgap_ft_soup.torch"), device=dev)

    # evaluate all three on the SAME MP test + OOD test
    pb, ps, pr = (_eval(base.model, mp_te, dev), _eval(soup.model, mp_te, dev),
                  _eval(rh.model, mp_te, dev))
    ob, osp, orh = (_eval(base.model, al_oo, dev), _eval(soup.model, al_oo, dev),
                    _eval(rh.model, al_oo, dev))
    print("\n=== MP test (n=%d) MAE ===" % len(mp_te_y))
    print(f"  base {_mae(pb,mp_te_y):.3f}  naive-soup {_mae(ps,mp_te_y):.3f}  "
          f"rehearsal {_mae(pr,mp_te_y):.3f}")
    print("=== OOD test (n=%d) MAE ===" % len(al_oo_y))
    print(f"  base {_mae(ob,al_oo_y):.3f}  naive-soup {_mae(osp,al_oo_y):.3f}  "
          f"rehearsal {_mae(orh,al_oo_y):.3f}")

    torch.save({"state": rh.model.state_dict(), "note": "rehearsal MP+AL"},
               os.path.join(SURR, "TSENN_bandgap_ft_rehearsal.torch"))
    np.savez(os.path.join(ART, "mp_test_parity3.npz"),
             y=mp_te_y, base=pb, soup=ps, rehearsal=pr)
    print("saved rehearsal checkpoint + artifacts/mp_test_parity3.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
