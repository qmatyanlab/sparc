#!/usr/bin/env python3
"""
Phase C: fine-tune the E3NN band-gap surrogate on OOD DFT (PBE) labels.

Reuses the exact validated inference path (E3NNBandGap: _NetWrapper, build_graph,
onehot, softplus-OFF Network, float64), loads the frozen base checkpoint
data/surrogates/TSENN_bandgap.torch, and fine-tunes on the consolidated
dielectric_DFPT labels (Phase B). Writes a NEW checkpoint (base left intact) and
a metrics/parity artifact. Run several --seed values to build the deep ensemble
consumed by e3nn_bandgap_ensemble.py.

Forgetting control (keep PBE-IPA fidelity, don't wreck in-distribution):
  * low LR + early stopping on the val split;
  * optional --freeze-body (train only the embedding heads + final layer);
  * optional --rehearsal-cache (mix in a sample of the original MP training
    cache; run as a GPU job since the cache is ~4 GB).

Usage:
  python finetune_bandgap.py --seed 0 --epochs 80 --lr 1e-4 [--freeze-body]
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
from pymatgen.core import Structure

from rewards.calculators.e3nn_bandgap import E3NNBandGap
from rewards.calculators.tsenn.calc import build_graph

DATA = os.path.join(HERE, "data")
ART = os.path.join(HERE, "artifacts")
SURR = os.path.join(ROOT, "data", "surrogates")


def _load_split(name):
    recs = json.load(open(os.path.join(DATA, f"gap_labels_{name}.json")))
    return recs


def _make_data(recs, base):
    """Structures -> torch_geometric Data list (+ aligned gap targets)."""
    data_list, gaps = [], []
    for r in recs:
        try:
            s = Structure.from_dict(r["structure"])
            g = build_graph(s, base.type_onehot, base.mass_onehot,
                            base.dipole_onehot, base.radius_onehot,
                            base.type_encoding, base.r_max, dtype=base.dtype)
        except Exception:
            continue
        d = Data(pos=g.pos.double(), x=g.x.double(), z=g.z.double(),
                 edge_index=g.edge_index.long(), edge_vec=g.edge_vec.double())
        d.y = torch.tensor([float(r["gap"])], dtype=torch.float64)
        data_list.append(d)
        gaps.append(float(r["gap"]))
    return data_list, np.array(gaps)


def _evaluate(model, data_list, device, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            preds.append(model(batch.to(device)).view(-1).cpu().numpy())
    p = np.clip(np.concatenate(preds), 0.0, None)  # clip_negative like inference
    return p


def _metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss = float(np.sum((true - true.mean())**2))
    r2 = float(1 - np.sum(err**2) / ss) if ss > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(true))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--freeze-body", action="store_true",
                    help="train only embedding heads + final layer (less forgetting)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(ART, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = E3NNBandGap(root_dir=os.path.join(ART, "_base_cache"), device=device)
    model = base.model  # base checkpoint already loaded

    if args.freeze_body:
        for n, p in model.named_parameters():
            p.requires_grad = ("em_x" in n or "em_z" in n
                               or n.startswith("layers.") and ".3." in n)  # heads+last
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"device={device} trainable params={sum(p.numel() for p in trainable)} "
          f"(freeze_body={args.freeze_body})", flush=True)

    tr = _load_split("train"); va = _load_split("val"); oo = _load_split("ood_test")
    tr_d, tr_y = _make_data(tr, base)
    va_d, va_y = _make_data(va, base)
    oo_d, oo_y = _make_data(oo, base)
    print(f"graphs: train={len(tr_d)} val={len(va_d)} ood={len(oo_d)}", flush=True)

    base_val = _metrics(_evaluate(model, va_d, device), va_y)
    base_ood = _metrics(_evaluate(model, oo_d, device), oo_y)
    print(f"BASE  val MAE {base_val['mae']:.3f}  ood MAE {base_ood['mae']:.3f}", flush=True)

    opt = torch.optim.Adam(trainable, lr=args.lr)
    lossf = torch.nn.MSELoss()
    best_val, best_state, since = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in DataLoader(tr_d, batch_size=args.batch_size, shuffle=True):
            opt.zero_grad()
            out = model(batch.to(device)).view(-1)
            loss = lossf(out, batch.y.view(-1).to(device))
            loss.backward()
            opt.step()
        vm = _metrics(_evaluate(model, va_d, device), va_y)["mae"]
        if vm < best_val - 1e-4:
            best_val, best_state, since = vm, {k: v.detach().cpu().clone()
                                               for k, v in model.state_dict().items()}, 0
        else:
            since += 1
        if ep % 5 == 0 or since == 0:
            print(f"  ep{ep:3d} val MAE {vm:.3f} (best {best_val:.3f})", flush=True)
        if since >= args.patience:
            print(f"  early stop @ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ft_val = _metrics(_evaluate(model, va_d, device), va_y)
    ft_ood = _metrics(_evaluate(model, oo_d, device), oo_y)
    print(f"FT    val MAE {ft_val['mae']:.3f}  ood MAE {ft_ood['mae']:.3f} "
          f"(base ood {base_ood['mae']:.3f})", flush=True)

    out = args.out or os.path.join(SURR, f"TSENN_bandgap_ft_seed{args.seed}.torch")
    torch.save({"state": model.state_dict(), "seed": args.seed,
                "base": base.model_path}, out)
    metrics = {"seed": args.seed, "lr": args.lr, "epochs_ran": ep,
               "freeze_body": args.freeze_body,
               "base": {"val": base_val, "ood": base_ood},
               "finetuned": {"val": ft_val, "ood": ft_ood}}
    json.dump(metrics, open(os.path.join(ART, f"ft_metrics_seed{args.seed}.json"), "w"),
              indent=2)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
