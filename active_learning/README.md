# active_learning

Active-learning tooling for the SPARC SLME surrogates. Motivation: single-shot
DFT verification of the generator's top SLME candidates diverges sharply from the
frozen PBE-trained surrogates (e.g. IrF₃: ML gap 1.53 eV → QE 0.49 eV; several
"1.5 eV" candidates are metals). SLME is ~3:1 **band-gap-dominated** at 0.3 µm,
and the divergence is **OOD reward-hacking**, so the highest-leverage fix is to
fine-tune the band-gap surrogate on out-of-distribution DFT labels and add an
uncertainty gate — keeping PBE-IPA fidelity.

## Contents
- `consolidate_gap_labels.py` — build `(relaxed_structure → PBE gap)` fine-tuning
  labels from the sibling VASP project `dielectric_DFPT` (+ optional QE gaps).
  Family-level train/val/OOD split. → `data/gap_labels_{train,val,ood_test}.json`
- `attribution.py` — SLME-error attribution: 4-way swap (ML/QE gap × ML/QE ε₂)
  over the QE-verified candidates → how much of the η error is gap vs spectrum.
- `finetune_bandgap.py` — fine-tune `data/surrogates/TSENN_bandgap.torch` on the
  consolidated labels (MP-cache rehearsal to avoid forgetting) → `_ft.torch`.
- `e3nn_bandgap_ensemble.py` — deep ensemble wrapper → mean + std (OOD gate),
  drop-in for `E3NNBandGap` (same `predict`).

## Data provenance / fidelity
All labels are **MP-VASP-PBE** (spin-polarized, GGA(+U)) on **relaxed** geometry
— the same fidelity as the surrogate's original training set. r2SCAN is excluded.
QE-PBE gaps (unrelaxed cells) are a flagged, secondary source only.

## Env
Run under the repo `.venv` (`source ../.venv/bin/activate`); torch 2.2 + e3nn +
torch_geometric + torch_scatter + pymatgen.

Reproducibility: fixed seeds, saved splits, artifacts written under `data/` and
`artifacts/`. The original `TSENN_bandgap.torch` is never overwritten.
