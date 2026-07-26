# SPARC Runs

Run catalog for SPARC. All commands run from the SPARC repo root, after `source scripts/env.sh`
(which sets `$MODEL_PATH` and the runtime dirs) and with the assets present (see the README).
Commands use `uv run python` against the pinned `.venv`.

> **Cluster submission:** the current dielectric runs ship as ready-to-submit Slurm scripts —
> `sbatch scripts/<name>.sbatch` (each one `cd`s to the repo, `source`s `scripts/env.sh`, exports
> `HF_HUB_OFFLINE=1 SPARC_SKIP_ASSET_DOWNLOAD=1`, then runs the command shown below). For new jobs,
> fill in your account/partition in `scripts/slurm_gpu.sbatch` / `scripts/slurm_shared.sbatch`.

```
# Set by sourcing scripts/env.sh:
#   $MODEL_PATH   — data/symmcd_pretrained/mp_20
```

---

## Dielectric — in-plane isotropy (flagship / paper run)

`dielectric_inplane_isotropy_gapgate_mprime_newbg_b96` — SymmCD backbone, `reward=fom_inplane_isotropy_gapgate`.
Drives `ε_xx = ε_yy` (in-plane isotropy — the signature of the trigonal/tetragonal/hexagonal
families; cubic is excluded by a light z-anisotropy floor `z_anisotropy_floor=0.01`), gated by band
gap 0.3→0.8 eV so candidates stay non-metallic and worth DFT verification. 120 RL loops, generation
batch 96, MLIP relaxation on. **This is the run shown in the README animation.**
Submit: `sbatch scripts/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96.sbatch`, or directly:

```bash
uv run python main.py \
    expname=dielectric_inplane_isotropy_gapgate_mprime_newbg_b96 \
    pipeline=sparc model=symmcd reward=fom_inplane_isotropy_gapgate \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=120 pipeline.save_freq=10 \
    seed=1 deterministic_torch=true \
    'sample_cfg.filter.metrics=[validity,novel,unique]' \
    model.model_path="$MODEL_PATH" \
    model.sample_cfg.generation_batch_size=96 model.sample_cfg.batch_size=96 \
    model.sample_cfg.sg_temperature=100.0 model.sample_cfg.finetune_on_representative=true \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=5.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.15 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.symprec=0.01 \
    model.finetune_cfg.lr=3e-5 pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0
```

---

## Dielectric — in-plane emphasis (anisotropy-emphasis variant)

`dielectric_gapgate_inplane_emphasis_mprime_newbg_b96` — identical launcher, but
`reward=fom_layered_uniaxial_gapgate_inplane_emphasis` (scalar_mode `layered_uniaxial`,
`inplane_penalty_power=4.0`): scores the layered/uniaxial FoM (out-of-plane anisotropy `ε_zz ≠ ε_∥`)
with a strong in-plane-isotropy penalty, over the ascending window `[0.0, 0.20]`. Same band-gap
gate, same training knobs as the isotropy run.
Submit: `sbatch scripts/dielectric_gapgate_inplane_emphasis_mprime_newbg_b96.sbatch`.

---

## Dielectric — backbone × relaxation ablation (2×2)

Same in-plane-isotropy reward, toggling the diffusion backbone (SymmCD vs DiffCSP) and MLIP
relaxation. Ready-to-submit scripts:

| backbone | relax | submit script |
|----------|-------|---------------|
| SymmCD   | on    | `scripts/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96.sbatch` (flagship) |
| SymmCD   | off   | `scripts/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96_relaxF.sbatch` |
| DiffCSP  | on    | `scripts/dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_relaxT.sbatch` |
| DiffCSP  | off   | `scripts/dielectric_inplane_isotropy_gapgate_diffcsp_newbg_b96_relaxF.sbatch` |

The relax-off scripts add `+sample_cfg.filter.relax=false`; the DiffCSP scripts use `model=diffcsp`
(no `model.model_path` / `sg_temperature` / adaptive-space-group overrides — just `batch_size=96`).

---

## SLME — band gap + solar-cell efficiency (paper run)

`tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128` — SymmCD + M-prime,
`reward=tsenn_slme_optimate_bgcenter13_eta08_e3nngap`. Weighted (not gated) reward:
**0.2·band gap** + **0.8·SLME η** (`reduce: weight`). The band-gap term is a **hard-zero tent
centered at 1.3 eV** (Shockley–Queisser PV optimum, minv=0/maxv=1) scored with the project's own
E3NN band-gap model (`TSENN_bandgap.torch`); η (TSENN dielectric-spectra surrogate) is integrated
over the AM1.5G spectrum from that E3NN gap up at 0.3 µm thickness. Generation batch 128,
`sg_temperature=1.0`, `finetune_on_representative=true` (M-prime), `lr=3e-5`, `lr_decay=0.98`,
adaptive space groups on.

The sparc_v1 rerun ships as `sbatch scripts/slme_optimate_bgcenter13_eta08_e3nngap_mprime_b128.sbatch`
(`rl_epoch=120`; the original sparc_v0 run used `rl_epoch=200`). Direct command:

```bash
uv run python main.py \
    expname=slme_bgcenter13_eta08_e3nngap_mprime_b128 \
    pipeline=sparc model=symmcd reward=tsenn_slme_optimate_bgcenter13_eta08_e3nngap \
    logger=csv device=cuda \
    eval_size=30 rl_epoch=120 pipeline.save_freq=10 \
    seed=1 deterministic_torch=true \
    model.model_path="$MODEL_PATH" \
    model.sample_cfg.generation_batch_size=128 model.sample_cfg.batch_size=128 \
    model.sample_cfg.sg_temperature=1.0 model.sample_cfg.finetune_on_representative=true \
    sample_cfg.adaptive_spacegroup.enabled=true \
    sample_cfg.adaptive_spacegroup.ema_decay=0.9 \
    sample_cfg.adaptive_spacegroup.reward_scale=3.0 \
    sample_cfg.adaptive_spacegroup.prior_mix=0.1 \
    sample_cfg.adaptive_spacegroup.min_prob=1.0e-4 \
    +sample_cfg.adaptive_spacegroup.mix_previous_distribution=true \
    +sample_cfg.adaptive_spacegroup.symprec=0.01 \
    model.finetune_cfg.lr=3e-5 pipeline.topk_ratio=0.5 \
    pipeline.finetune_cfg.sigma=0.025 pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=0.98
```

**Ascending-gap variant** (`reward=tsenn_slme_optimate_bg02_eta08_e3nngap`): band gap ascending
[0.5, 3.0] eV instead of the centered tent. Same E3NN gap, η term, and weights.
