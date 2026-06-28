# SPARC Runs

Run catalog for SPARC. All commands run from the SPARC root (`/home/angush/sparc/`).

> **Before running any command:** `source scripts/env.sh`
> **On NERSC:** edit `scripts/submit_nersc.sh` with your command, then `sbatch scripts/submit_nersc.sh`

```
# Set by sourcing scripts/env.sh:
#   $PYTHON       — .venv/bin/python
#   $MODEL_PATH   — data/symmcd_pretrained/mp_20
```

---

## bandgap4_symmcd_v2 — 2026-04-?? (early test, ran in sym_prop_mat)

Band gap 3.0 eV with SymmCD. First working run with OptFilter enabled.
No sg_temperature override, no lr/sigma tuning. Short run (60 epochs).

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap4_symmcd_v2 \
    pipeline=sparc model=symmcd reward=band_gap \
    logger=csv device=cuda \
    eval_size=64 rl_epoch=60 \
    model.model_path="${MODEL_PATH}" \
    > exp_res/bandgap4_symmcd_v2.log 2>&1 &
```

---

## bandgap3_symmcd_sgtemp3_test_v4 — 2026-04-18 (ran in sym_prop_mat)

Band gap 3.0 eV. `sg_temperature=3.0` to encourage SG diversity.
`reward=band_gap` at the time had `maxv=3.0` (wider reward basin).
lr=1e-4 (default), sigma=0.025 fixed. **Best SG diversity of all runs.**

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap3_symmcd_sgtemp3_test_v4 \
    pipeline=sparc model=symmcd reward=band_gap \
    logger=wandb device=cuda \
    eval_size=32 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    > exp_res/bandgap3_symmcd_sgtemp3_test_v4.log 2>&1 &
```

---

## bandgap3_symmcd_sgtemp3_test_v5 — 2026-04-19 (ran in sym_prop_mat)

Band gap 3.0 eV. `reward=band_gap` config changed to `maxv=2.0` (narrower basin).
`fmax=0.05, max_n_steps=5000` for more accurate MLIP relaxation.
lr=1e-4, sigma=0.025 fixed. **SG diversity collapsed vs v4 (SG123/191 dominated).**

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap3_symmcd_sgtemp3_test_v5 \
    pipeline=sparc model=symmcd reward=band_gap \
    logger=wandb device=cuda \
    eval_size=32 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    > exp_res/bandgap3_symmcd_sgtemp3_test_v5.log 2>&1 &
```

---

## bandgap3_symmcd_sgtemp3_test_v6 — 2026-04-20 (ran in sym_prop_mat)

Band gap 3.0 eV. Attempted sigma inverse-sqrt scheduling (`sigma=0.05, decay_steps=50`).
**Bug**: `sigma_decay_steps` override silently ignored (Hydra struct mode, key not in YAML).
Sigma stayed fixed at 0.025 (default). lr=1e-4 still too high — strong oscillation.

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap3_symmcd_sgtemp3_test_v6 \
    pipeline=sparc model=symmcd reward=band_gap \
    logger=wandb device=cuda \
    eval_size=32 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    pipeline.finetune_cfg.sigma=0.05 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    > exp_res/bandgap3_symmcd_sgtemp3_test_v6.log 2>&1 &
```

---

## bandgap3_symmcd_sgtemp3_test_v7 — 2026-04-21 (ran in sym_prop_mat)

Band gap 3.0 eV. **lr fixed to paper's 1e-5** (was 1e-4, 10x too high).
sigma scheduling now works (YAML key added to `configs/pipeline/sparc.yaml`).

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap3_symmcd_sgtemp3_test_v7 \
    pipeline=sparc model=symmcd reward=band_gap \
    logger=wandb device=cuda \
    eval_size=48 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=128 \
    model.sample_cfg.sg_temperature=3.0 \
    model.finetune_cfg.lr=1e-5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    > exp_res/bandgap3_symmcd_sgtemp3_test_v7.log 2>&1 &
```

---

## bandgap3_symmcd_sgtemp3_test_v8 — TBD

Band gap 3.0 eV. **Restores v4's wide reward basin** (`reward=band_gap_wide`, `maxv=3.0`)
with all v7 training fixes (lr=1e-5, sigma scheduling, generation_batch_size=256).
Goal: recover v4's SG diversity while improving bandgap precision.

```bash
nohup ${PYTHON} -u main.py \
    expname=bandgap3_symmcd_sgtemp3_test_v8 \
    pipeline=sparc model=symmcd reward=band_gap_wide \
    logger=wandb device=cuda \
    eval_size=32 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=256 \
    model.sample_cfg.sg_temperature=3.0 \
    model.finetune_cfg.lr=1e-5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    > exp_res/bandgap3_symmcd_sgtemp3_test_v8.log 2>&1 &
```

---

## slme_symmcd_sgtemp3_v1 — TBD

First SLME solar cell efficiency run. Multi-objective: band_gap (weight=0.4) + SLME η (weight=0.6).
Uses TSENN-predicted dielectric + AM1.5G spectrum for η computation.
Same training config as v7/v8.

```bash
nohup ${PYTHON} -u main.py \
    expname=slme_symmcd_sgtemp3_v1 \
    pipeline=sparc model=symmcd reward=tsenn_slme \
    logger=wandb device=cuda \
    eval_size=32 rl_epoch=200 \
    model.model_path="${MODEL_PATH}" \
    model.sample_cfg.generation_batch_size=256 \
    model.sample_cfg.sg_temperature=3.0 \
    model.finetune_cfg.lr=1e-5 \
    pipeline.finetune_cfg.sigma=0.025 \
    pipeline.finetune_cfg.sigma_decay_steps=50 \
    pipeline.finetune_cfg.lr_decay=1.0 \
    > exp_res/slme_symmcd_sgtemp3_v1.log 2>&1 &
```
