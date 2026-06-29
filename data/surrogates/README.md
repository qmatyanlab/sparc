# SPARC reward-surrogate checkpoints (`data/surrogates/`)

Trained TSENN / E3NN equivariant regressors used by the RL reward functions. The `.torch`
weights are **auto-downloaded from HuggingFace on first use** (`utils/assets.py`); each model
ships a config sidecar here (`*.yaml` / `*.meta.json`) documenting the **architecture + training
parameters**, so the checkpoints can be reused/retrained. The reward configs in
`configs/reward/` pass these same architecture params at load time — they must match the weights.

| checkpoint | predicts | out_dim / mode | em / layers / mul / lmax | r_max / num_neighbors | calculator | reward configs | sidecar |
|---|---|---|---|---|---|---|---|
| `TSENN_dielectric_spectra.torch` | isotropic ε₂(ω) **spectrum** (trace) → SLME η | 201 / trace | 128 / 4 / 64 / 2 | 6.0 / 38.86 | `TSENNSLME`→`TSENN` | `tsenn_slme_optimate*` | `.yaml` |
| `TSENN_dielectric_tensor_spectra.torch` | full ε₂(ω) **tensor** spectrum → SLME η | 300 / tensor | 128 / 2 / 64 / 2 | 6.0 / 47.94 | `TSENNSLME`→`TSENN` | `tsenn_slme`, `tsenn_slme_bandgap*` | `.meta.json` |
| `TSENN_static_dielectric_tensor.torch` | **static** dielectric tensor (3×3) | 1 / tensor | 64 / 2 / 32 / 2 | 6.0 / 59.90 | `TSENNStaticDielectric` | `tsenn_static_dielectric*`, `fom_layered_uniaxial*` | `.yaml` |
| `TSENN_bandgap.torch` | scalar **band gap** (softplus-OFF E3NN) | 1 / scalar | 128 / 4 / 16 / 2 | 6.0 / 55.33 | `E3NNBandGap` | `*_e3nngap`, `*_gapgate*` | `.yaml` |

All trained on Materials Project dielectric / band-gap data (preprocessed caches in
`data/dielectric/cached_*.pt`). The dielectric models use the TSENN `Network`; `TSENN_bandgap`
**must** be loaded through the softplus-OFF `Network` (see `rewards/calculators/e3nn_bandgap.py`)
— running it with softplus on floors every prediction to ~0.69 eV.
