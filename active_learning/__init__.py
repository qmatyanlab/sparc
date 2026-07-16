"""Active-learning tooling for the SPARC SLME surrogates.

See README.md. Modules:
  consolidate_gap_labels  - build PBE (structure -> gap) fine-tuning labels
  attribution             - SLME-error attribution (gap vs spectra)
  finetune_bandgap        - fine-tune the E3NN band-gap surrogate
  e3nn_bandgap_ensemble   - deep-ensemble wrapper (mean + std OOD gate)
"""
