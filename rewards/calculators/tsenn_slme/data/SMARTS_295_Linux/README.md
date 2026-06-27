# SMARTS reference solar spectrum (trimmed)

This directory originally vendored the full **SMARTS 2.9.5 (Linux)** distribution. For this
release it has been trimmed to only the precomputed spectrum actually read at runtime:

```
Examples/Example 6-USSA_084/smarts295.ext.txt
```

That file is the AM1.5-class global-tilted irradiance consumed by
`rewards/calculators/tsenn_slme/calc.py` (`solar_source="smarts"`, the default) and by
`scripts/analysis/plot_distribution_shift.py`.

`AAgreement_LICENSE.txt` is the original SMARTS license, retained for attribution. SMARTS was
written by Dr. Christian A. Gueymard; the full program (binary, source, documentation, and the
other example spectra) is **not** redistributed here — obtain it from NREL / the author if needed.

The sibling file `../ASTMG173.csv` (ASTM G173-03 reference spectra) is the alternative spectrum
selected via `solar_source="astm"`.
