def _safe_import(module_path: str, attr_name: str):
    try:
        module = __import__(module_path, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        return None


TSENN = _safe_import("rewards.calculators.tsenn.calc", "TSENN")
TSENNSLME = _safe_import("rewards.calculators.tsenn_slme.calc", "TSENNSLME")
TSENNStaticDielectric = _safe_import(
    "rewards.calculators.tsenn_static_dielectric", "TSENNStaticDielectric"
)
E3NNBandGap = _safe_import("rewards.calculators.e3nn_bandgap", "E3NNBandGap")
