import os
from pathlib import Path
from typing import Any


class Calculator:
    def __init__(
        self,
        root_dir: str,
        task: str,
    ) -> None:
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path
        self.root_dir = str(root_path)
        self.task = task
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

    def calc(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # --- shared model-architecture config loading (YAML `model:` block) --------------
    # Surrogate calculators load architecture params from a model-card YAML (the
    # data/surrogates/*.yaml files) instead of listing every param inline in each reward
    # config. `config_path` (or a per-calculator default) points at that YAML; only its
    # top-level `model:` sub-block is read. Precedence: explicit __init__ arg > YAML value
    # > hardcoded default (see `pick`).
    @staticmethod
    def load_model_cfg(config_path: str | None, default_config: str | None = None) -> dict:
        """Return the `model:` sub-dict of the resolved YAML, or {} if absent/missing."""
        import yaml
        from utils.assets import resolve_path

        cfg = config_path or default_config
        if not cfg:
            return {}
        cfg_path = str(resolve_path(cfg, optional=True))
        if not os.path.exists(cfg_path):
            return {}
        return (yaml.safe_load(open(cfg_path)) or {}).get("model", {}) or {}

    @staticmethod
    def pick(mcfg: dict, arg: Any, key: str, default: Any) -> Any:
        """explicit arg (if not None) > YAML `model:` value > hardcoded default."""
        return arg if arg is not None else mcfg.get(key, default)
