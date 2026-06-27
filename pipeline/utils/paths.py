from pathlib import Path


def resolve_save_dir(save_dir: str | None) -> Path:
    if save_dir is None or str(save_dir).strip() in {"", ".", "./"}:
        try:
            from hydra.core.hydra_config import HydraConfig

            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path.cwd()

    save_path = Path(save_dir)
    if not save_path.is_absolute():
        return Path.cwd() / save_path
    return save_path
