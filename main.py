import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("calc", eval, replace=True)


def _configure_randomness(cfg: DictConfig) -> None:
    seed = cfg.get("seed")
    if seed is None:
        return
    seed_value = int(seed)
    deterministic_torch = bool(cfg.get("deterministic_torch", True))
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import random
    import numpy as np
    import torch

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger.info(
        "Configured reproducible seed=%d deterministic_torch=%s",
        seed_value,
        deterministic_torch,
    )


@hydra.main(config_path="configs", config_name="base", version_base="1.1")
def main(cfg: DictConfig) -> None:
    _configure_randomness(cfg)
    OmegaConf.save(cfg, "hparams.yaml")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    reinl = hydra.utils.instantiate(
        cfg.pipeline,
        model_suite=cfg.model,
        reward=cfg.reward,
        logger=cfg.logger,
    )
    reinl.run_rl()


if __name__ == "__main__":
    main()
