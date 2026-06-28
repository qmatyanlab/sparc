from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from typing import Any, cast


def build_mlp(
    in_dim: int, hidden_dim: int, fc_num_layers: int, out_dim: int
) -> nn.Sequential:
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(fc_num_layers - 1):
        mods += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        hparams = cast(Any, self.hparams)
        opt = hydra.utils.instantiate(
            hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystGNN_Supervise(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
