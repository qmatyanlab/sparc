import os
from pathlib import Path
import numpy as np
from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure


def linear_scaling(values, minv=0.0, maxv=6.0):
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


def gaussian_target_reward(values, target, tau):
    diff = (values - target) / tau
    return np.exp(-0.5 * diff * diff)


def average_props(prop_dict):
    prop_num = len(prop_dict)
    prop_list = list(prop_dict.values())
    prop_sum = prop_list[0]
    for _prop in prop_list[1:]:
        prop_sum += _prop
    prop_mean = prop_sum / prop_num

    return prop_mean


def min_props(prop_dict):
    prop_list = list(prop_dict.values())
    prop_arr = np.array(prop_list)
    prop_min = prop_arr.min(axis=0)
    return prop_min


class Reward:
    def __init__(
        self,
        root_dir: str,
        prop_cfg: DictConfig,
        reward_threshold: float,
        reduce: str = "mean",
        **kwargs,
    ) -> None:
        assert reduce in ["mean", "min", "weight"]
        project_root = Path(__file__).resolve().parents[1]
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        self.root_dir = str(root_path)
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.cfg = OmegaConf.create(kwargs)
        self.reduce = reduce
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

    def calc_props(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        prop_dict, prop_list = {}, []
        for _cfg in self.prop_cfg:
            _prop1 = _cfg.calculator.calc(samples, label)
            prop_list.append(_prop1)
            _prop2 = np.nan_to_num(_prop1, nan=0.0)
            prop_dict[_cfg.name] = _prop2.astype(float)

        prop_list = np.array(prop_list)
        none_ids = np.isnan(prop_list).any(axis=0)

        return prop_dict, none_ids

    def _score_property(self, cfg: DictConfig, values: np.ndarray) -> np.ndarray:
        mode = str(cfg.get("reward_mode", "linear_target")).lower()
        if cfg.target == "ascending":
            return linear_scaling(
                values=values,
                minv=cfg.minv,
                maxv=cfg.maxv,
            )
        if cfg.target == "descending":
            return linear_scaling(
                values=-values,
                minv=-cfg.maxv,
                maxv=-cfg.minv,
            )
        if not isinstance(cfg.target, float):
            raise TypeError(
                "prop cfg.target must be a float or descending or ascending"
            )
        if mode == "linear_target":
            diff = np.abs(values - cfg.target)
            return linear_scaling(
                values=-diff,
                minv=-cfg.maxv,
                maxv=-cfg.minv,
            )
        if mode == "gaussian_target":
            tau = float(cfg.get("tau", 0.3))
            if tau <= 0.0:
                raise ValueError("gaussian_target reward requires tau > 0")
            reward = gaussian_target_reward(values=values, target=cfg.target, tau=tau)
            success_tol = cfg.get("success_tol", None)
            success_bonus = float(cfg.get("success_bonus", 0.0))
            if success_tol is not None and success_bonus != 0.0:
                success = (np.abs(values - cfg.target) < float(success_tol)).astype(
                    float
                )
                reward = reward + success_bonus * success
            return np.clip(reward, 0.0, 1.0)
        raise ValueError(f"Unknown reward_mode: {mode}")

    def scoring(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        prop_dict, failed_mask = self.calc_props(samples, label)

        scaled_prop_dict = {}
        for _cfg in self.prop_cfg:
            _sprop = self._score_property(_cfg, prop_dict[_cfg.name])
            scaled_prop_dict[_cfg.name] = _sprop

        if self.reduce == "mean":
            rewards = average_props(scaled_prop_dict)
        elif self.reduce == "min":
            rewards = min_props(scaled_prop_dict)
        elif self.reduce == "weight":
            for _cfg in self.prop_cfg:
                w = _cfg.weight
                scaled_prop_dict[_cfg.name] = scaled_prop_dict[_cfg.name] * w
            sprop_list = list(scaled_prop_dict.values())
            sprop_arr = np.array(sprop_list)
            rewards = sprop_arr.sum(axis=0)
        else:
            raise ValueError(f"Unknown reward reduce mode: {self.reduce}")

        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask
