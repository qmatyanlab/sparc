import csv
import json
import math
import os
import time
import logging
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pipeline.base import ReinL
from pipeline.filters import OptEval, invalid_filter
from pipeline.utils.save import save_data_records, save_structures
from pipeline.utils.logger import Logger
from rewards.reward import Reward
from models.suite.base import ModelSuite


class MatInvent(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
        checkpoint_eval: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        if "filter" not in self.sample_cfg:
            self.opt_eval = OptEval()

        self.best_reward_mean = float("-inf")
        self.best_reward_step = -1
        self.checkpoint_eval = checkpoint_eval
        self.best_hit_rate = float("-inf")
        self.best_hit_step = -1
        self.best_hit_count = -1
        self.best_hit_reward_mean = float("-inf")

        self.load_model()
        self._init_adaptive_spacegroup_policy()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.requires_grad = True
        # Freeze the parameter of prior (pretrained) model
        for param in self.prior.parameters():
            param.requires_grad = False
        self.agent.to(self.device)
        self.prior.to(self.device)

    def _load_adaptive_sg_distribution_csv(
        self, path: str, support_mask: np.ndarray
    ) -> np.ndarray:
        raw_dist = np.zeros(230, dtype=float)
        seen = set()
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Adaptive SG distribution CSV is empty: {path}")
            missing = {"spacegroup", "current_prob"} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Adaptive SG distribution CSV {path} is missing columns: {sorted(missing)}"
                )
            for row in reader:
                try:
                    sg = int(row["spacegroup"])
                    prob = float(row["current_prob"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid adaptive SG distribution row in {path}: {row}"
                    ) from exc
                if sg < 1 or sg > 230:
                    raise ValueError(
                        f"Adaptive SG distribution CSV {path} has invalid spacegroup {sg}"
                    )
                if sg in seen:
                    raise ValueError(
                        f"Adaptive SG distribution CSV {path} has duplicate spacegroup {sg}"
                    )
                if not np.isfinite(prob) or prob < 0.0:
                    raise ValueError(
                        f"Adaptive SG distribution CSV {path} has invalid probability {prob} for SG {sg}"
                    )
                seen.add(sg)
                raw_dist[sg - 1] = prob
        if len(seen) != 230:
            missing_sgs = sorted(set(range(1, 231)) - seen)
            raise ValueError(
                f"Adaptive SG distribution CSV {path} is missing {len(missing_sgs)} spacegroups; first missing SG={missing_sgs[0]}"
            )

        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != raw_dist.shape:
            raise ValueError(
                f"Adaptive SG support mask has shape {support_mask.shape}, expected {raw_dist.shape}"
            )
        unsupported_mass = float(raw_dist[~support_mask].sum())
        if unsupported_mass > 1e-8:
            logging.warning(
                "Adaptive SG distribution %s assigns %.6g probability to unsupported spacegroups; dropping that mass before normalization.",
                path,
                unsupported_mass,
            )
        raw_dist[~support_mask] = 0.0
        total = float(raw_dist.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(
                f"Adaptive SG distribution CSV {path} has no positive supported probability mass"
            )
        return raw_dist / total

    def _init_adaptive_spacegroup_policy(self) -> None:
        cfg = OmegaConf.select(self.sample_cfg, "adaptive_spacegroup")
        self.adaptive_sg_cfg = cfg if cfg is not None else OmegaConf.create({})
        self.adaptive_sg_enabled = bool(self.adaptive_sg_cfg.get("enabled", False))
        self.adaptive_sg_dir = os.path.join(self.save_dir, "adaptive_spacegroup")
        self.adaptive_sg_base_dist = None
        self.adaptive_sg_dist = None
        self.adaptive_sg_reward_ema = None
        mix_previous_cfg = OmegaConf.select(
            self.adaptive_sg_cfg, "mix_previous_distribution"
        )
        legacy_carryover_cfg = OmegaConf.select(
            self.adaptive_sg_cfg, "carryover_distribution"
        )
        if mix_previous_cfg is None:
            self.adaptive_sg_mix_previous = bool(legacy_carryover_cfg or False)
        else:
            self.adaptive_sg_mix_previous = bool(mix_previous_cfg)
            if (
                legacy_carryover_cfg is not None
                and bool(legacy_carryover_cfg) != self.adaptive_sg_mix_previous
            ):
                logging.warning(
                    "adaptive_spacegroup.mix_previous_distribution=%s overrides legacy carryover_distribution=%s",
                    self.adaptive_sg_mix_previous,
                    bool(legacy_carryover_cfg),
                )
        if not self.adaptive_sg_enabled:
            return
        if not hasattr(self.sampler, "get_sampling_sg_distribution"):
            logging.warning(
                "adaptive_spacegroup.enabled=true but sampler %s does not expose SG proposal control; disabling adaptive SG policy.",
                type(self.sampler).__name__,
            )
            self.adaptive_sg_enabled = False
            return
        ema_decay = float(self.adaptive_sg_cfg.get("ema_decay", 0.8))
        prior_mix = float(self.adaptive_sg_cfg.get("prior_mix", 0.5))
        reward_scale = float(self.adaptive_sg_cfg.get("reward_scale", 1.0))
        min_prob = float(self.adaptive_sg_cfg.get("min_prob", 1e-4))
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                f"adaptive_spacegroup.ema_decay must be in [0, 1), got {ema_decay}"
            )
        if not 0.0 <= prior_mix <= 1.0:
            raise ValueError(
                f"adaptive_spacegroup.prior_mix must be in [0, 1], got {prior_mix}"
            )
        if not np.isfinite(reward_scale):
            raise ValueError("adaptive_spacegroup.reward_scale must be finite")
        if min_prob < 0.0:
            raise ValueError(
                f"adaptive_spacegroup.min_prob must be >= 0, got {min_prob}"
            )
        base_dist = np.asarray(
            self.sampler.get_sampling_sg_distribution(
                self.agent,
                restrict_spacegroups=self.sample_cfg.get("restrict_spacegroups"),
            ),
            dtype=float,
        )
        self.adaptive_sg_base_dist = base_dist
        self.adaptive_sg_dist = base_dist.copy()
        self.adaptive_sg_reward_ema = np.zeros_like(base_dist)
        proposal_source = "base"
        initial_distribution_csv = OmegaConf.select(
            self.adaptive_sg_cfg, "initial_distribution_csv"
        )
        if initial_distribution_csv:
            initial_distribution_csv = str(initial_distribution_csv)
            self.adaptive_sg_dist = self._load_adaptive_sg_distribution_csv(
                initial_distribution_csv,
                support_mask=base_dist > 0,
            )
            top_index = int(np.argmax(self.adaptive_sg_dist))
            proposal_source = f"csv:{initial_distribution_csv}"
            logging.info(
                "Loaded adaptive SG initial distribution from %s: top_sg=%d prob=%.4f",
                initial_distribution_csv,
                top_index + 1,
                float(self.adaptive_sg_dist[top_index]),
            )
        os.makedirs(self.adaptive_sg_dir, exist_ok=True)
        self._save_adaptive_sg_snapshot(
            step=-1,
            step_counts=np.zeros_like(base_dist, dtype=int),
            step_mean_rewards=np.full_like(base_dist, np.nan, dtype=float),
            prior_prob_used=self.adaptive_sg_dist.copy(),
        )
        logging.info(
            "Adaptive SG policy enabled with prior_mix=%.3f, ema_decay=%.3f, reward_scale=%.3f, min_prob=%.6f, proposal_source=%s, mix_previous_distribution=%s",
            prior_mix,
            ema_decay,
            reward_scale,
            min_prob,
            proposal_source,
            self.adaptive_sg_mix_previous,
        )

    def _extract_realized_spacegroup(self, structure: Structure) -> int | None:
        symprec = float(self.adaptive_sg_cfg.get("symprec", 0.01))
        try:
            return int(
                SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
            )
        except Exception:
            return None

    def _compute_adaptive_sg_step_stats(
        self, sample_struc: list[Structure], rewards: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(230, dtype=int)
        reward_sums = np.zeros(230, dtype=float)
        for structure, reward in zip(sample_struc, rewards):
            if not np.isfinite(reward):
                continue
            spacegroup = self._extract_realized_spacegroup(structure)
            if spacegroup is None or spacegroup < 1 or spacegroup > 230:
                continue
            index = spacegroup - 1
            counts[index] += 1
            reward_sums[index] += float(reward)
        mean_rewards = np.full(230, np.nan, dtype=float)
        observed = counts > 0
        mean_rewards[observed] = reward_sums[observed] / counts[observed]
        return counts, mean_rewards

    def _save_adaptive_sg_snapshot(
        self,
        step: int,
        step_counts: np.ndarray,
        step_mean_rewards: np.ndarray,
        prior_prob_used: np.ndarray,
    ) -> None:
        if not self.adaptive_sg_enabled:
            return
        rows = []
        assert self.adaptive_sg_base_dist is not None
        assert self.adaptive_sg_dist is not None
        assert self.adaptive_sg_reward_ema is not None
        for sg in range(1, 231):
            index = sg - 1
            rows.append(
                {
                    "spacegroup": sg,
                    "base_prob": float(self.adaptive_sg_base_dist[index]),
                    "prior_prob_used": float(prior_prob_used[index]),
                    "current_prob": float(self.adaptive_sg_dist[index]),
                    "reward_ema": float(self.adaptive_sg_reward_ema[index]),
                    "step_count": int(step_counts[index]),
                    "step_mean_reward": ""
                    if np.isnan(step_mean_rewards[index])
                    else float(step_mean_rewards[index]),
                }
            )
        filename = "initial.csv" if step < 0 else f"step_{step:0>4d}.csv"
        path = os.path.join(self.adaptive_sg_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _adaptive_sg_log_dict(self) -> Dict:
        if not self.adaptive_sg_enabled:
            return {}
        assert self.adaptive_sg_dist is not None
        positive = self.adaptive_sg_dist[self.adaptive_sg_dist > 0]
        top_index = int(np.argmax(self.adaptive_sg_dist))
        return {
            "adaptive_sg_entropy": float(-(positive * np.log(positive)).sum()),
            "adaptive_sg_top_sg": top_index + 1,
            "adaptive_sg_top_prob": float(self.adaptive_sg_dist[top_index]),
        }

    def _update_adaptive_sg_policy(
        self, sample_struc: list[Structure], rewards: np.ndarray
    ) -> None:
        if not self.adaptive_sg_enabled or len(sample_struc) == 0 or len(rewards) == 0:
            return
        assert self.adaptive_sg_base_dist is not None
        assert self.adaptive_sg_dist is not None
        assert self.adaptive_sg_reward_ema is not None
        counts, mean_rewards = self._compute_adaptive_sg_step_stats(
            sample_struc, rewards
        )
        observed = counts > 0
        if not observed.any():
            logging.info("Adaptive SG policy skipped: no valid realized-SG rewards")
            return
        ema_decay = float(self.adaptive_sg_cfg.get("ema_decay", 0.8))
        reward_scale = float(self.adaptive_sg_cfg.get("reward_scale", 1.0))
        prior_mix = float(self.adaptive_sg_cfg.get("prior_mix", 0.5))
        min_prob = float(self.adaptive_sg_cfg.get("min_prob", 1e-4))
        previous_dist = self.adaptive_sg_dist.copy()
        self.adaptive_sg_reward_ema *= ema_decay
        self.adaptive_sg_reward_ema[observed] += (1.0 - ema_decay) * mean_rewards[
            observed
        ]
        support_mask = self.adaptive_sg_base_dist > 0
        if self.adaptive_sg_mix_previous:
            prior_prob_used = previous_dist
            scores = reward_scale * self.adaptive_sg_reward_ema
            proposal = self.adaptive_sg_base_dist * np.exp(scores - np.max(scores))
        else:
            prior_prob_used = self.adaptive_sg_base_dist
            scores = reward_scale * self.adaptive_sg_reward_ema
            proposal = self.adaptive_sg_base_dist * np.exp(scores - np.max(scores))
        proposal[~support_mask] = 0.0
        proposal = proposal / proposal.sum()
        if min_prob > 0.0:
            proposal[support_mask] = np.maximum(proposal[support_mask], min_prob)
            proposal[~support_mask] = 0.0
            proposal = proposal / proposal.sum()
        if self.adaptive_sg_mix_previous:
            proposal = prior_mix * previous_dist + (1.0 - prior_mix) * proposal
        else:
            proposal = (
                prior_mix * self.adaptive_sg_base_dist + (1.0 - prior_mix) * proposal
            )
        self.adaptive_sg_dist = proposal / proposal.sum()
        self._save_adaptive_sg_snapshot(
            self.step,
            counts,
            mean_rewards,
            prior_prob_used.copy(),
        )
        top_index = int(np.argmax(self.adaptive_sg_dist))
        logging.info(
            "Adaptive SG policy updated: top_sg=%d prob=%.4f observed_sgs=%d",
            top_index + 1,
            float(self.adaptive_sg_dist[top_index]),
            int(observed.sum()),
        )

    def _save_best_checkpoint(self, reward_mean: float) -> None:
        if not np.isfinite(reward_mean) or reward_mean <= self.best_reward_mean:
            return
        self.best_reward_mean = float(reward_mean)
        self.best_reward_step = int(self.step)
        ckpt_dir = os.path.join(self.models_dir, "best_reward")
        self.model_suite.save_model(self.agent, ckpt_dir)
        meta_path = os.path.join(self.models_dir, "best_reward.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": self.best_reward_step,
                    "reward_mean": self.best_reward_mean,
                    "checkpoint_dir": ckpt_dir,
                },
                f,
                indent=2,
            )
        logging.info(
            f"New best reward checkpoint: step={self.best_reward_step}, "
            f"reward_mean={self.best_reward_mean:.4f}, dir={ckpt_dir}"
        )

    def _compute_checkpoint_eval(
        self, prop_dict: Dict, rewards: np.ndarray
    ) -> Dict | None:
        if self.checkpoint_eval is None:
            return None
        prop_key = self.checkpoint_eval.get("prop_key")
        if prop_key is None or prop_key not in prop_dict:
            return None
        hit_low = float(self.checkpoint_eval.get("hit_low", 0.30))
        hit_high = float(self.checkpoint_eval.get("hit_high", 0.35))
        values = np.asarray(prop_dict[prop_key], dtype=float)
        if len(values) == 0:
            return None
        hit_mask = (values >= hit_low) & (values <= hit_high)
        return {
            "prop_key": str(prop_key),
            "hit_low": hit_low,
            "hit_high": hit_high,
            "hit_rate": float(hit_mask.mean()),
            "hit_count": int(hit_mask.sum()),
            "num_scored": int(len(values)),
            "reward_mean": float(np.mean(rewards)) if len(rewards) else float("nan"),
            "prop_mean": float(np.mean(values)),
            "prop_std": float(np.std(values)),
        }

    def _save_best_hit_checkpoint(self, checkpoint_eval: Dict) -> None:
        hit_rate = float(checkpoint_eval["hit_rate"])
        hit_count = int(checkpoint_eval["hit_count"])
        reward_mean = float(checkpoint_eval["reward_mean"])
        should_save = hit_rate > self.best_hit_rate or (
            np.isclose(hit_rate, self.best_hit_rate)
            and (
                hit_count > self.best_hit_count
                or (
                    hit_count == self.best_hit_count
                    and reward_mean > self.best_hit_reward_mean
                )
            )
        )
        if not should_save:
            return
        self.best_hit_rate = hit_rate
        self.best_hit_step = int(self.step)
        self.best_hit_count = hit_count
        self.best_hit_reward_mean = reward_mean
        ckpt_dir = os.path.join(self.models_dir, "best_hit")
        self.model_suite.save_model(self.agent, ckpt_dir)
        meta_path = os.path.join(self.models_dir, "best_hit.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": self.best_hit_step,
                    "checkpoint_dir": ckpt_dir,
                    **checkpoint_eval,
                },
                f,
                indent=2,
            )
        logging.info(
            "New best hit checkpoint: step=%d, prop=%s, hit_rate=%.4f, hit_count=%d, dir=%s",
            self.best_hit_step,
            checkpoint_eval["prop_key"],
            self.best_hit_rate,
            self.best_hit_count,
            ckpt_dir,
        )

    def sample_step(self):
        generate_kwargs = dict(self.sample_cfg)
        if self.adaptive_sg_enabled and self.adaptive_sg_dist is not None:
            generate_kwargs["sg_distribution"] = self.adaptive_sg_dist.copy()
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent,
            **generate_kwargs,
        )
        # Filter invalid samples
        sample_data, sample_struc = invalid_filter(
            sample_data,
            sample_struc,
            max_atoms_per_structure=OmegaConf.select(
                self.sample_cfg, "max_atoms_per_structure"
            ),
            min_volume_per_atom=OmegaConf.select(
                self.sample_cfg, "min_volume_per_atom"
            ),
            exclude_atomic_numbers=OmegaConf.select(
                self.sample_cfg, "exclude_atomic_numbers"
            ),
        )

        pre_relax_max_num = OmegaConf.select(self.sample_cfg, "pre_relax_max_num")
        if pre_relax_max_num is not None and len(sample_struc) > pre_relax_max_num:
            sample_data = sample_data[:pre_relax_max_num]
            sample_struc = sample_struc[:pre_relax_max_num]
            logging.info(
                "Pre-relax cap applied: limiting candidates to %d structures",
                pre_relax_max_num,
            )

        # save all generated valid structures
        valid_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.extxyz",
        )
        save_data_records(
            data_list=sample_data,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.pt",
        )

        # MLIP relaxation
        if self.sample_cfg.get("mlip_opt"):
            mlip_opt = self.sample_cfg.mlip_opt
            sample_struc, energies = mlip_opt(sample_struc, valid_xyz_path)
        else:
            energies = None

        # Release fragmented PyTorch GPU cache before MatterSim relaxation.
        # Without this, reserved-but-unallocated cache fragments accumulate across
        # RL steps and can prevent contiguous allocations even when total free > needed.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Filter bad samples by selected metrics
        if self.sample_cfg.get("filter"):
            filter = self.sample_cfg.filter
            stability_csv = os.path.join(
                self.sample_dir, f"step_{self.step:0>4d}_stability.csv"
            )
            sample_data, sample_struc, metrics = filter(
                sample_data,
                sample_struc,
                energies,
                save_path=stability_csv,
            )
            logging.info(f"Number of stability-filtered samples: {len(sample_struc)}")
        else:
            # metrics, _ = self.opt_eval(sample_struc, energies)
            metrics = {}

        if sample_struc and (
            self.sample_cfg.get("mlip_opt") or self.sample_cfg.get("filter")
        ):
            post_relax_mask = invalid_filter(
                sample_data,
                sample_struc,
                return_mask=True,
                max_atoms_per_structure=OmegaConf.select(
                    self.sample_cfg, "max_atoms_per_structure"
                ),
                min_volume_per_atom=OmegaConf.select(
                    self.sample_cfg, "min_volume_per_atom"
                ),
                exclude_atomic_numbers=OmegaConf.select(
                    self.sample_cfg, "exclude_atomic_numbers"
                ),
                label="Post-relax invalid filter",
            )
            post_relax_total = len(sample_struc)
            post_relax_kept = int(post_relax_mask.sum())
            if post_relax_kept < post_relax_total:
                sample_data = [
                    data for data, keep in zip(sample_data, post_relax_mask) if keep
                ]
                sample_struc = [
                    struc for struc, keep in zip(sample_struc, post_relax_mask) if keep
                ]
                logging.info(
                    "Post-relax invalid filter removed %d/%d samples before scoring",
                    post_relax_total - post_relax_kept,
                    post_relax_total,
                )
            metrics["post_relax_filter_kept_ratio"] = (
                float(post_relax_mask.mean())
                if post_relax_total > 0
                else float("nan")
            )
            metrics["post_relax_filter_total"] = post_relax_total
            metrics["post_relax_filter_kept"] = post_relax_kept

        logging.info(f"Number of filtered samples: {len(sample_struc)}")
        log_str = [f"{k}: {v:.6f}" for k, v in metrics.items()]
        logging.info(", ".join(log_str))

        # max sample size to score/reward
        if self.sample_cfg.get("max_num"):
            max_num = self.sample_cfg.max_num
            if len(sample_struc) > max_num:
                sample_data = sample_data[:max_num]
                sample_struc = sample_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.extxyz",
        )
        save_data_records(
            data_list=sample_data,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.pt",
        )

        return sample_data, sample_struc, eval_xyz_path, metrics

    def _compute_advantage(self, reward: torch.Tensor, baseline: float) -> torch.Tensor:
        mode = str(self.finetune_cfg.get("advantage_mode", "raw")).lower()
        if mode == "raw":
            return reward
        if mode == "centered":
            baseline_tensor = torch.as_tensor(
                baseline,
                device=reward.device,
                dtype=reward.dtype,
            )
            return reward - baseline_tensor
        raise ValueError(f"Unknown finetune advantage_mode: {mode}")

    def ft_step(self, data_list, rewards, baseline):
        if self.finetune_cfg.epochs <= 0:
            logging.info("FINETUNE SKIPPED: epochs <= 0")
            return
        if not all(
            hasattr(self.agent, method_name)
            for method_name in ("add_noise", "calc_sample_loss", "calc_kl_reg")
        ):
            logging.info("FINETUNE SKIPPED: model does not expose DiffCSP finetune API")
            return
        # Tensor Core acceleration for new GPUs (Ampere, Hopper, etc)
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg
        optimize_site_symm = bool(cfg.get("optimize_site_symm", True))
        setattr(self.agent, "rl_optimize_site_symm", optimize_site_symm)
        setattr(self.prior, "rl_optimize_site_symm", optimize_site_symm)
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=len(data_list),
        )

        # LR decay: lr * lr_decay^step (lr_decay=1.0 means no decay, backwards-compatible)
        effective_lr = cfg.lr * (cfg.get("lr_decay", 1.0) ** self.step)
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=effective_lr)
        logging.info(f"ft lr={effective_lr:.2e} (step={self.step})")

        # Sigma schedule: inverse-sqrt decay over sigma_decay_steps (inf = fixed)
        sigma_decay_steps = cfg.get("sigma_decay_steps", float("inf"))
        sigma = cfg.sigma / math.sqrt(1.0 + self.step / sigma_decay_steps)
        logging.info(f"sigma={sigma:.4f} (step={self.step})")
        logging.info(
            f"advantage_mode={str(cfg.get('advantage_mode', 'raw')).lower()} (baseline={baseline:.4f})"
        )

        accum_steps = cfg.accum_steps  # accumulation_steps

        for epoch in range(cfg.epochs):
            # logging.info(f"Epoch {epoch} starts:")
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0.0, 0.0, 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss, loss_diff, loss_kl = 0.0, 0.0, 0.0

                for t in range(cfg.timesteps):
                    noised_input = self.agent.add_noise(batch, t)
                    sample_loss, agent_pred = self.agent.calc_sample_loss(noised_input)
                    _, prior_pred = self.prior.calc_sample_loss(noised_input)
                    adv = self._compute_advantage(batch.reward, baseline)
                    _loss_diff = adv * sample_loss

                    kl_term = self.agent.calc_kl_reg(agent_pred, prior_pred, batch)
                    _loss_kl = kl_term * (1.1 - batch.reward)

                    _loss = (_loss_diff + _loss_kl * sigma).mean() / accum_steps
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                if (t + 1) % accum_steps != 0:
                    optimizer.step()

                loss_all += loss * batch.num_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            loss_dict = {
                "loss": loss_all / len(data_list),
                "loss_diff": loss_diff_all / len(data_list),
                "loss_kl": loss_kl_all / len(data_list),
            }
            log_str = [f"{k}: {v:.4f}" for k, v in loss_dict.items()]
            logging.info(f"Epoch {epoch}: " + ", ".join(log_str))

    def rl_step(self):
        logging.info(f"*****   LOOP {self.step} START   *****")
        start_time = time.time()

        logging.info("SAMPLE:")
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        if len(sample_struc) == 0:
            logging.warning(
                "No samples remain after filtering at step %d; skipping scoring and finetuning.",
                self.step,
            )
            log_dict = {}
            for prop_cfg in self.reward.prop_cfg:
                log_dict[f"{prop_cfg.name} mean"] = np.nan
                log_dict[f"{prop_cfg.name} std"] = np.nan
            log_dict.update({"reward mean": np.nan, "reward std": np.nan})
            log_dict.update(sample_metrics)
            log_dict.update(self._adaptive_sg_log_dict())
            metrics = (
                self.ltm.calc_metrics(self.reward.threshold)
                if len(self.ltm) > 0
                else (None, None)
            )
            self.ltm.save(os.path.join(self.sample_dir, "long_term_memory.csv"))
            log_dict.update(
                {
                    "crystal_num": len(self.ltm),
                    "unique_comps": len(self.ltm.unique_comps),
                    "burden": metrics[0],
                    "div_ratio": metrics[1],
                    "cost": self.cost,
                }
            )
            if self.logger is not None:
                self.logger.log(log_dict, step=self.step)
            log_dict["step"] = self.step
            self._append_metrics_csv(log_dict)
            end_time = time.time()
            total_time = (end_time - start_time) / 60
            logging.info(f"*****   LOOP {self.step} FINISH   *****")
            logging.info(f"Total time taken: {total_time:.2f} min.\n\n")
            return

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info("SCORE:")
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list,
            sample_struc,
            xyz_path,
            f"step_{self.step:0>4d}",
        )

        # penalize_unstable: structures tagged is_unstable by OptFilter (failed ONLY the
        # e_hull gate) are split off as a hard-negative stream. The rest of the loop runs
        # on the STABLE survivors exactly as before, so bookkeeping/logging stays directly
        # comparable to a penalty-off run. When the flag is off, is_unstable is all-False
        # => neg_* are empty and behavior is identical to legacy.
        is_unstable = np.array(
            [bool(s.properties.get("is_unstable", False)) for s in sample_struc],
            dtype=bool,
        )
        neg_data = [d for d, u in zip(sample_list, is_unstable) if u]
        n_unstable = int(is_unstable.sum())
        if n_unstable > 0:
            keep = ~is_unstable
            sample_list = [d for d, k in zip(sample_list, keep) if k]
            sample_struc = [s for s, k in zip(sample_struc, keep) if k]
            rewards = rewards[keep]
            prop_dict = {k: v[keep] for k, v in prop_dict.items()}
            logging.info(
                f"penalize_unstable: split off {n_unstable} unstable structures as "
                f"hard negatives; {len(sample_struc)} stable survivors remain."
            )
            if len(sample_struc) == 0:
                logging.warning(
                    "No STABLE survivors at step %d; skipping finetuning this step.",
                    self.step,
                )
                end_time = time.time()
                logging.info(f"*****   LOOP {self.step} FINISH   *****")
                logging.info(
                    f"Total time taken: {(end_time - start_time) / 60:.2f} min.\n\n"
                )
                return

        log_dict = {f"{k} mean": v.mean() for k, v in prop_dict.items()}
        log_dict.update({f"{k} std": v.std() for k, v in prop_dict.items()})
        log_dict.update({"reward mean": rewards.mean(), "reward std": rewards.std()})
        log_dict["n_unstable"] = n_unstable
        log_dict.update(sample_metrics)
        self._update_adaptive_sg_policy(sample_struc, rewards)
        log_dict.update(self._adaptive_sg_log_dict())

        self._save_best_checkpoint(float(rewards.mean()))
        checkpoint_eval = self._compute_checkpoint_eval(prop_dict, rewards)
        if checkpoint_eval is not None:
            eval_label = (
                f"{checkpoint_eval['prop_key']}_hit_"
                f"{checkpoint_eval['hit_low']:.2f}_{checkpoint_eval['hit_high']:.2f}"
            )
            log_dict[f"{eval_label}_rate"] = checkpoint_eval["hit_rate"]
            log_dict[f"{eval_label}_count"] = checkpoint_eval["hit_count"]
            self._save_best_hit_checkpoint(checkpoint_eval)

        # long-term memory
        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, "long_term_memory.csv"))
        logging.info(
            f"{len(self.ltm)} crystals generated so far, "
            + f"{len(self.ltm.unique_comps)} unique components."
            + f"  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}."
        )
        log_dict.update(
            {
                "crystal_num": len(self.ltm),
                "unique_comps": len(self.ltm.unique_comps),
                "burden": metrics[0],
                "div_ratio": metrics[1],
                "cost": self.cost,
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        log_dict["step"] = self.step
        self._append_metrics_csv(log_dict)

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f"Diversity filter: tol_n={tol_n}, buff_n={buff_n}")

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()
            ft_data = sample_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(sample_topk, strucs_topk, reward_topk)
            logging.info(f"replay buffer size={len(self.replay)}")
            # print(f'replay rewards={reward_replay}')
            logging.info(
                f"buffer reward mean={self.replay.buffer['reward'].values.mean()}"
            )
            # print(f'buffer rewards={replay.buffer["reward"].values}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # hard negatives: append a capped set of unstable structures at the floor reward
        # so they enter finetuning with a negative advantage (needs a central baseline,
        # below). n_hard_negatives=0 (default) keeps this inert => legacy behavior.
        n_hard = int(self.finetune_cfg.get("n_hard_negatives", 0))
        if n_hard > 0 and len(neg_data) > 0:
            unstable_floor = float(self.finetune_cfg.get("unstable_floor", 0.0))
            # cap negatives relative to the positive batch so they can't dominate the
            # update (job 54523927 collapsed: 11 negatives vs 5 positives -> KL blow-up).
            # hard_negative_ratio<=0 disables the cap (legacy: take up to n_hard).
            ratio = float(self.finetune_cfg.get("hard_negative_ratio", 0.5))
            cap = max(1, int(ratio * len(ft_data))) if ratio > 0 else len(neg_data)
            take = min(n_hard, len(neg_data), cap)
            neg_sel = neg_data[:take]
            neg_reward = np.full(take, unstable_floor, dtype=float)
            ft_data = ft_data + neg_sel
            ft_reward = np.concatenate((ft_reward, neg_reward))
            log_dict["n_hard_negatives"] = take
            logging.info(
                f"Added {take}/{len(neg_data)} unstable hard negatives "
                f"(floor={unstable_floor}, cap={cap} @ ratio {ratio}, "
                f"{len(ft_data) - take} positives) to the finetune batch"
            )

        # finetuning
        logging.info("FINETUNE:")
        baseline_mode = str(self.finetune_cfg.get("advantage_baseline", "min")).lower()
        if baseline_mode == "mean":
            baseline = float(ft_reward.mean())
        elif baseline_mode == "median":
            baseline = float(np.median(ft_reward))
        else:  # "min" — legacy behavior
            baseline = self.ltm.get_baseline(self.step)
            baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f"*****   LOOP {self.step} FINISH   *****")
        logging.info(f"Total time taken: {total_time:.2f} min.\n\n")

    def _append_metrics_csv(self, row: dict):
        path = os.path.join(self.save_dir, "metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def run_rl(self):
        logging.info("*****   RL START   *****")
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f"loop_{step:0>4d}")
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, "final")
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info("*****   RL END   *****")
        end_time = time.time()
        logging.info("Total time taken: {} s.".format(int(end_time - start_time)))


class SPARC(MatInvent):
    pass
