#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


OmegaConf.register_new_resolver("calc", eval, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hit-low", type=float, default=0.30)
    parser.add_argument("--hit-high", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-filter", action="store_true")
    parser.add_argument("--eta-only", action="store_true")
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_run_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return OmegaConf.load(require_path(run_dir / "hparams.yaml", "run config"))


def discover_checkpoint_dirs(run_dir: Path) -> list[Path]:
    models_dir = require_path(run_dir / "models", "models directory")
    dirs = []
    for path in sorted(models_dir.iterdir()):
        if not path.is_dir():
            continue
        if any(path.glob("*.ckpt")):
            dirs.append(path)
    if not dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {models_dir}")
    return dirs


def resolve_checkpoint_dirs(run_dir: Path, names: list[str] | None) -> list[Path]:
    if not names:
        return discover_checkpoint_dirs(run_dir)
    out = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = run_dir / "models" / name
        out.append(require_path(path, f"checkpoint directory '{name}'"))
    return out


def maybe_load_best_reward_meta(run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / "models" / "best_reward.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_hydra_runtime_resolver(project_root: Path) -> None:
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: str(project_root) if str(key) == "runtime.cwd" else None,
        replace=True,
    )


def build_eval_pipeline(
    run_cfg: Any,
    checkpoint_dir: Path,
    save_dir: Path,
    args: argparse.Namespace,
):
    from pipeline.mat_invent import MatInvent

    ensure_hydra_runtime_resolver(Path.cwd())

    model_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg.model, resolve=True))
    model_cfg.model_path = str(checkpoint_dir)
    if args.generation_batch_size is not None:
        model_cfg.sample_cfg.generation_batch_size = args.generation_batch_size
    if args.batch_size is not None:
        model_cfg.sample_cfg.batch_size = args.batch_size
    if args.num_batches is not None:
        model_cfg.sample_cfg.num_batches = args.num_batches

    reward_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg.reward, resolve=True))
    if args.eta_only:
        reward_cfg.prop_cfg = [
            prop for prop in reward_cfg.prop_cfg if str(prop.name) != "band_gap"
        ]
    sample_cfg = OmegaConf.create(
        OmegaConf.to_container(run_cfg.sample_cfg, resolve=True)
    )
    pipeline_cfg = OmegaConf.create(
        OmegaConf.to_container(run_cfg.pipeline, resolve=True)
    )
    finetune_cfg = OmegaConf.create(
        OmegaConf.to_container(pipeline_cfg.finetune_cfg, resolve=True)
    )

    if args.eval_size is not None:
        finetune_cfg.batch_size = args.eval_size
        if args.batch_size is None:
            model_cfg.sample_cfg.batch_size = args.eval_size * 12
        sample_cfg.max_num = args.eval_size

    if args.disable_filter:
        sample_cfg.filter = None

    device = args.device or str(run_cfg.device)
    model_suite = hydra.utils.instantiate(model_cfg)
    reward = hydra.utils.instantiate(reward_cfg)
    df_args = None
    if OmegaConf.select(pipeline_cfg, "df_args") is not None:
        df_args = OmegaConf.to_container(pipeline_cfg.df_args, resolve=True)

    return MatInvent(
        rl_epoch=1,
        model_suite=model_suite,
        reward=reward,
        sample_cfg=cast(DictConfig, sample_cfg),
        finetune_cfg=cast(DictConfig, finetune_cfg),
        topk_ratio=float(pipeline_cfg.topk_ratio),
        save_dir=str(save_dir),
        save_freq=int(pipeline_cfg.save_freq),
        device=device,
        logger=cast(Any, None),
        replay=False,
        replay_args=cast(Any, None),
        div_filter=bool(pipeline_cfg.div_filter),
        df_args=cast(Any, df_args),
    )


def evaluate_checkpoint(
    run_cfg: Any,
    checkpoint_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
):
    ckpt_name = checkpoint_dir.name
    ckpt_out = output_dir / ckpt_name
    ckpt_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        pipeline = build_eval_pipeline(
            run_cfg, checkpoint_dir, ckpt_out / f"repeat_{repeat:02d}", args
        )
        pipeline.step = repeat
        sample_data, sample_struc, xyz_path, sample_metrics = pipeline.sample_step()
        xyz_path = str(xyz_path)
        sample_data, sample_struc, rewards, prop_dict = pipeline.reward_step(
            sample_data,
            sample_struc,
            xyz_path,
            label=f"eval_{ckpt_name}_{repeat:02d}",
        )

        # eta prop name varies by reward (tsenn_slme_eta vs tsenn_slme_optimate_eta)
        eta_key = next(
            (
                k
                for k in ("tsenn_slme_optimate_eta", "tsenn_slme_eta")
                if k in prop_dict
            ),
            next((k for k in prop_dict if k.endswith("_eta")), None),
        )
        if eta_key is None:
            raise KeyError(f"no *_eta key in prop_dict; keys={list(prop_dict)}")
        eta = np.asarray(prop_dict[eta_key], dtype=float)
        band_gap = np.asarray(prop_dict.get("band_gap", []), dtype=float)
        hit_mask = (eta >= args.hit_low) & (eta <= args.hit_high)
        rows.append(
            {
                "checkpoint": ckpt_name,
                "repeat": repeat,
                "seed": seed,
                "num_scored": int(len(rewards)),
                "reward_mean": float(np.mean(rewards))
                if len(rewards)
                else float("nan"),
                "reward_std": float(np.std(rewards)) if len(rewards) else float("nan"),
                "eta_mean": float(np.mean(eta)) if len(eta) else float("nan"),
                "eta_std": float(np.std(eta)) if len(eta) else float("nan"),
                "eta_hit_rate": float(np.mean(hit_mask))
                if len(hit_mask)
                else float("nan"),
                "eta_hit_count": int(hit_mask.sum()),
                "eta_p90": float(np.percentile(eta, 90)) if len(eta) else float("nan"),
                "band_gap_mean": float(np.mean(band_gap))
                if len(band_gap)
                else float("nan"),
                "frac_stable_structures": float(
                    sample_metrics.get("frac_stable_structures", np.nan)
                ),
                "filter_kept_ratio": float(
                    sample_metrics.get("filter_kept_ratio", np.nan)
                ),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "checkpoint_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), []).append(row)

    rank_rows = []
    for checkpoint, ckpt_rows in grouped.items():
        hit_rates = np.array([float(r["eta_hit_rate"]) for r in ckpt_rows], dtype=float)
        reward_means = np.array(
            [float(r["reward_mean"]) for r in ckpt_rows], dtype=float
        )
        eta_means = np.array([float(r["eta_mean"]) for r in ckpt_rows], dtype=float)
        rank_rows.append(
            {
                "checkpoint": checkpoint,
                "eta_hit_rate_mean": float(np.nanmean(hit_rates)),
                "eta_hit_rate_std": float(np.nanstd(hit_rates)),
                "reward_mean_mean": float(np.nanmean(reward_means)),
                "eta_mean_mean": float(np.nanmean(eta_means)),
                "repeats": len(ckpt_rows),
            }
        )
    rank_rows.sort(
        key=lambda row: (row["eta_hit_rate_mean"], row["reward_mean_mean"]),
        reverse=True,
    )

    rank_path = output_dir / "checkpoint_ranking.json"
    with rank_path.open("w", encoding="utf-8") as handle:
        json.dump(rank_rows, handle, indent=2)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_cfg = load_run_config(run_dir)
    checkpoint_dirs = resolve_checkpoint_dirs(run_dir, args.checkpoints)
    output_dir = (
        args.output_dir or (run_dir / "deliverables" / "checkpoint_scores")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = maybe_load_best_reward_meta(run_dir)
    if meta is not None:
        with (output_dir / "best_reward_meta.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(meta, handle, indent=2)

    all_rows: list[dict[str, Any]] = []
    for checkpoint_dir in checkpoint_dirs:
        all_rows.extend(evaluate_checkpoint(run_cfg, checkpoint_dir, output_dir, args))
    if not all_rows:
        raise RuntimeError("No checkpoint evaluations completed")
    write_summary(all_rows, output_dir)
    print(f"Saved checkpoint evaluation to {output_dir}")


if __name__ == "__main__":
    main()
