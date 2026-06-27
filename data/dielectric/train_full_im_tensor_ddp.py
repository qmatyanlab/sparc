import os
import sys

# --------- path setup ----------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# --------- imports ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch_geometric as tg
import torch_scatter
from typing import Dict, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import pandas as pd

import e3nn.o3 as o3
from utils.utils_data import (
    load_data,
    train_valid_test_split,
    save_or_load_onehot,
    build_data,
    plot_spherical_harmonics_comparison,
    plot_cartesian_tensor_comparison,
    compute_aniso_mae,
)
from utils.utils_model_full_tensor import Network, train
from utils.normalize_cart import (
    cart_to_sph,
    sph_to_cart,
)

import wandb

# --------- global settings ----------
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
torch.manual_seed(3407)

fontsize = 12
textsize = 12
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize
plt.rcParams['text.usetex'] = False


# colors for datasets
palette = ['#2876B2', '#F39957', '#67C7C2', '#C86646']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
# --------- DDP helpers ----------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # change if needed
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# --------- shared helpers ----------
def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)


def load_cache_only():
    """
    Load cached preprocessed data (no dataloaders, so we can build per-rank loaders).
    This assumes you already ran get_dataloaders_and_metadata() once.
    """
    cache = torch.load("../dataset/cached_dielectric_preprocessed_data.pt", map_location="cpu")

    df = cache["df"]
    idx_train = cache["idx_train"]
    idx_valid = cache["idx_valid"]
    idx_test = cache["idx_test"]
    out_dim = cache["out_dim"]
    r_max = cache["r_max"]
    scale_0e = cache["scale_0e"]
    scale_2e = cache["scale_2e"]

    return df, idx_train, idx_valid, idx_test, out_dim, r_max, scale_0e, scale_2e


class NetWrapper(Network):
    """
    Same as in your original single-GPU script, but used inside DDP.
    """
    def __init__(self, in_dim, em_dim, **kwargs):
        # override the `reduce_output` keyword to instead perform an average over atom contributions
        self.pool = False
        if kwargs['reduce_output'] is True:
            kwargs['reduce_output'] = False
            self.pool = True

        super().__init__(**kwargs)

        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))

        output = super().forward(data)

        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output


class LearnableUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_0e = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_2e = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_0e, loss_2e):
        precision_0e = torch.exp(-2 * self.log_sigma_0e)
        precision_2e = torch.exp(-2 * self.log_sigma_2e)
        weighted = 0.5 * (precision_0e * loss_0e + precision_2e * loss_2e)
        reg = self.log_sigma_0e + self.log_sigma_2e
        return weighted + reg


# --------- plotting helpers (same as original script) ----------
inds_diag = [(0, 0), (1, 1), (2, 2)]
inds_off = [(0, 1), (0, 2), (1, 2)]


def compute_symmetric_errors(pred, true):
    diffs = []
    for i, j in inds_diag + inds_off:
        diff = pred[:, i, j] - true[:, i, j]  # shape: (N,)
        diffs.append(diff)
    diffs = torch.stack(diffs, dim=0)  # shape: (6, N)
    mse = torch.mean(diffs ** 2, dim=0)  # (N,)
    mae = torch.mean(torch.abs(diffs), dim=0)  # (N,)
    return mse, mae

def plot_sph_scatter_single(
    sph_true_tensor: torch.Tensor,
    sph_pred_tensor: torch.Tensor,
    indices,
    title="Spherical basis",
    save_path=None,
):
    sph_true = sph_true_tensor.detach().cpu().numpy()
    sph_pred = sph_pred_tensor.detach().cpu().numpy()

    comp_labels = ["0e", "2e(m=-2)", "2e(m=-1)", "2e(m=0)", "2e(m=1)", "2e(m=2)"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for k, ax in enumerate(axes.ravel()):
        true_vals = sph_true[indices, k]
        pred_vals = sph_pred[indices, k]

        r2_k = r2_score(true_vals, pred_vals)

        ax.scatter(true_vals, pred_vals, s=8, alpha=0.5, color=palette[0])

        ax.set_title(f"{comp_labels[k]} (R²={r2_k:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    plt.close(fig)
def plot_cart_scatter_single(
    cart_true_tensor: torch.Tensor,
    cart_pred_tensor: torch.Tensor,
    indices,
    title="Cartesian basis",
    save_path=None,
):
    cart_true = cart_true_tensor.detach().cpu().numpy()
    cart_pred = cart_pred_tensor.detach().cpu().numpy()

    inds = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    labels = ["xx", "yy", "zz", "xy", "xz", "yz"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for k, ((i, j), name) in enumerate(zip(inds, labels)):
        ax = axes.ravel()[k]

        true_vals = cart_true[indices, i, j]
        pred_vals = cart_pred[indices, i, j]

        r2_k = r2_score(true_vals, pred_vals)

        ax.scatter(true_vals, pred_vals, s=8, alpha=0.5, color=palette[0])

        ax.set_title(f"{name} (R²={r2_k:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    plt.close(fig)
# --------- main worker ----------
def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.set_default_dtype(default_dtype)

    # ---- load cached data & build per-rank loaders ----
    df, idx_train, idx_valid, idx_test, out_dim, r_max, scale_0e, scale_2e = load_cache_only()

    # ndarray of Data objects
    train_dataset = df.iloc[idx_train]['data'].values
    valid_dataset = df.iloc[idx_valid]['data'].values

    batch_size = 16
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader_train = tg.loader.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size
    )
    dataloader_valid = tg.loader.DataLoader(
        valid_dataset, batch_size=batch_size
    )
    n_train = get_neighbors(df, idx_train)
    if rank == 0:
        print(f"Number of nodes (atoms) per training graph: mean={n_train.mean():.1f}")
    # ---- model / loss / opt / scheduler ----
    em_dim = 64
    use_batch_norm = False
    dropout_prob = 0.4
    lr = 1e-2
    lmax = 2
    layers = 2
    mul = 32

    model = NetWrapper(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{out_dim}x0e + {out_dim}x2e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=layers,
        mul=mul,
        lmax=lmax,
        max_radius=r_max,
        num_neighbors=n_train.mean(),
        reduce_output=True,
        dropout_prob=dropout_prob,
        use_batch_norm=use_batch_norm,
    )
    model.to(device)

    # wrap in DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    loss_balancer = LearnableUncertaintyLoss().to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=lr,
        weight_decay=0.05,
    )

    max_iter = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=1, eta_min=0
    )

    loss_fn = torch.nn.MSELoss()
    loss_fn_mae = torch.nn.L1Loss()
    loss_fn_eval = torch.nn.MSELoss()
    loss_fn_mae_eval = torch.nn.L1Loss()

    run_name = (
        f'indep_e3_dielectric_DDP_Lmax{lmax}_Lr{lr}_bs{batch_size}_'
        f'em{em_dim}_layers{layers}_mul{mul}'
    )

    # ---- wandb only on rank 0 ----
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        wandb.init(
            project="Dielectric Tensor Predict",
            name=run_name,
            config={
                "max_iter": max_iter,
                "lr": opt.param_groups[0]["lr"],
                "use_weighting": True,
                "r_max": r_max,
                "batch_size": batch_size,
                "dropout_prob": dropout_prob,
                "normalization": True,
                "batch_norm": use_batch_norm,
                "scheduler": type(scheduler).__name__,
                "loss_function": type(loss_fn).__name__,
            },
        )
        # unwrap for watch
        wandb.watch(model.module, log='all', log_freq=100)

    # ---- training ----
    train(
        model,
        opt,
        dataloader_train,
        dataloader_valid,
        loss_fn,
        loss_fn_mae,
        loss_fn_eval,
        loss_fn_mae_eval,
        run_name,
        max_iter=max_iter,
        scheduler=scheduler,
        device=device,
        alpha=1.0,
        beta=1.0,
        loss_balancer=loss_balancer,
    )

    # ---- evaluation & plots only on rank 0 ----
    if rank == 0:
        # load history for loss plot
        history = torch.load(f'../model/{run_name}.torch', map_location=device)['history']
        steps = [d['step'] + 1 for d in history]
        loss_train = [d['train']['loss'] for d in history]
        loss_valid = [d['valid']['loss'] for d in history]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(steps, loss_train, 'o-', label="Training", color='C0', markersize=3)
        ax.plot(steps, loss_valid, 'o-', label="Validation", color='C3', markersize=3)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(frameon=False)
        plt.tight_layout()
        save_png_dir = "../pngs"
        os.makedirs(save_png_dir, exist_ok=True)
        save_path = os.path.join(save_png_dir, run_name + '_loss.png')
        fig.savefig(save_path, dpi=300)
        wandb.log({"Loss Plot": wandb.Image(save_path)})

        # ---- prediction on all data ----
        checkpoint = torch.load(f'../model/{run_name}_best.torch', map_location=device)
        state_dict = checkpoint['state']

        eval_model = model.module if isinstance(model, DDP) else model
        eval_model.load_state_dict(state_dict)
        eval_model.pool = True
        eval_model.to(device)
        eval_model.eval()

        # full dataloader (no DDP) on rank 0
        dataloader_full = tg.loader.DataLoader(df['data'].values, batch_size=64)

        df['mse_sph'] = 0.0
        df['y_pred_sph'] = None

        predictions = []
        i0 = 0

        scale_0e_t = torch.tensor(scale_0e, dtype=default_dtype, device=device)
        scale_2e_t = torch.tensor(scale_2e, dtype=default_dtype, device=device)

        with torch.no_grad():
            for i, d in tqdm(
                enumerate(dataloader_full),
                total=len(dataloader_full),
                bar_format=bar_format,
            ):
                d = d.to(device)
                output = eval_model(d)

                irreps_0e = eval_model.irreps_out.count(o3.Irrep("0e"))
                irreps_2e = eval_model.irreps_out.count(o3.Irrep("2e")) * 5
                out_dim_eval = irreps_0e  # =1 here

                # denorm
                output_0e = output[:, :irreps_0e] * scale_0e_t
                output_2e = output[:, irreps_0e:irreps_0e + irreps_2e] * scale_2e_t

                y_0e = d.y[:, 0:1] * scale_0e_t
                y_2e = d.y[:, 1:] * scale_2e_t

                loss_0e = F.mse_loss(output_0e, y_0e)
                loss_2e = F.mse_loss(output_2e, y_2e)
                loss = loss_0e + loss_2e

                combined_output = torch.cat([output_0e, output_2e], dim=-1)
                predictions.append(combined_output.cpu())

                for batch_idx in range(d.y.shape[0]):
                    df.at[i0 + batch_idx, 'y_pred_sph'] = [combined_output[batch_idx].cpu().numpy()]
                    df.at[i0 + batch_idx, 'mse_sph'] = loss.cpu().numpy()

                i0 += d.y.shape[0]

        # unwrap list-of-[vec] to list-of-vec
        df['y_pred_sph'] = df['y_pred_sph'].map(lambda x: x[0])

        # sph -> cart
        sph_pred = torch.tensor(np.stack(df['y_pred_sph'].values), device=device)
        cart_pred = sph_to_cart(sph_pred)
        df['y_pred_cart'] = list(cart_pred.detach().cpu().numpy())

        cart_true = np.stack(df['dielectric_tensor'].values)
        cart_pred_np = np.stack(df['y_pred_cart'].values)

        cart_true_tensor = torch.tensor(cart_true, dtype=default_dtype)
        cart_pred_tensor = torch.tensor(cart_pred_np, dtype=default_dtype)

        mse_torch, mae_cart = compute_symmetric_errors(cart_pred_tensor, cart_true_tensor)
        mse_torch = mse_torch.cpu().numpy()
        mae_cart = mae_cart.cpu().numpy()

        sph_true_tensor = torch.tensor(
            np.stack(df['sph_coefs_true'].values), dtype=default_dtype
        )
        sph_pred_tensor = torch.tensor(
            np.stack(df['y_pred_sph'].values), dtype=default_dtype
        )

        df['mse_cart'] = mse_torch
        df['mae_cart'] = mae_cart

        mae_sph = torch.mean(
            torch.abs(sph_pred_tensor - sph_true_tensor), dim=(0, 1)
        ).cpu().numpy()
        df['mae_sph'] = mae_sph

        mae_sph_mean = df['mae_sph'].mean()
        mae_sph_std = df['mae_sph'].std()
        mae_cart_mean = df['mae_cart'].mean()
        mae_cart_std = df['mae_cart'].std()

        wandb.log({
            "Mean MAE in cart": mae_cart_mean,
            "Std MAE in cart": mae_cart_std,
            "Mean MAE in sph": mae_sph_mean,
            "Std MAE in sph": mae_sph_std,
        })

        # scatter plots
        sph_true_tensor_gpu = sph_true_tensor.to(device)
        sph_pred_tensor_gpu = sph_pred_tensor.to(device)
        cart_true_tensor_gpu = cart_true_tensor.to(device)
        cart_pred_tensor_gpu = cart_pred_tensor.to(device)

        save_png_dir = "../pngs"
        os.makedirs(save_png_dir, exist_ok=True)

        # Train
        plot_sph_scatter_single(
            sph_true_tensor_gpu,
            sph_pred_tensor_gpu,
            idx_train,
            title="Train set — Spherical basis",
            save_path=os.path.join(save_png_dir, run_name + "_train_sph.png")
        )

        plot_cart_scatter_single(
            cart_true_tensor_gpu,
            cart_pred_tensor_gpu,
            idx_train,
            title="Train set — Cartesian basis",
            save_path=os.path.join(save_png_dir, run_name + "_train_cart.png")
        )

        # Validation
        plot_sph_scatter_single(
            sph_true_tensor_gpu,
            sph_pred_tensor_gpu,
            idx_valid,
            title="Validation set — Spherical basis",
            save_path=os.path.join(save_png_dir, run_name + "_valid_sph.png")
        )

        plot_cart_scatter_single(
            cart_true_tensor_gpu,
            cart_pred_tensor_gpu,
            idx_valid,
            title="Validation set — Cartesian basis",
            save_path=os.path.join(save_png_dir, run_name + "_valid_cart.png")
        )

        # Test
        plot_sph_scatter_single(
            sph_true_tensor_gpu,
            sph_pred_tensor_gpu,
            idx_test,
            title="AnisoNet dataset Test set — Spherical basis",
            save_path=os.path.join(save_png_dir, run_name + "_test_sph.png")
        )

        plot_cart_scatter_single(
            cart_true_tensor_gpu,
            cart_pred_tensor_gpu,
            idx_test,
            title="AnisoNet dataset Test set — Cartesian basis",
            save_path=os.path.join(save_png_dir, run_name + "_test_cart.png")
        )
        wandb.finish()

    cleanup()


def main():
    world_size = 4  # adjust to number of GPUs
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
