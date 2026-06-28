from __future__ import annotations

import functools
import importlib
import math
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

from models.symmcd.common.data_utils import (
    lattice_ks_to_matrix_torch,
    lattice_params_to_matrix_torch,
    mask_ks,
    sg_to_ks_mask,
)
from models.symmcd.pl_modules.diff_utils import d_log_p_wrapped_normal
from models.symmcd.pl_modules.model import build_mlp


MAX_ATOMIC_NUM = 94
NUM_SPACEGROUPS = 231
SITE_SYMM_AXES = 15
SITE_SYMM_PGS = 13
SITE_SYMM_DIM = SITE_SYMM_AXES * SITE_SYMM_PGS
SG_CONDITION_DIM = 397


def _spacegroup_group(spacegroup: int):
    Group = importlib.import_module("pyxtal.symmetry").Group
    return Group(spacegroup)


def _search_closest_wp(spacegroup: int, wp, op, frac_coord):
    search_cloest_wp = importlib.import_module("pyxtal.symmetry").search_cloest_wp
    return search_cloest_wp(_spacegroup_group(spacegroup), wp, op, frac_coord)


@functools.lru_cache(maxsize=230)
def _wp_to_site_symm(spacegroup: int):
    wp_map = {}
    for wp in _spacegroup_group(spacegroup).Wyckoff_positions:
        wp.get_site_symmetry()
        wp_map[wp] = wp.get_site_symmetry_object().to_one_hot()
    return wp_map


class DiscreteNoise(nn.Module):
    def __init__(
        self, atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a
    ):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.site_symm_prior_per_sg = site_symm_prior_per_sg
        self.atom_type_prior = atom_type_prior
        self.P_ss = P_ss
        self.P_a = P_a
        self.site_symm_pgs = SITE_SYMM_PGS
        self.site_symm_axes = SITE_SYMM_AXES
        self.max_atomic_num = MAX_ATOMIC_NUM
        self.ss_lengths = [self.site_symm_pgs] * self.site_symm_axes

    def ss_to_sections(self, ss):
        return [
            ss[..., i * self.site_symm_pgs : (i + 1) * self.site_symm_pgs]
            for i in range(self.site_symm_axes)
        ]

    def reshape_ss(self, ss):
        return ss.reshape(-1, SITE_SYMM_AXES, self.site_symm_pgs)

    def multiply_block_diagonal(self, Qs, d):
        outs = []
        idx = 0
        for Qi in Qs:
            ni = Qi.shape[-1]
            outs.append(d[..., idx : idx + ni] @ Qi)
            idx += ni
        return torch.cat(outs, -1)

    def q_t(self, P, t):
        alpha = self.beta_scheduler.alphas[t]
        num_classes = P.shape[-1]
        eye = torch.eye(num_classes, device=P.device).unsqueeze(0)
        return alpha.view(-1, 1, 1) * eye + (1 - alpha.view(-1, 1, 1)) * P

    def q_t_atom(self, t):
        return self.q_t(self.P_a, t)

    def q_t_ss(self, t, sgs):
        return [self.q_t(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def q_t_bar(self, P, t):
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        num_classes = P.shape[-1]
        eye = torch.eye(num_classes, device=P.device).unsqueeze(0)
        return alpha_bar.view(-1, 1, 1) * eye + (1 - alpha_bar.view(-1, 1, 1)) * P

    def q_t_bar_atom(self, t):
        return self.q_t_bar(self.P_a, t)

    def q_t_bar_ss(self, t, sgs):
        return [self.q_t_bar(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def apply_atom_noise(self, atom_type, t):
        return atom_type @ self.q_t_bar_atom(t)

    def apply_site_symm_noise(self, site_symm, t, sgs):
        return self.multiply_block_diagonal(self.q_t_bar_ss(t, sgs), site_symm)

    def sample_limit_dist(self, node_mask, sgs):
        bs, n_max = node_mask.shape
        atom_type_prior = self.P_a[0] if self.P_a.ndim == 2 else self.atom_type_prior
        a_limit = atom_type_prior.to(node_mask.device).expand(bs, n_max, -1)
        U_a = a_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
        U_a = F.one_hot(
            U_a, num_classes=a_limit.shape[-1]
        ).float() * node_mask.unsqueeze(-1)

        U_ss_list = []
        for i, ss_priors_i_per_sg in enumerate(self.site_symm_prior_per_sg):
            if self.P_ss[i].ndim == 3:
                ss_priors_i_per_sg = self.P_ss[i][:, 0, :]
            ss_priors_i = ss_priors_i_per_sg.to(node_mask.device)[sgs]
            ss_limit_i = ss_priors_i.unsqueeze(-2).expand(bs, n_max, -1)
            U_ss_i = ss_limit_i.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
            U_ss_i = F.one_hot(U_ss_i, num_classes=ss_limit_i.shape[-1]).float()
            U_ss_list.append(U_ss_i)
        return U_a, torch.cat(U_ss_list, dim=-1)

    def sample_discrete_features(self, prob_a, prob_ss, node_mask):
        bs, n = node_mask.shape
        prob_a = prob_a.clone()
        prob_ss = prob_ss.clone()
        prob_a[~node_mask] = 1 / prob_a.shape[-1]
        prob_ss_list = self.ss_to_sections(prob_ss)
        for i in range(SITE_SYMM_AXES):
            prob_ss_list[i][~node_mask] = 1 / prob_ss_list[i].shape[-1]
        prob_a_flat = prob_a.reshape(bs * n, -1)
        atom_t = prob_a_flat.multinomial(1).reshape(bs, n)
        atom_t = F.one_hot(atom_t, num_classes=prob_a_flat.shape[-1]).float()

        site_symm_t_list = []
        for prob_ss_i in prob_ss_list:
            prob_ss_flat = prob_ss_i.reshape(bs * n, -1)
            site_symm_t_i_cat = prob_ss_flat.multinomial(1).reshape(bs, n)
            site_symm_t_i = F.one_hot(
                site_symm_t_i_cat, num_classes=self.site_symm_pgs
            ).float()
            site_symm_t_list.append(site_symm_t_i)
        return atom_t, torch.cat(site_symm_t_list, -1)

    def p_s_and_t_given_0(self, z_t, Qt, Qsb, Qtb):
        Qt_T = Qt.transpose(-1, -2)
        left_term = (z_t @ Qt_T).unsqueeze(dim=2)
        right_term = Qsb.unsqueeze(1)
        numerator = left_term * right_term
        prod = (Qtb @ z_t.transpose(-1, -2)).transpose(-1, -2)
        denominator = prod.unsqueeze(-1)
        denominator[denominator == 0] = 1e-6
        return numerator / denominator

    def p_s_and_t_given_0_a(self, z_t_a, t, s):
        return self.p_s_and_t_given_0(
            z_t_a, self.q_t_atom(t), self.q_t_bar_atom(s), self.q_t_bar_atom(t)
        )

    def p_s_and_t_given_0_ss(self, z_t_ss, t, s, sgs):
        Qtb_ss = self.q_t_bar_ss(t, sgs)
        Qsb_ss = self.q_t_bar_ss(s, sgs)
        Qt_ss = self.q_t_ss(t, sgs)
        return [
            self.p_s_and_t_given_0(z_t_ss[i], Qt_ss[i], Qsb_ss[i], Qtb_ss[i])
            for i in range(len(self.P_ss))
        ]

    def sample_zs_from_zt_and_pred(
        self, z_t_a, z_t_ss, pred_a, pred_ss, t, s, node_mask, sgs
    ):
        z_t_ss_split = self.ss_to_sections(z_t_ss)
        p_atom = self.p_s_and_t_given_0_a(z_t_a, t, s)
        p_site = self.p_s_and_t_given_0_ss(z_t_ss_split, t, s, sgs)
        weighted_a = pred_a.unsqueeze(-1) * p_atom
        prob_a = weighted_a.sum(dim=2)
        prob_a[torch.sum(prob_a, dim=-1) == 0] = 1e-5
        prob_a = prob_a / torch.sum(prob_a, dim=-1, keepdim=True)

        pred_ss_split = self.ss_to_sections(pred_ss)
        prob_ss_list = []
        for pred_ss_i, p_site_i in zip(pred_ss_split, p_site):
            weighted_ss = pred_ss_i.unsqueeze(-1) * p_site_i
            prob_ss = weighted_ss.sum(dim=2)
            prob_ss[torch.sum(prob_ss, dim=-1) == 0] = 1e-5
            prob_ss = prob_ss / torch.sum(prob_ss, dim=-1, keepdim=True)
            prob_ss_list.append(prob_ss)
        return self.sample_discrete_features(
            prob_a, torch.cat(prob_ss_list, -1), node_mask
        )

    def discrete_loss(self, sample_a, sample_ss, pred_a, pred_ss):
        loss_a = F.nll_loss(torch.log(pred_a + 1e-20), sample_a)
        pred_ss_split = self.ss_to_sections(pred_ss)
        losses = [
            F.nll_loss(torch.log(pred_ss_i + 1e-20), sample_ss[..., i])
            for i, pred_ss_i in enumerate(pred_ss_split)
        ]
        return loss_a, torch.stack(losses).mean()


class DiscreteNoiseMarginal(DiscreteNoise):
    def __init__(self, atom_marginals_path, ss_marginals_path, beta_scheduler):
        if atom_marginals_path and Path(atom_marginals_path).exists():
            atom_type_prior = torch.load(atom_marginals_path)
        else:
            atom_type_prior = torch.full((MAX_ATOMIC_NUM,), 1.0 / MAX_ATOMIC_NUM)
        if ss_marginals_path and Path(ss_marginals_path).exists():
            site_symm_prior_per_sg = torch.load(ss_marginals_path)
        else:
            site_symm_prior_per_sg = [
                torch.full((NUM_SPACEGROUPS, SITE_SYMM_PGS), 1.0 / SITE_SYMM_PGS)
                for _ in range(SITE_SYMM_AXES)
            ]
        P_ss = nn.ParameterList(
            [
                nn.Parameter(
                    site_symm_prior_per_sg[i]
                    .unsqueeze(-2)
                    .expand(NUM_SPACEGROUPS, SITE_SYMM_PGS, SITE_SYMM_PGS)
                    .clone(),
                    requires_grad=False,
                )
                for i in range(SITE_SYMM_AXES)
            ]
        )
        P_a = nn.Parameter(
            atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM, -1).clone(),
            requires_grad=False,
        )
        super().__init__(
            atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a
        )


def find_num_atoms(dummy_ind, total_num_atoms):
    actual_num_atoms = []
    atoms = 0
    for num in total_num_atoms:
        actual_num_atoms.append(torch.sum(dummy_ind[atoms : atoms + num] == 0).item())
        atoms += num
    return torch.tensor(actual_num_atoms)


def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
    spacegroup = (
        int(spacegroup.item()) if torch.is_tensor(spacegroup) else int(spacegroup)
    )
    site_symm_axis = site_symm.reshape(-1, SITE_SYMM_AXES, SITE_SYMM_PGS).detach().cpu()
    wp_to_site_symm = _wp_to_site_symm(spacegroup)

    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    min_ss_dists, wp_projection_dists = [], []
    for sym, frac_coord, atm_type in zip(site_symm_axis, frac_coords, atom_types):
        frac_coord = frac_coord.detach().cpu().numpy()
        wp_to_ss_dist = {
            wp: torch.norm(sym.flatten() - torch.as_tensor(ss).flatten())
            for wp, ss in wp_to_site_symm.items()
        }
        min_ss_dist = min(wp_to_ss_dist.values())
        min_ss_dists.append(float(min_ss_dist.item()))
        closest_ss_wps = [
            wp for wp, dist in wp_to_ss_dist.items() if dist == min_ss_dist
        ]
        closes = []
        for wp in closest_ss_wps:
            for orbit_index in range(len(wp.ops)):
                close = (
                    _search_closest_wp(spacegroup, wp, wp.ops[orbit_index], frac_coord)
                    % 1.0
                )
                closes.append(
                    (
                        close,
                        wp,
                        orbit_index,
                        np.linalg.norm(
                            np.minimum(
                                (close - frac_coord) % 1.0, (frac_coord - close) % 1.0
                            )
                        ),
                    )
                )
        try:
            closest = sorted(closes, key=lambda x: x[-1])[0]
            wyckoff = closest[1]
            repr_index = closest[2]
            wp_projection_dists.append(float(closest[3]))
            frac_coord = closest[0]
            for index in range(len(wyckoff)):
                new_frac_coords.append(
                    wyckoff[(index + repr_index) % len(wyckoff)].operate(frac_coord)
                    % 1.0
                )
                new_atom_types.append(atm_type.detach().cpu().numpy())
                new_site_symm.append(sym.detach().cpu().numpy())
        except Exception:
            new_frac_coords.append(frac_coord)
            new_atom_types.append(atm_type.detach().cpu().numpy())
            new_site_symm.append(sym.detach().cpu().numpy())

    return (
        np.stack(new_frac_coords),
        len(new_frac_coords),
        np.stack(new_atom_types),
        np.stack(new_site_symm),
        min_ss_dists,
        wp_projection_dists,
    )


def modify_frac_coords(traj, spacegroups, num_repr):
    device = traj["frac_coords"].device
    total_atoms = 0
    updated_frac_coords, updated_num_atoms, updated_atom_types, updated_site_symm = (
        [],
        [],
        [],
        [],
    )
    min_ss_dists, wp_projection_dists = [], []
    for index in range(len(num_repr)):
        if num_repr[index] > 0:
            outputs = modify_frac_coords_one(
                traj["frac_coords"][total_atoms : total_atoms + num_repr[index]],
                traj["site_symm"][total_atoms : total_atoms + num_repr[index]],
                traj["atom_types"][total_atoms : total_atoms + num_repr[index]],
                spacegroups[index],
            )
            (
                new_frac_coords,
                new_num_atoms,
                new_atom_types,
                new_site_symm,
                min_ss_dist,
                wp_projection_dist,
            ) = outputs
            if new_num_atoms:
                updated_frac_coords.append(new_frac_coords)
                updated_num_atoms.append(new_num_atoms)
                updated_atom_types.append(new_atom_types)
                updated_site_symm.append(new_site_symm)
                min_ss_dists.append(min_ss_dist)
                wp_projection_dists.append(wp_projection_dist)
        total_atoms += num_repr[index]

    traj["frac_coords"] = torch.cat(
        [torch.from_numpy(x) for x in updated_frac_coords]
    ).to(device)
    traj["atom_types"] = torch.cat(
        [torch.from_numpy(x) for x in updated_atom_types]
    ).to(device)
    traj["num_atoms"] = torch.tensor(updated_num_atoms).to(device)
    traj["site_symm"] = torch.cat([torch.from_numpy(x) for x in updated_site_symm]).to(
        device
    )
    traj["min_ss_dists"] = min_ss_dists
    traj["wp_projection_dists"] = wp_projection_dists
    return traj


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


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hparams = cast(Any, self.hparams)
        self.decoder = hydra.utils.instantiate(
            hparams.decoder,
            time_dim=hparams.time_dim + hparams.latent_dim,
            latent_dim=hparams.latent_dim,
            pred_type=True,
            pred_site_symm_type=True,
            smooth=True,
            max_atoms=MAX_ATOMIC_NUM,
            mask_token=0,
        )
        self.beta_scheduler = hydra.utils.instantiate(hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(hparams.sigma_scheduler)
        self.time_dim = hparams.time_dim
        self.latent_dim = hparams.latent_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.spacegroup_embedding = build_mlp(
            in_dim=SG_CONDITION_DIM,
            hidden_dim=128,
            fc_num_layers=2,
            out_dim=self.latent_dim,
        )
        self.keep_lattice = hparams.cost_lattice < 1e-5
        self.keep_coords = hparams.cost_coord < 1e-5
        self.use_ks = hparams.use_ks
        self.discrete_noise = DiscreteNoiseMarginal(
            hparams.data.datamodule.atom_marginals_path,
            hparams.data.datamodule.ss_marginals_path,
            self.beta_scheduler,
        )
        self.rl_optimize_site_symm = True

    def _spacegroup_time_emb(self, times, sg_condition):
        spacegroup_emb = self.spacegroup_embedding(
            sg_condition.reshape(-1, SG_CONDITION_DIM)
        )
        return torch.cat([self.time_embedding(times), spacegroup_emb], dim=-1)

    def add_noise(self, batch, time=None):
        batch_size = batch.num_graphs
        if time is None:
            times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        else:
            time_arr = np.arange(self.beta_scheduler.timesteps, 0, -1)
            times = torch.full(
                (batch_size,), int(time_arr[time]), device=self.device, dtype=torch.long
            )
        time_emb = self._spacegroup_time_emb(times, batch.sg_condition)

        atom_types, node_mask = to_dense_batch(
            batch.atom_types - 1, batch.batch, fill_value=0
        )
        site_symms, _ = to_dense_batch(
            batch.site_symm.flatten(-2, -1), batch.batch, fill_value=0
        )
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1.0 - alphas_cumprod)
        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        ks = batch.ks
        if self.use_ks:
            lattices = lattice_ks_to_matrix_torch(batch.ks)
            ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
        else:
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            ks_mask = ks_add = None

        frac_coords = batch.frac_coords
        rand_x = torch.randn_like(frac_coords)
        rand_ks = torch.randn_like(ks)
        rand_l = torch.randn_like(lattices)
        input_ks = None
        if self.use_ks:
            assert ks_mask is not None and ks_add is not None
            input_ks = mask_ks(
                c0[:, None] * ks + c1[:, None] * rand_ks, ks_mask, ks_add
            )
            input_lattice = lattice_ks_to_matrix_torch(input_ks)
        else:
            input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        gt_atom_types_onehot = F.one_hot(
            atom_types, num_classes=self.discrete_noise.max_atomic_num
        ).float()
        atom_type_noised_probs = self.discrete_noise.apply_atom_noise(
            gt_atom_types_onehot, times
        )
        site_symm_noised_probs = self.discrete_noise.apply_site_symm_noise(
            site_symms, times, batch.spacegroup
        )
        atom_types_noised, site_symms_noised = (
            self.discrete_noise.sample_discrete_features(
                atom_type_noised_probs, site_symm_noised_probs, node_mask
            )
        )

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
            input_ks = ks

        lattice_feats = input_ks if self.use_ks else input_lattice
        noised_input = (
            time_emb,
            atom_types_noised[node_mask],
            input_frac_coords,
            lattice_feats,
            input_lattice,
            batch.num_atoms,
            batch.batch,
            site_symms_noised[node_mask],
        )
        noises = {
            "rand_l": rand_l,
            "rand_ks": rand_ks,
            "tar_x": d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom)
            / torch.sqrt(sigmas_norm_per_atom),
            "atom_types": batch.atom_types - 1,
            "site_symm": batch.site_symm.argmax(-1),
            "ks_mask": ks_mask,
        }
        return noised_input, noises, batch.batch

    def calc_sample_loss(self, input_all):
        noised_input, noises, batch_index = input_all
        hparams = cast(Any, self.hparams)
        pred_l, pred_x, pred_t_logit, pred_symm_logit = self.decoder(*noised_input)
        pred_t = F.softmax(pred_t_logit, -1)
        pred_symm = F.softmax(
            self.discrete_noise.reshape_ss(pred_symm_logit), -1
        ).flatten(-2, -1)

        if self.use_ks:
            target_lattice = cast(torch.Tensor, noises["ks_mask"]) * noises["rand_ks"]
        else:
            target_lattice = noises["rand_l"]
        loss_lattice = ((pred_l - target_lattice) ** 2).mean(
            dim=tuple(range(1, pred_l.dim()))
        )
        loss_coord = scatter(
            ((pred_x - noises["tar_x"]) ** 2).mean(dim=1),
            batch_index,
            dim=0,
            reduce="mean",
        )
        loss_type = scatter(
            F.nll_loss(
                torch.log(pred_t + 1e-20), noises["atom_types"], reduction="none"
            ),
            batch_index,
            dim=0,
            reduce="mean",
        )

        pred_symm_sections = self.discrete_noise.ss_to_sections(pred_symm)
        symm_targets = noises["site_symm"]
        loss_symm_parts = []
        for i, pred_ss_i in enumerate(pred_symm_sections):
            per_node = F.nll_loss(
                torch.log(pred_ss_i + 1e-20), symm_targets[:, i], reduction="none"
            )
            loss_symm_parts.append(scatter(per_node, batch_index, dim=0, reduce="mean"))
        loss_symm = torch.stack(loss_symm_parts).mean(dim=0)
        symm_weight = (
            hparams.cost_symm
            if bool(getattr(self, "rl_optimize_site_symm", True))
            else 0.0
        )
        loss = (
            hparams.cost_lattice * loss_lattice
            + hparams.cost_coord * loss_coord
            + hparams.cost_type * loss_type
            + symm_weight * loss_symm
        )
        return loss, (pred_l, pred_x, pred_t, pred_symm)

    def calc_kl_reg(self, agent_pred, prior_pred, batch):
        pred_l, pred_x, pred_t, pred_symm = agent_pred
        pred_l_p, pred_x_p, pred_t_p, pred_symm_p = prior_pred
        kl_term0 = ((pred_l - pred_l_p.detach()) ** 2).mean(
            dim=tuple(range(1, pred_l.dim()))
        )
        kl_term1 = scatter(
            ((pred_x - pred_x_p.detach()) ** 2).mean(dim=1),
            batch.batch,
            dim=0,
            reduce="mean",
        )
        kl_term2 = scatter(
            ((pred_t - pred_t_p.detach()) ** 2).mean(dim=1),
            batch.batch,
            dim=0,
            reduce="mean",
        )
        kl_term3 = scatter(
            ((pred_symm - pred_symm_p.detach()) ** 2).mean(dim=1),
            batch.batch,
            dim=0,
            reduce="mean",
        )
        if not bool(getattr(self, "rl_optimize_site_symm", True)):
            kl_term3 = torch.zeros_like(kl_term3)
        return kl_term0 + kl_term1 + kl_term2 + kl_term3

    def forward(self, batch):
        noised = self.add_noise(batch)
        loss, _ = self.calc_sample_loss(noised)
        return {"loss": loss.mean()}

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5):
        del diff_ratio
        batch_size = batch.num_graphs
        ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
        k_T = mask_ks(torch.randn([batch_size, 6], device=self.device), ks_mask, ks_add)
        l_T = lattice_ks_to_matrix_torch(k_T)
        x_T = torch.rand([batch.num_nodes, 3], device=self.device)
        _, node_mask = to_dense_batch(batch.batch, batch.batch, fill_value=0)
        t_T, symm_T = self.discrete_noise.sample_limit_dist(node_mask, batch.spacegroup)
        t_T = t_T[node_mask]
        symm_T = symm_T[node_mask]

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            k_T = batch.ks
            l_T = (
                lattice_ks_to_matrix_torch(k_T)
                if self.use_ks
                else lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            )

        traj = {
            self.beta_scheduler.timesteps: {
                "num_atoms": batch.num_atoms,
                "atom_types": t_T,
                "site_symm": symm_T,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
                "ks": k_T,
                "spacegroup": batch.spacegroup,
            }
        }
        for t in range(self.beta_scheduler.timesteps, 0, -1):
            times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            time_emb = self._spacegroup_time_emb(times, batch.sg_condition)
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = x_T if self.keep_coords else traj[t]["frac_coords"]
            l_t = l_T if self.keep_lattice else traj[t]["lattices"]
            k_t = k_T if self.keep_lattice else traj[t]["ks"]
            t_t = traj[t]["atom_types"]
            symm_t = traj[t]["site_symm"]

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            lattice_feats_t = k_t if self.use_ks else l_t
            _, pred_x, _, _ = self.decoder(
                time_emb,
                t_t,
                x_t,
                lattice_feats_t,
                l_t,
                batch.num_atoms,
                batch.batch,
                site_symm_probs=symm_t,
            )
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = (
                x_t - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = torch.sqrt(
                (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2))
                / (sigma_x**2)
            )
            lattice_feats_t_minus_05 = k_t if self.use_ks else l_t
            pred_l, pred_x, pred_t_logit, pred_symm_logit = self.decoder(
                time_emb,
                t_t,
                x_t_minus_05,
                lattice_feats_t_minus_05,
                l_t,
                batch.num_atoms,
                batch.batch,
                site_symm_probs=symm_t,
            )
            pred_t = F.softmax(pred_t_logit, -1)
            pred_symm = F.softmax(
                self.discrete_noise.reshape_ss(pred_symm_logit), -1
            ).flatten(-2, -1)
            pred_t, _ = to_dense_batch(pred_t, batch.batch, fill_value=0)
            pred_symm, _ = to_dense_batch(pred_symm, batch.batch, fill_value=0)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = (
                x_t_minus_05 - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )

            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
            k_t_minus_1 = (
                c0 * (k_t - c1 * pred_l) + sigmas * rand_k
                if not self.keep_lattice
                else k_t
            )
            k_t_minus_1 = mask_ks(k_t_minus_1, ks_mask, ks_add)
            l_t_minus_1 = (
                lattice_ks_to_matrix_torch(k_t_minus_1)
                if not self.keep_lattice
                else l_t
            )

            t_t_dense, _ = to_dense_batch(t_t, batch.batch, fill_value=0)
            symm_t_dense, _ = to_dense_batch(symm_t, batch.batch, fill_value=0)
            t_t_minus_1, symm_t_minus_1 = (
                self.discrete_noise.sample_zs_from_zt_and_pred(
                    t_t_dense,
                    symm_t_dense,
                    pred_t,
                    pred_symm,
                    times,
                    times - 1,
                    node_mask,
                    batch.spacegroup,
                )
            )
            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": t_t_minus_1[node_mask],
                "site_symm": symm_t_minus_1[node_mask],
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
                "ks": k_t_minus_1,
                "spacegroup": batch.spacegroup,
            }

        dummy_ind = (
            traj[0]["atom_types"].argmax(dim=-1) == self.discrete_noise.max_atomic_num
        ).long()
        traj[0]["frac_coords"] = traj[0]["frac_coords"][(1 - dummy_ind).bool()]
        traj[0]["atom_types"] = traj[0]["atom_types"][(1 - dummy_ind).bool()]
        traj[0]["site_symm"] = traj[0]["site_symm"][(1 - dummy_ind).bool()]
        traj[0]["num_atoms"] = find_num_atoms(dummy_ind, batch.num_atoms).to(
            self.device
        )
        empty_crystals = (traj[0]["num_atoms"] == 0).long()
        traj[0]["ks"] = traj[0]["ks"][(1 - empty_crystals).bool()]
        traj[0]["lattices"] = traj[0]["lattices"][(1 - empty_crystals).bool()]
        traj[0] = modify_frac_coords(traj[0], batch.spacegroup, traj[0]["num_atoms"])
        return traj[0], {}
