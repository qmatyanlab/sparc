from typing import Dict, Union
import math

import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate, Dropout, BatchNorm, FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct


class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class CustomComposeWithDropoutAndBN(torch.nn.Module):
    def __init__(self, first, second, dropout, batch_norm, use_batch_norm=True):
        super().__init__()
        self.first = first
        self.second = second
        self.dropout = dropout
        self.batch_norm = batch_norm if use_batch_norm else None
        self.use_batch_norm = use_batch_norm
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = input[0]
        x_in = x.clone()
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        if self.use_batch_norm and self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.dropout(x)
        if x.shape[-1] == x_in.shape[-1]:
            x = x + x_in
        return x


def to_irreps(x):
    if x is None:
        return None
    if isinstance(x, o3.Irreps):
        return x
    if isinstance(x, int):
        return o3.Irreps(f"{x}x0e")
    return o3.Irreps(x)


class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        num_neighbors=1.0,
        num_nodes=1.0,
        reduce_output=True,
        dropout_prob=0.5,
        use_batch_norm=True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        fc_neurons = [self.number_of_basis, 100]

        self.irreps_in = irreps_in
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = irreps_node_attr

        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            dropout = Dropout(gate.irreps_out, p=dropout_prob)
            batch_norm = (
                BatchNorm(gate.irreps_out, normalization="component")
                if self.use_batch_norm
                else None
            )

            self.layers.append(
                CustomComposeWithDropoutAndBN(
                    conv, gate, dropout, batch_norm, use_batch_norm=self.use_batch_norm
                )
            )
            irreps = gate.irreps_out

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                fc_neurons,
                num_neighbors,
            )
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]):
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        if "edge_index" in data:
            edge_src = data["edge_index"][0]
            edge_dst = data["edge_index"][1]
            edge_vec = data["edge_vec"]
        else:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            x = data["x"]
        else:
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        x = self._apply_softplus_to_0e(x)

        if self.reduce_output:
            return scatter(x, batch, dim_size=batch.max().item() + 1).div(
                self.num_nodes**0.5
            )
        return x

    def _apply_softplus_to_0e(self, x: torch.Tensor) -> torch.Tensor:
        slices = self.irreps_out.slices()
        irreps = self.irreps_out
        out_chunks = []

        for i, (_, ir) in enumerate(irreps):
            chunk = x[..., slices[i]]
            if ir == o3.Irrep("0e"):
                out_chunks.append(torch.nn.functional.softplus(chunk))
            else:
                out_chunks.append(chunk)

        return torch.cat(out_chunks, dim=-1)


class Convolution(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_out
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_in
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_out
        )

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded
    ) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim_size=x.shape[0]).div(
            self.num_neighbors**0.5
        )

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)
