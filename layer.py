#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:40
"""
from typing import Tuple
import torch
from torch import nn
from torch_geometric.utils import degree
import sys


# 中心性编码，将节点的入度和出度作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入
class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(
            torch.randn((max_in_degree, node_dim))
        )  # 生成一个随机张量,形状为(max_in_degree, node_dim)
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        # 每个节点的入度，edge_index[1]是边的终点，即入度
        in_degree = self.decrease_to_max_value(
            degree(index=edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = self.decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )

        # Test
        # print("\n x: ", x.shape)
        # print("edge_index: ", edge_index)
        # print("num_nodes: ", num_nodes)
        # print("in_degree: ", in_degree)
        # print("out_degree: ", out_degree)
        # print("z_in: ", self.z_in)
        # print("z_out: ", self.z_out)
        # sys.exit(0)

        # 将每个节点度的数值作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入
        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x

    def decrease_to_max_value(self, x, max_value):
        "限制节点度的最大值"
        x[x > max_value] = max_value

        return x


# 空间编码，将节点之间的最短路径长度作为索引，挑选b的每行，形成每个节点的嵌入
class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """

        # 初始化空间编码矩阵,形状为(num_nodes, num_nodes),每个元素为0,并移动到GPU
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(
            next(self.parameters()).device
        )

        for src in paths:
            for dst in paths[src]:
                spatial_matrix[src][dst] = self.b[
                    min(len(paths[src][dst]), self.max_path_distance) - 1
                ]  # 索引从 0 到 max_path_distance-1
                # Test
                # print("\n src: ", src)
                # print("dst: ", dst)
                # print("paths[src][dst]: ", paths[src][dst])
                # print("len(paths[src][dst]): ", len(paths[src][dst]))
                # print(
                #     "min(len(paths[src][dst]), self.max_path_distance): ",
                #     min(len(paths[src][dst]), self.max_path_distance),
                # )
                # print("spatial_matrix[src][dst]: ", spatial_matrix[src][dst])
                # sys.exit(0)

        return spatial_matrix


# 边特征,将节点之间的最短路径作为索引,
class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(
            torch.randn(self.max_path_distance, self.edge_dim)
        )

    def forward(
        self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths
    ) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][
                    : self.max_path_distance
                ]  # 获取最短路径（截断）
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = self.dot_product(
                    self.edge_vector[weight_inds], edge_attr[path_ij]
                ).mean()

                # Test
                # print("weight_inds: ", weight_inds)
                # print("cij[src][dst]: ", cij[src][dst])
                # sys.exit(0)

        cij = torch.nan_to_num(cij)  # 路径可能无数值，后续计算产生NaN

        return cij

    def dot_product(self, x1, x2) -> torch.Tensor:
        return (x1 * x2).sum(
            dim=1
        )  # 沿着第二维度求和，即对二维张量的每行求和（返回值为一维张量，一行多列）


# 注意力头,将节点的特征矩阵作为输入，计算注意力权重,并返回节点嵌入
class GraphormerAttentionHead(nn.Module):
    def __init__(
        self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int
    ):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()

        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    # 通过query和key计算注意力权重，然后通过value计算节点嵌入
    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        b: torch.Tensor,
        edge_paths,
        ptr=None,
    ) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        #
        # 没有用到的地方，填充为-inf
        batch_mask_neg_inf = torch.full(
            size=(x.shape[0], x.shape[0]), fill_value=-1e6
        ).to(next(self.parameters()).device)

        # 用于填充为0
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(
            next(self.parameters()).device
        )

        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(
                next(self.parameters()).device
            )
            batch_mask_zeros += 1
        else:
            # 实际ptr不为空，即为批图
            # 批图的mask,邻接矩阵以对角阵组合
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1

        # 3个线性层
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        # 计算边编码
        c = self.edge_encoding(x, edge_attr, edge_paths)

        # 计算注意力权重
        a = self.compute_a(key, query, ptr)

        # 计算注意力权重的分子,分母为0的地方填充为-inf,b为空间编码,c为边编码
        a = (a + b + c) * batch_mask_neg_inf

        # 通过softmax计算注意力权重,并将inf填充为0
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros  # e^(-inf) ——> 0

        # 矩阵乘法,计算节点嵌入
        x = softmax.mm(value)

        return x

    # 通过query和key计算注意力权重,对应公式的第一部分,最后除以根号d,即除以特征维度的平方根
    def compute_a(self, key, query, ptr=None):
        "Query-Key product(normalization)"
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = (
                    query[ptr[i] : ptr[i + 1]].mm(
                        key[ptr[i] : ptr[i + 1]].transpose(0, 1)
                    )
                    / query.size(-1) ** 0.5
                )

        return a


# 多头注意力,将多个注意力头的输出拼接,并通过线性层输出
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_in: int,
        dim_q: int,
        dim_k: int,
        edge_dim: int,
        max_path_distance: int,
    ):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                GraphormerAttentionHead(
                    dim_in, dim_q, dim_k, edge_dim, max_path_distance
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(
        self, x: torch.Tensor, edge_attr: torch.Tensor, b: torch.Tensor, edge_paths, ptr
    ) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat(
                [
                    attention_head(x, edge_attr, b, edge_paths, ptr)
                    for attention_head in self.heads
                ],
                dim=-1,
            )
        )


# Graphormer编码器层,包含多头注意力和前馈网络
class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param num_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(
        self, x: torch.Tensor, edge_attr: torch.Tensor, b: torch, edge_paths, ptr
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """

        # 上述公式
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
