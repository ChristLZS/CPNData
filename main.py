#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:41
"""
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils.convert import to_networkx

from layer import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    edge_index = []
    edge_attr = []
    y = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 4:
            continue

        source = int(parts[0])
        relation = int(parts[1])
        target = int(parts[2])
        time = int(parts[3])

        edge_index.append([source, target])
        edge_attr.append([relation, time])
        y.append([time])  # Example: Using 'time' as the target value

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    x = torch.zeros(edge_index.max().item() + 1, dtype=torch.float)  # Assuming node features are all zeros

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# Custom layers for Graphormer model
class Graphormer(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features, num_classes):
        super(Graphormer, self).__init__()

        self.num_layers = args["num_layers"]
        self.input_node_dim = num_node_features
        self.node_dim = args["node_dim"]
        self.input_edge_dim = num_edge_features
        self.edge_dim = args["edge_dim"]
        self.output_dim = num_classes
        self.num_heads = args["num_heads"]
        self.max_in_degree = args["max_in_degree"]
        self.max_out_degree = args["max_out_degree"]
        self.max_path_distance = args["max_path_distance"]

        # Define layers
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim,
        )
        self.spatial_encoding = SpatialEncoding(
            max_path_distance=self.max_path_distance,
        )
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    num_heads=self.num_heads,
                    max_path_distance=self.max_path_distance,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if isinstance(data, Data):
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)
        x = global_mean_pool(x, data.batch)

        return x


def shortest_path_distance(
        data: Data,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


# 批量获取最短路径数据
def batched_shortest_path_distance(
        data,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []  # 重新标记节点的图
    shift = 0  # 节点偏移量
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(
            nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)})
        )
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths


def evaluate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        # Initialize evaluation metrics
        total_mr = 0.0
        total_mrr = 0.0
        total_hits_at_1 = 0.0
        total_hits_at_3 = 0.0
        total_hits_at_10 = 0.0
        num_batches = len(data_loader)

        for data in data_loader:
            data = data.to(device)
            # Perform forward pass
            output = model(data)

            # Example: Compute evaluation metrics (MR, MRR, Hits@k)
            # Adjust this part based on your specific evaluation metrics
            # Here's a simple example assuming output is the predicted values
            # and data.y is the ground truth

            # Example calculation for MR, MRR, Hits@k
            pred_values = output.flatten()  # Assuming output is a single value prediction
            ground_truth = data.y.flatten()  # Assuming ground truth is also a single value

            # Compute Mean Rank (MR)
            mr = torch.mean(pred_values).item()
            total_mr += mr

            # Compute Mean Reciprocal Rank (MRR)
            sorted_indices = torch.argsort(pred_values, descending=True)
            rank = torch.where(sorted_indices == ground_truth.unsqueeze(1))[1]
            mrr = torch.mean(1.0 / (rank.float() + 1)).item()
            total_mrr += mrr

            # Compute Hits@k (for example, Hits@1, Hits@3, Hits@10)
            k = 10
            hits_at_1 = torch.mean((rank < 1).float()).item()
            hits_at_3 = torch.mean((rank < 3).float()).item()
            hits_at_10 = torch.mean((rank < 10).float()).item()

            total_hits_at_1 += hits_at_1
            total_hits_at_3 += hits_at_3
            total_hits_at_10 += hits_at_10

        # Calculate average metrics across all batches
        avg_mr = total_mr / num_batches
        avg_mrr = total_mrr / num_batches
        avg_hits_at_1 = total_hits_at_1 / num_batches
        avg_hits_at_3 = total_hits_at_3 / num_batches
        avg_hits_at_10 = total_hits_at_10 / num_batches

    # Return average metrics
    return avg_mr, avg_mrr, avg_hits_at_1, avg_hits_at_3, avg_hits_at_10


# 遍历图,查询所有节点对之间的最短路径
def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


# 遍历图,查询从 source 节点到所有节点的最短路径
def floyd_warshall_source_to_all(G, source, cutoff=None):
    "Floyd-Warshall算法查询最短路径(BFS遍历图)"
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if cutoff is not None and cutoff <= level:
            break

    # Test
    # print(node_paths)
    # print(edge_paths)
    # sys.exit(0)
    # 0: [0], 1: [0, 1], 10: [0, 10], 2: [0, 1, 2], 9: [0, 10, 9], 11: [0, 10, 11], 3: [0, 1, 2, 3], 8: [0, 10, 9, 8], 6: [0, 10, 11, 6], 4: [0, 1, 2, 3, 4], 7: [0, 10, 9, 8, 7], 5: [0, 10, 11, 6, 5]}
    # {0: [], 1: [0], 10: [1], 2: [0, 3], 9: [1, 23], 11: [1, 24], 3: [0, 3, 5], 8: [1, 23, 20], 6: [1, 24, 26], 4: [0, 3, 5, 8], 7: [1, 23, 20, 18], 5: [1, 24, 26, 13]}

    return node_paths, edge_paths


# Example main function to train and evaluate the model
def main():
    # Example arguments
    args = {
        'num_layers'       : 3,
        'node_dim'         : 128,
        'edge_dim'         : 128,
        'num_heads'        : 4,
        'max_in_degree'    : 10,
        'max_out_degree'   : 10,
        'max_path_distance': 5,
        'num_node_features': 16,
        'num_edge_features': 16,
        'num_classes'      : 1
    }

    # Load your dataset from the generated files in the data directory
    train_data = load_data('data/train.txt')
    test_data = load_data('data/test.txt')
    val_data = load_data('data/val.txt')

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and criterion
    model = Graphormer(args, args['num_node_features'], args['num_edge_features'], args['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1).float())  # Adjust loss calculation as per your task
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}")

    # Example evaluation (you should implement your own evaluation function)
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss}")


if __name__ == '__main__':
    main()
