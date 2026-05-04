import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_dynamic_graph(X, similarity_threshold=0.8, max_neighbors=20):
    """
    Build graph using cosine similarity + dynamic neighbor selection
    """

    sim_matrix = cosine_similarity(X)

    edge_index = []
    edge_weight = []

    num_nodes = sim_matrix.shape[0]

    for i in range(num_nodes):
        # Get sorted neighbors (excluding self)
        neighbors = np.argsort(-sim_matrix[i])
        count = 0

        for j in neighbors:
            if i == j:
                continue

            sim_score = sim_matrix[i][j]

            # Dynamic selection (threshold-based)
            if sim_score < similarity_threshold:
                break

            edge_index.append([i, j])
            edge_weight.append(sim_score)

            count += 1
            if count >= max_neighbors:
                break

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_weight


class GraphDatasetBuilder:
    def __init__(self, path, output_path):
        self.path = path
        self.output_path = output_path

    def load_data(self):
        df = pd.read_csv(self.path)

        self.y = torch.tensor(df["label"].values, dtype=torch.long)
        self.X = torch.tensor(
            df.drop(columns=["label"]).values,
            dtype=torch.float
        )

    def build_edges(self):
        edge_index, edge_weight = build_dynamic_graph(
            self.X.numpy(),
            similarity_threshold=0.75,   # tune this
            max_neighbors=15             # prevents dense graph
        )

        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def build_data(self):
        self.data = Data(
            x=self.X,
            edge_index=self.edge_index,
            edge_attr=self.edge_weight,
            y=self.y
        )

    def save(self):
        torch.save(self.data, self.output_path)
        print("Graph dataset saved!")

    def run(self):
        self.load_data()
        self.build_edges()
        self.build_data()
        self.save()


















# import pandas as pd
# import torch
# from torch_geometric.data import Data
# from sklearn.neighbors import kneighbors_graph
#
# def build_graph(X, k=5):
#     A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
#     edge_index = A.nonzero()
#     return edge_index
#
# class GraphDatasetBuilder:
#     def __init__(self, path, output_path):
#         self.path = path
#         self.output_path = output_path
#
#     def load_data(self):
#         df = pd.read_csv(self.path)
#
#         self.y = torch.tensor(df["label"].values, dtype=torch.long)
#         self.X = torch.tensor(
#             df.drop(columns=["label"]).values,
#             dtype=torch.float
#         )
#
#     def build_edges(self):
#         edge_index = build_graph(self.X.numpy(), k=5)
#         self.edge_index = torch.tensor(edge_index, dtype=torch.long)
#
#     def build_data(self):
#         self.data = Data(
#             x=self.X,
#             edge_index=self.edge_index,
#             y=self.y
#         )
#
#     def save(self):
#         torch.save(self.data, self.output_path)
#         print("Graph dataset saved!")
#
#     def run(self):
#         self.load_data()
#         self.build_edges()
#         self.build_data()
#         self.save()