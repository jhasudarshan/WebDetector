import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import faiss
import time
from sklearn.model_selection import train_test_split


class FaissGraphBuilder:
    def __init__(self,
                 prob_path,
                 output_path,
                 k=20,
                 sim_threshold=0.7,
                 use_gpu=False):

        self.prob_path = prob_path
        self.output_path = output_path
        self.k = k
        self.sim_threshold = sim_threshold
        self.use_gpu = use_gpu

    # -----------------------------
    # Load Data
    # -----------------------------
    def load_data(self):
        start = time.time()

        df = pd.read_csv(self.prob_path)
        df, _ = train_test_split(
            df,
            train_size=150000,
            stratify=df["label"],
            random_state=42
        )

        print(f"[INFO] Loaded data: {df.shape}")

        self.X = np.ascontiguousarray(
            df.drop(columns=["label"]).values.astype('float32')
        )

        self.y = torch.tensor(df["label"].values, dtype=torch.long)

        print(f"[INFO] Feature matrix shape: {self.X.shape}")
        print(f"[INFO] Labels shape: {self.y.shape}")

        # Normalize
        faiss.normalize_L2(self.X)
        print("[INFO] Features normalized for cosine similarity")

        print(f"[TIME] load_data: {time.time() - start:.2f}s")

    # -----------------------------
    # Build FAISS Index
    # -----------------------------
    def build_index(self):
        start = time.time()

        dim = self.X.shape[1]
        print(f"[INFO] Building FAISS index (dim={dim})")

        index = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            print("[INFO] Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(self.X)
        self.index = index

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors")
        print(f"[TIME] build_index: {time.time() - start:.2f}s")

    # -----------------------------
    # Build Edges
    # -----------------------------
    def build_edges(self):
        start = time.time()

        print(f"[INFO] Searching top-{self.k} neighbors...")
        distances, indices = self.index.search(self.X, self.k)

        edges = set()
        weights = []

        n = self.X.shape[0]

        print("[INFO] Building edges...")

        for i in range(n):

            if i % 50000 == 0 and i > 0:
                print(f"[PROGRESS] Processed {i}/{n} nodes")

            for j, sim in zip(indices[i], distances[i]):

                if i == j:
                    continue

                if sim >= self.sim_threshold:
                    edges.add((i, j))
                    edges.add((j, i))

                    weights.extend([sim, sim])

        if len(edges) == 0:
            raise ValueError("No edges created — lower threshold")

        print(f"[INFO] Total edges created: {len(edges)}")

        # Convert to tensors
        edge_list = list(edges)

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_weight = torch.tensor(weights, dtype=torch.float)

        self.X_tensor = torch.tensor(self.X, dtype=torch.float)

        print(f"[INFO] edge_index shape: {self.edge_index.shape}")
        print(f"[INFO] edge_weight shape: {self.edge_weight.shape}")

        print(f"[TIME] build_edges: {time.time() - start:.2f}s")

    # -----------------------------
    # Build PyG Data
    # -----------------------------
    def build_data(self):
        start = time.time()

        self.data = Data(
            x=self.X_tensor,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            y=self.y
        )

        print(f"[INFO] PyG Data object created")
        print(f"[INFO] Nodes: {self.data.num_nodes}")
        print(f"[INFO] Edges: {self.data.num_edges}")

        print(f"[TIME] build_data: {time.time() - start:.2f}s")

    # -----------------------------
    # Save
    # -----------------------------
    def save(self):
        start = time.time()

        torch.save(self.data, self.output_path)

        print(f"[INFO] Graph saved at {self.output_path}")
        print(f"[TIME] save: {time.time() - start:.2f}s")

    # -----------------------------
    # Run Pipeline
    # -----------------------------
    def run(self):
        total_start = time.time()

        print("========== FAISS GRAPH BUILD START ==========")

        self.load_data()
        self.build_index()
        self.build_edges()
        self.build_data()
        self.save()

        print("========== FAISS GRAPH BUILD COMPLETE ==========")
        print(f"[TOTAL TIME]: {time.time() - total_start:.2f}s")