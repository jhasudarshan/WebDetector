import pandas as pd
from sklearn.utils import resample
import torch
import joblib

from FeaturePipeline.feature_pipeline import build_feature_vector, normalize_url
from FeatureCleanPipeline.cleaning_train_pipeline import FeaturePipeline
from PreClassifierPipeline.preclassifier import PreClassifier
from GraphBuilderPipeline.graph_builder import GraphDatasetBuilder
from GnnModelPipeline.gnn_training import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import os

lock = threading.Lock()


def save_shards(df, num_splits, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    chunk_size = len(df) // num_splits

    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_splits - 1 else len(df)

        split = df.iloc[start:end]

        file_path = os.path.join(output_dir, f"{prefix}_part_{i+1}.csv")
        split.to_csv(file_path, index=False)

        print(f"[save_shards] Saved {file_path} ({len(split)} rows)")

class FullPipeline:
    def __init__(self, raw_path):
        self.raw_path = raw_path

    # -----------------------------
    # Step 1: Balance Dataset
    # -----------------------------
    def balance_dataset(self, df, samples_per_class):
        balanced = []
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            subset = resample(
                subset,
                replace=True,
                n_samples=samples_per_class,
                random_state=42
            )
            balanced.append(subset)
        return pd.concat(balanced)

    # -----------------------------
    # Step 2: Split Dataset
    # -----------------------------
    def prepare_data(self):
        df = pd.read_csv(self.raw_path)
        df = df[['url', 'type']].rename(columns={'type': 'label'})

        # ── Drop rows with missing or blank URLs right at the source ──
        df['url'] = df['url'].astype(str).str.strip()
        before = len(df)
        df = df[df['url'].notna() & (df['url'] != '') & (df['url'].str.lower() != 'nan')]
        dropped = before - len(df)
        if dropped:
            print(f"[prepare_data] Dropped {dropped} rows with empty/null URLs.")

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )

        # Balance dataset
        train_df = self.balance_dataset(train_df, 7500)
        test_df = self.balance_dataset(test_df, 1500)

        # Save shards instead of single file
        save_shards(train_df, num_splits=5, prefix="train", output_dir="./DemoDataSet/train")
        save_shards(test_df, num_splits=2, prefix="test", output_dir="./DemoDataSet/test")

    # -----------------------------
    # Step 3: Feature Extraction
    # -----------------------------
    def extract_features(self, input_path, output_path, max_workers=7):
        df = pd.read_csv(input_path)

        # ── Normalise URLs in-place before any processing ──
        # This fixes bare domains, scheme-relative URLs, and NaN values
        # so no worker ever receives an empty or malformed URL.
        df['url'] = (
            df['url']
            .astype(str)
            .str.strip()
            .apply(normalize_url)
        )

        # Drop rows where normalisation still produced an empty string
        # (these are truly unrecoverable — e.g. cells that were just whitespace)
        before = len(df)
        df = df[df['url'] != '']
        dropped = before - len(df)
        if dropped:
            print(f"[extract_features] Skipped {dropped} unrecoverable URL rows in {input_path}.")

        urls   = df['url'].tolist()
        labels = df['label'].tolist()
        total  = len(urls)

        records = [None] * total

        def process(i, url, label):
            try:
                features = build_feature_vector(url)   # url is already normalised
                features['label'] = label
                with lock:
                    print(f"[Feature Extraction] Processed row {i}/{total}")
                return i, features
            except Exception as e:
                with lock:
                    print(f"[ERROR] Row {i} ({url!r}) failed: {e}")
                return i, {"label": label}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process, i, url, label): i
                for i, (url, label) in enumerate(zip(urls, labels))
            }
            for future in as_completed(futures):
                i, features = future.result()
                records[i] = features

        pd.DataFrame(records).to_csv(output_path, index=False)

    # -----------------------------
    # Step 4: Cleaning Pipeline
    # -----------------------------
    def run_cleaning(self, input_path, output_path):
        pipeline = FeaturePipeline(input_path, output_path)
        pipeline.run()

    # -----------------------------
    # Step 5: Pre-Classifier
    # -----------------------------
    def run_preclassifier(self, input_path, output_path, test_output_path):
        pc = PreClassifier(input_path)
        pc.load_data()
        pc.train()
        pc.save_model("./TrainedModels/pre_classifier_new_new.pkl")
        pc.generate_semantic_features(output_path)
        pc.generate_semantic_features(test_output_path)

    # -----------------------------
    # Step 6: Graph Builder
    # -----------------------------
    def build_graph(self, input_path, output_path):
        builder = GraphDatasetBuilder(input_path, output_path)
        builder.run()

    # -----------------------------
    # Step 7: GNN Training
    # -----------------------------
    def train_gnn(self, train_graph_path, test_graph_path):
        train_data = torch.load(train_graph_path, weights_only=False)
        test_data  = torch.load(test_graph_path,  weights_only=False)
        model = train_model(train_data)
        torch.save(model.state_dict(), "./TrainedModels/gnn_model_m_new.pth")
        evaluate_model(model, test_data)

    # -----------------------------
    # RUN ALL
    # -----------------------------
    def run(self):
        # print("Step 1: Preparing data...")
        # self.prepare_data()

        print("Step 2: Feature extraction...")
        self.extract_features("./DemoDataSet/train/train_part_1.csv", "./DemoDataSet/features/train_features_part_1.csv")
        # self.extract_features("./DemoDataSet/test/test_part_1.csv",  "./DemoDataSet/features/test_features_part_1.csv")

        # print("Step 3: Cleaning...")
        # self.run_cleaning("./DemoDataSet/train_features.csv", "./DemoDataSet/train_clean.csv")
        # self.run_cleaning("./DemoDataSet/test_features.csv",  "./DemoDataSet/test_clean.csv")
        #
        # print("Step 4: Pre-classifier...")
        # self.run_preclassifier(
        #     "./DemoDataSet/train_clean.csv",
        #     "./DemoDataSet/train_semantic.csv",
        #     "./DemoDataSet/test_semantic.csv"
        # )
        #
        # print("Step 5: Graph building...")
        # self.build_graph("./DemoDataSet/train_semantic.csv", "./DemoDataSet/train_graph.pt")
        # self.build_graph("./DemoDataSet/test_semantic.csv",  "./DemoDataSet/test_graph.pt")
        #
        # print("Step 6: Training GNN...")
        # self.train_gnn("./DemoDataSet/train_graph.pt", "./DemoDataSet/test_graph.pt")
        #
        # print("Pipeline completed!")


p = FullPipeline("./DataSet/v3/final_dataset_with_all_features_v3.csv")
p.run()