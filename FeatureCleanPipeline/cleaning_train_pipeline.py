import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

class FeaturePipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.label_map = {
            "benign": 0,
            "phishing": 1,
            "defacement": 2,
            "malware": 3
        }

    # -----------------------------
    # Step 1: Load Data
    # -----------------------------
    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded data shape: {self.df.shape}")

    # -----------------------------
    # Step 2: Basic Cleaning
    # -----------------------------
    def clean_data(self):
        # Drop URL
        if "url" in self.df.columns:
            self.df.drop(columns=["url"], inplace=True)

        # # -------- REMOVE CATEGORICAL FIELDS --------
        categorical_cols = [
            "hosting_ip",
            "asn",
            "registrar",
            "whois_country",
            "ssl_issuer"
        ]

        self.df.drop(columns=[col for col in categorical_cols if col in self.df.columns], inplace=True)

        # -------- HANDLE NUMERIC --------
        self.df.fillna(-1, inplace=True)

        if "url_length" in self.df.columns:
            self.df["url_length"] = self.df["url_length"].clip(0, 2000)

    # -----------------------------
    # Step 3: Label Encoding
    # -----------------------------
    def encode_labels(self):
        self.df["label"] = self.df["label"].map(self.label_map)

    # -----------------------------
    # Step 4: Feature Separation
    # -----------------------------
    def split_features_labels(self):
        self.y = self.df["label"]
        self.X = self.df.drop(columns=["label"])

    # -----------------------------
    # Step 5: Feature Importance
    # -----------------------------
    def feature_selection(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        importances = pd.Series(model.feature_importances_, index=self.X.columns)

        # Keep only important features
        selected_features = importances[importances > 0.01].index
        print(f"Selected features: {len(selected_features)}")

        self.X = self.X[selected_features]

        joblib.dump(selected_features, "new_selected_features.pkl")

    # -----------------------------
    # Step 6: Correlation Removal
    # -----------------------------
    def remove_correlated_features(self, threshold=0.9):
        corr_matrix = self.X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Dropping correlated features: {len(to_drop)}")

        self.X.drop(columns=to_drop, inplace=True)

    # -----------------------------
    # Step 7: Normalization
    # -----------------------------
    def normalize_features(self):
        scaler = MinMaxScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns
        )

    # -----------------------------
    # Step 8: Save Output
    # -----------------------------
    def save_output(self):
        processed_df = pd.concat([self.X, self.y], axis=1)
        processed_df.to_csv(self.output_path, index=False)
        print(f"Saved processed data to {self.output_path}")

    # -----------------------------
    # Run Full Pipeline
    # -----------------------------
    def run(self):
        self.load_data()
        self.clean_data()
        self.encode_labels()
        self.split_features_labels()
        self.feature_selection()
        self.remove_correlated_features()
        self.normalize_features()
        self.save_output()
