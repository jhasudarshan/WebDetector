import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


class FeaturePipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

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

        # Drop unnecessary columns
        drop_cols = ["url", "type", "domain", "scan_date"]
        self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)

        # Ensure label exists
        if "label" not in self.df.columns:
            raise ValueError("Label column missing!")

        # Convert all to numeric safely
        self.df = self.df.apply(pd.to_numeric, errors='coerce')

        # Fill missing
        self.df.fillna(-1, inplace=True)

        # Clip extreme values (safety)
        if "url_len" in self.df.columns:
            self.df["url_len"] = self.df["url_len"].clip(0, 2000)

    # -----------------------------
    # Step 3: Feature Separation
    # -----------------------------
    def split_features_labels(self):
        self.y = self.df["label"]
        self.X = self.df.drop(columns=["label"])

    # -----------------------------
    # Step 4: Feature Selection
    # -----------------------------
    def feature_selection(self):

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        importances = pd.Series(model.feature_importances_, index=self.X.columns)

        selected_features = importances[importances > 0.01].index.tolist()

        print(f"Selected features: {len(selected_features)}")

        self.X = self.X[selected_features]

        # Save selected features
        joblib.dump(selected_features, "selected_features.pkl")

    # -----------------------------
    # Step 5: Correlation Removal
    # -----------------------------
    def remove_correlated_features(self, threshold=0.9):

        corr_matrix = self.X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        print(f"Dropping correlated features: {len(to_drop)}")

        self.X.drop(columns=to_drop, inplace=True)

        # Save final features
        joblib.dump(self.X.columns.tolist(), "final_features.pkl")

    # -----------------------------
    # Step 6: Normalization
    # -----------------------------
    def normalize_features(self):

        self.scaler = MinMaxScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns
        )

        # Save scaler
        joblib.dump(self.scaler, "scaler.pkl")

    # -----------------------------
    # Step 7: Save Output
    # -----------------------------
    def save_output(self):

        processed_df = pd.concat([self.X, self.y.reset_index(drop=True)], axis=1)

        processed_df.to_csv(self.output_path, index=False)

        print(f"Saved processed data to {self.output_path}")

    # -----------------------------
    # Run Full Pipeline
    # -----------------------------
    def run(self):
        self.load_data()
        self.clean_data()
        self.split_features_labels()
        self.feature_selection()
        self.remove_correlated_features()
        self.normalize_features()
        self.save_output()