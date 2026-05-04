import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class PreClassifier:
    def __init__(self, input_path):
        self.input_path = input_path

    def load_data(self):
        df = pd.read_csv(self.input_path)

        # Safety: remove any accidental non-numeric columns
        df = df.select_dtypes(include=["number"])

        self.y = df["label"]
        self.X = df.drop(columns=["label"])

    def train(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        self.model.fit(self.X, self.y)

        #save features names here
        joblib.dump(self.X.columns.tolist(), "./TrainedModels/PreClassifier/new_selected_feature_2.pkl")

    def save_model(self, path="pre_classifier.pkl"):
        joblib.dump(self.model, path)


    def generate_semantic_features(self, output_path="X_hat.csv"):
        # Probabilities → shape (N, 4)
        X_hat = self.model.predict_proba(self.X)

        X_hat_df = pd.DataFrame(
            X_hat,
            columns=[
                "prob_benign",
                "prob_phishing",
                "prob_defacement",
                "prob_malware"
            ]
        )

        # Save with labels (important for next stage)
        final_df = pd.concat([X_hat_df, self.y.reset_index(drop=True)], axis=1)
        final_df.to_csv(output_path, index=False)

        print(f"Saved semantic features to {output_path}")