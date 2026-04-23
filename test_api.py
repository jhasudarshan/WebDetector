import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from Helper.FeaturePipeline.feature_pipeline import build_feature_vector
from Helper.GnnModelPipeline.gnn_training import GNNModel
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed


class GNNUrlPredictor:
    def __init__(self, model_path, base_graph_path, pre_classifier_path, feature_path):
        """
        model_path: trained GNN weights (.pth)
        base_graph_path: existing graph dataset (.pt)
        """

        # Load base graph
        self.base_data = torch.load(base_graph_path, weights_only=False)

        # Initialize model
        self.model = GNNModel(
            input_dim=self.base_data.x.shape[1],
            hidden_dim=32,
            num_classes=4
        )

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load Pre-Classifier
        self.pre_model = joblib.load(pre_classifier_path)

        self.selected_features = joblib.load(feature_path)

        # Label mapping
        self.labels = ["benign", "phishing", "defacement", "malware"]

    # -----------------------------
    # Normalize URL
    # -----------------------------
    def normalize_url(self, url):
        if not isinstance(url, str):
            return None

        url = url.strip()

        if not url:
            return None

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        return url

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    def extract_features(self, url):
        url = self.normalize_url(url)
        if not url:
            return None

        #step1 : raw features (33)
        features = build_feature_vector(url)

        df = pd.DataFrame([features])
        df.fillna(-1, inplace=True)

        # Clip if used during training
        if "url_length" in df.columns:
            df["url_length"] = df["url_length"].clip(0, 2000)


        #step 2: align features)
        df = df.reindex(columns=self.selected_features, fill_value=0)

        #step3: convert -> semantic features(4)
        probs = self.pre_model.predict_proba(df)

        return torch.tensor(probs, dtype=torch.float)

    # -----------------------------
    # Build Graph Edges
    # -----------------------------
    def rebuild_edges(self, X, k=5):
        A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = A.nonzero()
        return torch.tensor(edge_index, dtype=torch.long)

    # -----------------------------
    # Predict Single URL
    # -----------------------------
    def predict(self, url):
        new_x = self.extract_features(url)

        if new_x is None:
            return {"error": "Invalid URL"}

        # Clone base graph (important!)
        data = self.base_data.clone()

        # Append new node
        data.x = torch.cat([data.x, new_x], dim=0)

        # Rebuild graph
        data.edge_index = self.rebuild_edges(data.x.numpy(), k=5)

        # Inference
        with torch.no_grad():
            out = self.model(data)

        probs = torch.softmax(out[-1], dim=0).numpy()
        pred_class = np.argmax(probs)

        return {
            "url": url,
            "prediction": self.labels[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": {
                label: float(prob)
                for label, prob in zip(self.labels, probs)
            }
        }

predictor = GNNUrlPredictor(
    model_path="./Helper/GnnModels/final_gnn_model_2.pth",
    base_graph_path="./Helper/GraphData/final_result_graph_data_1.pt",
    pre_classifier_path="./Helper/PreClassifier/final_result_pre_classifier.pk1",
    feature_path="Helper/PreClassifier/selected_features.pkl"
)


# Assuming your CSV file has a column named 'url'
file_path = "./Helper/DataSet/extracted_tools.csv"
result_path = "./Helper/ResultDataSet/unique_urls_result.csv"

results = []


def process_url(row, predictor):
    """Function to process a single URL prediction."""
    url = row['url']
    try:
        result = predictor.predict(url)

        record = {
            "url": url,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        }

        # Add probability distribution
        for label, prob in result["probabilities"].items():
            record[f"prob_{label}"] = prob

        return record
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


# --- Main Execution Block ---
try:
    # Load URLs from CSV
    df = pd.read_csv(file_path)
    results = []

    # Adjust max_workers based on your API limits or CPU (e.g., 5, 10, or 20)
    MAX_WORKERS = 8

    print(f"Starting parallel processing with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the rows to the thread pool
        future_to_url = {executor.submit(process_url, row, predictor): row['url'] for _, row in df.iterrows()}

        for i, future in enumerate(as_completed(future_to_url)):
            record = future.result()
            if record:
                results.append(record)

            if i % 10 == 0:  # Print progress every 10 URLs
                print(f"Processed {i} / {len(df)} URLs...")

    # Save results to CSV
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"✅ Results saved to {result_path}")

except Exception as e:
    print(f"An error occurred during file processing: {e}")

except FileNotFoundError:
    print("❌ Check your file path! The CSV was not found.")
except KeyError:
    print("❌ The column 'url' doesn't seem to exist in this file.")



test_urls = [
    # -------------------------
    # ✅ BENIGN (Legitimate)
    # -------------------------
    "google.com",
    "facebook.com",
    "amazon.com",
    "wikipedia.org",
    "github.com",
    "microsoft.com",
    "apple.com",
    "linkedin.com",
    "stackoverflow.com",
    "netflix.com",

    # -------------------------
    # ⚠️ SUSPICIOUS / GREY
    # -------------------------
    "secure-login-update.com",
    "account-verification-center.net",
    "user-authentication-required.org",
    "update-your-bank-details.com",
    "verify-account-security-alert.com",
    "login-session-expired-warning.net",
    "reset-password-immediately.com",
    "confirm-your-identity-now.org",
    "billing-support-alert.com",
    "security-checkpoint-required.net",

    # -------------------------
    # ❌ PHISHING-LIKE (Highly suspicious)
    # -------------------------
    "paypal-login-security-update.com",
    "amazon-verification-alert-login.com",
    "google-account-reset-password.net",
    "facebook-security-checkpoint-login.com",
    "apple-id-verify-now-login.com",
    "netflix-billing-update-required.com",
    "bankofamerica-secure-login-alert.com",
    "chase-bank-verification-warning.net",
    "instagram-password-reset-confirm.com",
    "outlook-email-security-alert-login.com",

    # -------------------------
    # ❌ MALICIOUS STYLE (obfuscated)
    # -------------------------
    "http://192.168.0.1/login",
    "http://45.33.32.156/secure-login",
    "http://bit.ly/3xYzAbC",
    "http://tinyurl.com/update-account",
    "http://free-gift-card-win-now.com",
    "http://click-here-to-claim-prize.net",
    "http://urgent-action-required-login.com",
    "http://verify-now-or-account-locked.com",

    # -------------------------
    # ❌ EDGE CASES
    # -------------------------
    "localhost",
    "test.local",
    "example.com",
    "http://invalid-domain",
    "http://thisdoesnotexist123456.com"
]



# new_test_urls = [
#     #wrongly classified
#     # "www.w3schools.com",
#     # "https://www.shiksha.com/",
#     # "https://www.instagram.com/accounts/login/",
#     # "https://www.amazon.in/"
#
#     #correctly classified
#     # "https://en.wikipedia.org",
#     # "https://leetcode.com/problemset/",
#     # "https://mail.google.com",
#     # "https://classroom.google.com",
#     # "https://chatgpt.com/",
#     # "https://www.youtube.com/",
#     # "https://www.chess.com/",
#     # "https://www.heritageit.edu/",
#     # "https://in.bookmyshow.com/explore/home/kolkata",
#     "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
# ]
#
# results = []
# for url in new_test_urls:
#     result = predictor.predict(url)
#
#     row = {
#         "url": url,
#         "prediction": result["prediction"],
#         "confidence": result["confidence"],
#     }
#
#     for label, prob in result["probabilities"].items():
#         row[f"prob_{label}"] = prob
#
#     row['risk_level'] = get_risk(result["confidence"], result["prediction"])
#
#     results.append(row)
#     print("Processed Url: ", url)
#     print("Url Result : " , row)
#
# df = pd.DataFrame(results)
# df.to_csv("./TestResult/url_predictions_1.csv", index=False)