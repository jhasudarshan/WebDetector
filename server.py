import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from Helper.FeaturePipeline.feature_pipeline import (
    extract_lexical_features,
    advanced_lexical_features,
    extract_content_features,
    extract_domain_features,
)
from Helper.GnnModelPipeline.gnn_training import GNNModel
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import traceback

# ─── FastAPI App ───────────────────────────────────────────────
app = FastAPI(title="WebDetector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────
class PredictRequest(BaseModel):
    url: str


class PredictResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    risk_level: str
    probabilities: dict
    features: dict
    analysis_time: float


# ─── Optimised Predictor ───────────────────────────────────────
class FastGNNPredictor:
    """
    Same GNN pipeline as test_api.py but with:
    - Parallel feature extraction (lexical + content + domain run concurrently)
    - Feature breakdown returned for the extension UI
    - Risk-level classification
    """

    def __init__(self, model_path, base_graph_path, pre_classifier_path, feature_path):
        # Load base graph
        self.base_data = torch.load(base_graph_path, weights_only=False)

        # GNN model
        self.model = GNNModel(
            input_dim=self.base_data.x.shape[1],
            hidden_dim=32,
            num_classes=4,
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Pre-classifier
        self.pre_model = joblib.load(pre_classifier_path)
        self.selected_features = joblib.load(feature_path)

        self.labels = ["benign", "phishing", "defacement", "malware"]

    # ── URL normalisation ──────────────────────────────────────
    @staticmethod
    def normalize_url(url: str) -> str | None:
        if not isinstance(url, str):
            return None
        url = url.strip()
        if not url:
            return None
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url

    # ── Parallel feature extraction (speed boost) ──────────────
    def extract_features_parallel(self, url: str):
        """
        Run lexical (instant), content (HTTP fetch), and domain (WHOIS/SSL)
        in parallel using threads → ~5s instead of ~15s sequential.
        Returns (feature_tensor, raw_features_dict).
        """
        results = {}

        def _lexical():
            lex = extract_lexical_features(url)
            adv = advanced_lexical_features(url)
            return {**lex, **adv}

        def _content():
            return extract_content_features(url)

        def _domain():
            return extract_domain_features(url)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_lexical): "lexical",
                pool.submit(_content): "content",
                pool.submit(_domain): "domain",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result(timeout=10)
                except Exception:
                    # Return safe defaults if a group fails
                    results[key] = {}

        # Merge all features into one dict
        all_features = {
            **results.get("lexical", {}),
            **results.get("content", {}),
            **results.get("domain", {}),
        }

        # Convert to DataFrame → align → pre-classifier probabilities
        df = pd.DataFrame([all_features])
        df.fillna(-1, inplace=True)
        if "url_length" in df.columns:
            df["url_length"] = df["url_length"].clip(0, 2000)
        df = df.reindex(columns=self.selected_features, fill_value=0)
        probs = self.pre_model.predict_proba(df)
        tensor = torch.tensor(probs, dtype=torch.float)

        return tensor, results

    # ── Risk-level mapping ─────────────────────────────────────
    @staticmethod
    def get_risk_level(prediction: str, confidence: float) -> str:
        if prediction == "benign":
            if confidence >= 0.8:
                return "safe"
            return "low"
        elif prediction == "phishing":
            if confidence >= 0.7:
                return "critical"
            return "high"
        elif prediction in ("defacement", "malware"):
            if confidence >= 0.6:
                return "critical"
            return "high"
        return "medium"

    # ── Main prediction ────────────────────────────────────────
    def predict(self, url: str) -> dict:
        start = time.time()

        url = self.normalize_url(url)
        if not url:
            return {"error": "Invalid URL"}

        new_x, raw_features = self.extract_features_parallel(url)

        # Clone graph, append new node, rebuild edges
        data = self.base_data.clone()
        data.x = torch.cat([data.x, new_x], dim=0)
        data.edge_index = self._rebuild_edges(data.x.numpy(), k=5)

        with torch.no_grad():
            out = self.model(data)

        probs = torch.softmax(out[-1], dim=0).numpy()
        pred_class = int(np.argmax(probs))
        prediction = self.labels[pred_class]
        confidence = float(probs[pred_class])

        elapsed = round(time.time() - start, 2)

        return {
            "url": url,
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": self.get_risk_level(prediction, confidence),
            "probabilities": {
                label: round(float(p), 4)
                for label, p in zip(self.labels, probs)
            },
            "features": raw_features,
            "analysis_time": elapsed,
        }

    @staticmethod
    def _rebuild_edges(X, k=5):
        A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
        edge_index = A.nonzero()
        return torch.tensor(edge_index, dtype=torch.long)


# ─── Initialise predictor at startup ──────────────────────────
predictor = FastGNNPredictor(
    model_path="./Helper/GnnModels/final_gnn_model_2.pth",
    base_graph_path="./Helper/GraphData/final_result_graph_data_1.pt",
    pre_classifier_path="./Helper/PreClassifier/final_result_pre_classifier.pk1",
    feature_path="./Helper/PreClassifier/selected_features.pkl",
)


# ─── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predictor.predict(req.url)
        if "error" in result:
            return {"error": result["error"]}
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "url": req.url,
            "prediction": "error",
            "confidence": 0,
            "risk_level": "unknown",
            "probabilities": {},
            "features": {"error": str(e)},
            "analysis_time": 0,
        }


# ─── Entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
