# 🛡️ WebDetector — AI-Powered Phishing & Malware Detection

<div align="center">

**Real-time URL threat detection powered by Graph Neural Networks (GNN)**

A Chrome extension + FastAPI backend that automatically scans every website you visit and alerts you if it's phishing, malware, or defacement — before you interact with it.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension%20MV3-4285F4?logo=googlechrome&logoColor=white)](#-chrome-extension-setup)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Feature Extraction Pipeline](#-feature-extraction-pipeline)
- [Model Pipeline](#-model-pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Backend](#-running-the-backend)
- [Chrome Extension Setup](#-chrome-extension-setup)
- [Using the Extension](#-using-the-extension)
- [API Reference](#-api-reference)
- [Deployment (Free Hosting)](#-deployment-free-hosting)
- [Performance](#-performance)
- [Contributing](#-contributing)

---

## 🔍 Overview

WebDetector is a **two-component system** that protects users from malicious websites in real time:

1. **Backend API** — A FastAPI server that extracts 33 features from a URL (lexical, content-based, and domain-based), runs them through a pre-classifier and a Graph Neural Network (GNN), and returns a threat verdict.

2. **Chrome Extension** — A Manifest V3 browser extension that intercepts every page navigation, sends the URL to the backend, and displays colour-coded badges, notifications, and a detailed analysis popup.

### What It Detects

| Threat Type | Description |
|---|---|
| ✅ **Benign** | Legitimate, safe websites |
| 🎣 **Phishing** | Fake sites impersonating real services to steal credentials |
| 🔨 **Defacement** | Websites that have been hacked and altered |
| 🦠 **Malware** | Sites distributing malicious software |

---

## ⚙️ How It Works

When you visit a website, the following happens automatically:

```
1. You navigate to a URL in Chrome
        ↓
2. Extension's background service worker detects the navigation
        ↓
3. URL is sent to the FastAPI backend via POST /predict
        ↓
4. Backend extracts 33 features in PARALLEL:
   ├── Lexical features (URL structure, entropy, keywords)
   ├── Content features (HTML forms, scripts, iframes)
   └── Domain features (WHOIS age, SSL, DNS records)
        ↓
5. Features → Pre-Classifier (Random Forest) → 4 probability scores
        ↓
6. Probabilities → Graph Neural Network (GraphSAGE) → Final prediction
        ↓
7. Result sent back to the extension
        ↓
8. Extension displays:
   ├── Colour-coded badge on the toolbar icon (🟢🟡🟠🔴)
   ├── Browser notification for HIGH/CRITICAL risk URLs
   └── Detailed popup with full analysis breakdown
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CHROME EXTENSION                        │
│                                                             │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────┐  │
│  │ background.js│──▶│  popup.html   │   │    icons/     │  │
│  │              │   │  popup.css    │   │  icon16.png   │  │
│  │ • Tab listen │   │  popup.js     │   │  icon48.png   │  │
│  │ • API calls  │   │              │   │  icon128.png  │  │
│  │ • Caching    │   │ • Verdict UI │   └───────────────┘  │
│  │ • Badges     │   │ • Prob bars  │                       │
│  │ • Alerts     │   │ • Features   │                       │
│  └──────┬───────┘   │ • History    │                       │
│         │           └───────────────┘                       │
└─────────┼───────────────────────────────────────────────────┘
          │ POST /predict
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    server.py                          │   │
│  │  FastGNNPredictor                                     │   │
│  │  ├── normalize_url()                                  │   │
│  │  ├── extract_features_parallel()  ◄── ThreadPool(3)   │   │
│  │  ├── get_risk_level()                                 │   │
│  │  └── predict()                                        │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              Feature Extraction Pipeline               │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │   Lexical   │ │   Content    │ │   Domain     │   │   │
│  │  │  (instant)  │ │ (HTTP fetch) │ │(WHOIS + SSL) │   │   │
│  │  │  15 features│ │ 10 features  │ │  8 features  │   │   │
│  │  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘   │   │
│  │         └───────────────┼────────────────┘            │   │
│  │                         ▼                              │   │
│  │              33 raw features (DataFrame)               │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Pre-Classifier (Random Forest)           │   │
│  │              33 features → 4 probabilities            │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           GNN Model (2-layer GraphSAGE)               │   │
│  │         4 probs → graph context → final verdict       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Feature Extraction Pipeline

The backend extracts **33 features** from every URL, grouped into three categories:

### Lexical Features (15) — Instant

Analysed directly from the URL string, no network requests needed.

| Feature | Description | Why It Matters |
|---|---|---|
| `url_length` | Total character count | Phishing URLs tend to be longer |
| `num_dots` | Count of `.` characters | Excessive dots suggest subdomain abuse |
| `num_slashes` | Count of `/` characters | Deep paths can hide malicious content |
| `num_digits` | Count of numeric characters | IP-based URLs use many digits |
| `num_special_chars` | Non-alphanumeric characters | Obfuscation indicator |
| `has_https` | Whether URL uses HTTPS | Phishing sites often skip HTTPS |
| `has_ip` | URL uses an IP address instead of domain | Strong phishing signal |
| `has_at_symbol` | Contains `@` character | Used to trick URL parsers |
| `has_dash` | Contains `-` character | Common in spoofed domains |
| `subdomain_count` | Number of subdomains | Excessive subdomains = suspicious |
| `entropy` | Shannon entropy of the URL | Random-looking URLs are suspicious |
| `suspicious_keywords` | Count of phishing keywords (login, secure, verify, etc.) | Keyword stuffing detection |
| `brand_similarity` | Levenshtein distance to popular brand names | Detects typosquatting |
| `tld_risk_score` | Whether TLD is commonly abused (.tk, .ml, .xyz, etc.) | Risky TLD detection |
| `path_length` | Number of path segments | Deep nesting can be suspicious |

### Content Features (10) — Requires HTTP Fetch

Fetches the actual HTML of the page and analyses its structure.

| Feature | Description | Why It Matters |
|---|---|---|
| `num_forms` | Count of `<form>` elements | Phishing pages have credential forms |
| `num_iframes` | Count of `<iframe>` elements | Used to embed hidden malicious content |
| `num_anchors` | Count of `<a>` link elements | Legitimate sites have more links |
| `num_scripts` | Count of `<script>` elements | Excessive scripts may be malicious |
| `has_login_form` | Detected username + password fields | Strong phishing indicator |
| `external_link_ratio` | Ratio of external to total links | Phishing sites link externally |
| `has_redirect` | Meta refresh redirect detected | Redirect-based attacks |
| `input_fields` | Total `<input>` elements | Many inputs = data harvesting |
| `password_fields` | Count of password-type inputs | Credential theft indicator |
| `js_obfuscation_score` | Detection of eval(), atob(), encoded JS | Code obfuscation = hiding intent |

### Domain Features (8) — Requires WHOIS + SSL Checks

Checks domain registration, DNS, and SSL certificate data.

| Feature | Description | Why It Matters |
|---|---|---|
| `domain_age` | Days since domain registration | New domains are suspicious |
| `time_to_expiry` | Days until domain expires | Short-lived domains are suspicious |
| `has_whois` | WHOIS record exists | Legitimate sites have WHOIS data |
| `has_dns_record` | DNS resolves successfully | Dangling DNS = suspicious |
| `ssl_valid` | Valid SSL certificate present | No SSL = untrusted |
| `cert_validity_days` | Duration of SSL cert (days) | Short certs can be suspicious |
| `domain_reputation` | TLD reputation check | Known risky TLDs |
| `hosting_country_risk` | Hosted in high-risk region (.ru, .cn, .kp) | Geo-based risk assessment |

---

## 🤖 Model Pipeline

### Stage 1: Pre-Classifier (Random Forest)

- Takes the **33 raw features** as input
- Outputs **4 probability scores** (one for each class: benign, phishing, defacement, malware)
- Acts as a **dimensionality reducer** from 33 → 4 semantic features
- Model file: `Helper/PreClassifier/final_result_pre_classifier.pk1`

### Stage 2: Graph Neural Network (GraphSAGE)

- Uses the **4 probability scores** as node features
- Builds a **k-nearest-neighbours graph** (k=5) using existing labelled data
- The new URL is added as a node to the graph, and edges are rebuilt
- **2-layer GraphSAGE** with 32 hidden dimensions propagates information across neighbouring nodes
- The graph context helps the model make better predictions by considering **similar URLs**
- Model file: `Helper/GnnModels/final_gnn_model_2.pth`
- Graph data: `Helper/GraphData/final_result_graph_data_1.pt`

### Why GNN?

Traditional classifiers treat each URL independently. The GNN leverages **graph structure** — if a new URL has features similar to known phishing URLs in the graph, the message-passing mechanism amplifies that signal. This makes the model more robust to adversarial URLs that might fool a standalone classifier.

---

## 📁 Project Structure

```
WebDetector/
├── server.py                          # FastAPI backend (run this)
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # For Hugging Face Spaces deployment
├── README.md                          # This file
├── calc.py                            # Model evaluation metrics
├── test_api.py                        # Original batch prediction script
│
├── Helper/
│   ├── FeaturePipeline/
│   │   ├── feature_pipeline.py        # Main feature extraction orchestrator
│   │   ├── logical_helper.py          # Lexical analysis functions
│   │   ├── content_helper.py          # HTML content analysis functions
│   │   └── domain_helper.py           # WHOIS, SSL, DNS analysis functions
│   │
│   ├── GnnModelPipeline/
│   │   └── gnn_training.py            # GNN model architecture (GraphSAGE)
│   │
│   ├── GnnModels/
│   │   └── final_gnn_model_2.pth      # Trained GNN weights
│   │
│   ├── GraphData/
│   │   └── final_result_graph_data_1.pt  # Base graph dataset
│   │
│   ├── PreClassifier/
│   │   ├── final_result_pre_classifier.pk1  # Trained Random Forest
│   │   └── selected_features.pkl      # Feature column order
│   │
│   ├── DataSet/                       # Training data
│   ├── NewDataSet/                    # Additional data
│   └── ResultDataSet/                 # Prediction outputs
│
└── extension/                         # Chrome Extension (Manifest V3)
    ├── manifest.json                  # Extension configuration
    ├── background.js                  # Service worker (tab listener, API calls)
    ├── popup.html                     # Popup UI structure
    ├── popup.css                      # Dark theme styles
    ├── popup.js                       # Popup rendering logic
    └── icons/
        ├── icon16.png                 # Toolbar icon
        ├── icon48.png                 # Extension page icon
        └── icon128.png                # Chrome Web Store icon
```

---

## 📦 Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **Google Chrome** — [Download](https://www.google.com/chrome/)
- **Git** — [Download](https://git-scm.com/)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/WebDetector.git
cd WebDetector
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `torch` | PyTorch for the GNN model |
| `torch-geometric` | Graph neural network layers (GraphSAGE) |
| `scikit-learn` | Pre-classifier (Random Forest) + KNN graph |
| `pandas` / `numpy` | Data processing |
| `joblib` | Model serialisation |
| `requests` | HTTP fetching for content analysis |
| `beautifulsoup4` | HTML parsing |
| `python-whois` | WHOIS domain lookups |
| `fastapi` | Web framework for the API |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation |

> **Note on PyTorch Geometric:** If `pip install torch-geometric` fails, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your specific PyTorch + CUDA version.

---

## ▶️ Running the Backend

### Start the Server

```bash
python server.py
```

The server starts at `http://localhost:8000`.

### Verify It's Running

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"google.com"}'
```

### Interactive API Docs

FastAPI provides automatic interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🌐 Chrome Extension Setup

### Step 1: Open Chrome Extensions Page

Navigate to `chrome://extensions/` in your browser.

### Step 2: Enable Developer Mode

Toggle the **"Developer mode"** switch in the top-right corner.

### Step 3: Load the Extension

1. Click **"Load unpacked"**
2. Browse to and select the `extension/` folder inside the WebDetector project:
   ```
   WebDetector/extension/
   ```
3. The WebDetector shield icon appears in your toolbar

### Step 4: Pin the Extension (Optional)

Click the puzzle piece icon (🧩) in Chrome's toolbar and **pin** WebDetector for easy access.

> **Important:** The backend server must be running for the extension to work. Start it with `python server.py` before browsing.

---

## 🎯 Using the Extension

### Automatic Scanning

Once installed, the extension **automatically scans every URL** you navigate to. No manual action needed.

### Badge Indicators

The extension icon shows a colour-coded badge:

| Badge | Risk Level | Meaning |
|---|---|---|
| 🟢 `✓` | **Safe** | Benign URL with high confidence |
| 🟡 `!` | **Low** | Likely safe but low confidence |
| 🟠 `!!` | **Medium** | Suspicious characteristics detected |
| 🔴 `⚠` | **High** | Likely phishing/malware |
| 🔴 `⛔` | **Critical** | Confirmed malicious with high confidence |
| 🟣 `…` | **Scanning** | Analysis in progress |
| ⚫ `?` | **Error** | Backend unreachable |

### Browser Notifications

For **high** and **critical** risk URLs, the extension automatically fires a browser notification alert — even if the popup is closed.

### Popup Panel

Click the extension icon to open the detailed analysis popup:

- **Threat Verdict** — Colour-coded badge showing SAFE / SUSPICIOUS / DANGEROUS / CRITICAL
- **Confidence Score** — Animated percentage gauge showing model confidence
- **Threat Breakdown** — Horizontal probability bars for all 4 classes (benign, phishing, defacement, malware)
- **Detection Methods** — Collapsible section showing all 33 extracted features grouped by:
  - Lexical Analysis (URL structure)
  - Content Analysis (HTML inspection)
  - Domain Analysis (WHOIS, SSL, DNS)
- **Recent Scans** — History of the last 20 scanned URLs with their verdicts

### Re-scanning

Click the refresh button (↻) in the popup header to force a re-scan of the current page. This clears the cache and runs a fresh analysis.

### Caching

Results are cached for **5 minutes** per URL to avoid redundant API calls when switching between tabs. After 5 minutes, the URL is automatically re-scanned on the next visit.

---

## 📡 API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Analyse a URL for threats.

**Request:**
```json
{
  "url": "example.com"
}
```

**Response:**
```json
{
  "url": "http://example.com",
  "prediction": "benign",
  "confidence": 0.8534,
  "risk_level": "safe",
  "probabilities": {
    "benign": 0.8534,
    "phishing": 0.0021,
    "defacement": 0.1203,
    "malware": 0.0242
  },
  "features": {
    "lexical": {
      "url_length": 19,
      "num_dots": 1,
      "has_https": 0,
      "has_ip": 0,
      "entropy": 3.45,
      "suspicious_keywords": 0
    },
    "content": {
      "num_forms": 0,
      "has_login_form": 0,
      "js_obfuscation_score": 0
    },
    "domain": {
      "domain_age": 10455,
      "ssl_valid": 1,
      "has_whois": 1
    }
  },
  "analysis_time": 4.21
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `url` | string | Normalised URL that was analysed |
| `prediction` | string | Predicted class: `benign`, `phishing`, `defacement`, or `malware` |
| `confidence` | float | Model confidence (0.0 – 1.0) |
| `risk_level` | string | Risk category: `safe`, `low`, `medium`, `high`, or `critical` |
| `probabilities` | object | Probability for each class |
| `features` | object | Raw feature values grouped by category |
| `analysis_time` | float | Time taken for analysis in seconds |

---

## ☁️ Deployment (Free Hosting)

### Hugging Face Spaces (Recommended — Free)

The project includes a `Dockerfile` pre-configured for [Hugging Face Spaces](https://huggingface.co/spaces).

#### Step 1: Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Enter a name (e.g., `WebDetector`)
3. Select **Docker** as the SDK
4. Choose **Blank** template
5. Click **Create Space**

#### Step 2: Push Your Code

```bash
# Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/WebDetector

# Push everything
git add .
git commit -m "Deploy WebDetector to HF Spaces"
git push hf main
```

#### Step 3: Wait for Build

Hugging Face will build the Docker image and start the server. This takes 5–10 minutes on the first deploy.

Your API will be available at:
```
https://YOUR_USERNAME-webdetector.hf.space
```

#### Step 4: Update the Extension

Open `extension/background.js` and change the API URL:

```javascript
// Change this line:
const API_BASE = "http://localhost:8000";

// To your Hugging Face Space URL:
const API_BASE = "https://YOUR_USERNAME-webdetector.hf.space";
```

Then reload the extension in `chrome://extensions/`.

> **Note:** Free Hugging Face Spaces sleep after ~15 minutes of inactivity. The first request after sleep takes ~30 seconds (cold start). Subsequent requests are fast.

---

## ⚡ Performance

### Analysis Speed

Feature extraction runs in **parallel** using `ThreadPoolExecutor` with 3 workers:

| Stage | Sequential | Parallel |
|---|---|---|
| Lexical features | 0.01s | ↘ |
| Content features (HTTP fetch) | 2–5s | → **All run concurrently** |
| Domain features (WHOIS + SSL) | 5–10s | ↗ |
| Pre-classifier + GNN inference | 0.05s | 0.05s |
| **Total** | **~15s** | **~4–6s** |

### Caching

The extension caches results for 5 minutes per URL, so:
- Switching between tabs is **instant**
- Re-visiting the same URL within 5 minutes shows cached results
- Maximum 100 URLs cached in memory

---

## 🤝 Contributing

Contributions are welcome! Some ideas:

- [ ] Add more phishing URL datasets for training
- [ ] Implement URL reputation API integration (VirusTotal, Google Safe Browsing)
- [ ] Add a settings page to configure the API URL from within the extension
- [ ] Support Firefox (WebExtension API is similar)
- [ ] Add batch URL scanning endpoint
- [ ] Improve cold-start time on Hugging Face Spaces

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using PyTorch, GraphSAGE, FastAPI, and Chrome Extension APIs**

</div>
