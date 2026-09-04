# Risk Manager: AI-Powered Return Fraud Detection

![Risk Manager](https://img.shields.io/badge/Track%202-AI%20Risk%20Manager-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

A defense-only, production-grade architecture AI Risk Manager (with demo-scale validation) designed to stop e-commerce return fraud (empty-box, substitution, wardrobing) through a combination of behavioral tabular scoring and visual verification.

Built for **Razorpay AI Buildathon 2026, Track 2 (AI Risk Manager)**.

> 💡 **Tip for Reviewers:** To view the video or report in a new tab without losing this page, please **Cmd + Click** (For Mac) or **Ctrl + Click** (For Windows) the buttons below.

<br>

[![Watch Pitch Video](https://img.shields.io/badge/▶_Watch_Pitch_Video-1FA463?style=for-the-badge&logo=youtube&logoColor=white)](#)&emsp;[![Read Full Report](https://img.shields.io/badge/📄_Read_Full_Report-1FA463?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1s4DQofHLi_u4k6jsi0vZ1CHTomNWSiEG/view?usp=sharing)
<br>

## 🚀 Quick Start (Demo)

This project includes a fully functional FastAPI backend and a professional, single-page interactive dashboard.

### Option 1: Docker (Highly Recommended)
Because the vision models are heavy (DINOv2) and require downloading weights, the Docker build automatically bakes the weights directly into the image. Once built, this guarantees the application will work perfectly even if the judging sandbox is completely disconnected from the internet at runtime.

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/Jeypraveen/Risk-Manager.git
   cd Risk-Manager
   ```
2. Build and run the entire stack (Backend + UI) with one command:
   ```bash
   docker-compose up --build
   ```
3. Open `http://localhost:8000/demo/index.html` in your web browser!

### Option 2: Native (Python) 

1. **Clone & Setup Environment:**
   ```bash
   git clone https://github.com/Jeypraveen/Risk-Manager.git
   cd Risk-Manager
   python -m venv venv
   ```

2. **Activate and Install:**
   ```bash
   # On Windows:
   .\venv\Scripts\Activate
   # On Mac/Linux:
   source venv/bin/activate

   # Install CPU PyTorch first (to avoid massive CUDA downloads)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   
   # Install remaining requirements
   pip install -r requirements.txt
   ```

3. **Generate Data and Train Models:**
   Because model checkpoints are not checked into Git, you must generate synthetic data and train the models before starting the server.
   ```bash
   python -m data.generate_data
   python -m src.tabular.train
   python -m scripts.train_meta_learner
   ```

4. **Start the API Server:**

   > ⚠️ **Important Note for Offline Reviewers:** The first time the API server runs, the `transformers` library will automatically download the 86MB DINOv2 model weights from the HuggingFace Hub. Ensure you have an active internet connection for the initial run. If your judging sandbox is fully disconnected from the internet, we recommend using **Option 1: Docker** above.

   ```bash
   python src/api/server.py
   ```

5. **Open the Demo UI:**
   Open `http://localhost:8000/demo/index.html` in any web browser.

## 🖼️ Sample Test Images

The repository includes pre-staged sample images so the vision pipeline can be tested immediately, without needing to source your own photos:

*   `data/images/catalog/` - reference product photos (e.g. iPhone, sneakers, headphones) representing what was originally ordered.
*   `data/images/returns/` - corresponding return photos, including genuine matches, blurry/damaged items, and clear substitution/fraud cases (e.g. an unrelated object returned in place of the ordered item).
*   `data/images/image_mapping.csv` - maps each catalog image to its intended return-image pairing and labels the fraud subtype (`none`, `substitution`, `empty_box`), so you can look up which pairing demonstrates which decision path before testing.

In the demo UI, use the **Catalog Reference** and **Customer Return** upload boxes under "Vision Verification" to pair any catalog image with any return image and see the Trust Score respond in real time. Pairing a catalog image with its matching return photo (per the mapping above) demonstrates the auto-approve path; pairing it with a mismatched or substituted image demonstrates the manual-review path.

> **Note on image sourcing:** Catalog reference photos and return-photo staging (damaged items, empty boxes, substitutions) are fully generic or AI-generated specifically for this project, containing no real-world brand trademarks or logos.

## 🧠 System Architecture

This system uses a **Late-Fusion Meta-Learner** architecture to ensure graceful degradation when photos are missing or vision services fail.

1. **Tabular Scorer (LightGBM):** Analyzes behavioral metadata (account age, return velocity, COD status, etc.).
2. **Vision Pipeline (DINOv2 + Heuristics):** Analyzes return photos for substitution and empty-box fraud.
3. **Circuit Breaker:** Wraps the vision pipeline to gracefully handle image upload timeouts or model checkpoint mismatches.
4. **Meta-Learner:** Fuses the tabular and vision signals based on `modality_confidence`.
5. **Three-Way Router:** Routes the final trust score to `AUTO_APPROVE`, `NUDGE`, or `MANUAL_REVIEW`.
6. **Audit Trail:** Logs every decision, score, and failure for compliance.

Read the full details in [docs/architecture.md](docs/architecture.md).

## 📊 Honest Metrics & Boundaries

As per the prompt guidelines, we present honest metrics that consider false-positive costs. The system was evaluated on a held-out test set of 2,000 synthetic return requests with an 8% base fraud rate.

*   **Test PR-AUC (Tabular):** `0.447`
*   **Test PR-AUC (Fusion):** `0.876`*
*   **Cost-Optimal Threshold (Fusion):** `0.778` (Precision: `0.879`, Recall: `0.769`)

> **Note on Fusion Evaluation:** Both the Fusion PR-AUC and the cost-optimal threshold/precision/recall above are evaluated using simulated vision scores that are statistically representative of the pipeline's expected behavior - they do not reflect a full end-to-end evaluation of the actual DINOv2 pipeline on real images across the full test set. The *real* vision pipeline (DINOv2 similarity, empty-box detection) was separately and directly verified on the project's staged image set - see the [Validity Boundaries](evaluation/results/validity_boundaries.md) document for the real, measured similarity scores and what they prove.

For full transparency regarding what these synthetic metrics *do* and *do not* prove, please read our [Validity Boundaries](evaluation/results/validity_boundaries.md) document.

## 🛡️ Strict Defense-Only

This system adheres strictly to the "defense-only" requirement:
*   **No Auto-Denials:** The system *never* automatically denies a return. The lowest trust scores are safely routed to `MANUAL_REVIEW` for human intervention.
*   **No Attack Vector Generation:** We do not employ GANs or adversarial image detection, as building those models requires training offensive capabilities.
*   **Generic Nudges:** When requesting photos, the system uses generic messaging so fraudsters cannot reverse-engineer the detection logic.

## 📁 Repository Structure

*   `src/`: Core logic (config, tabular, vision, fusion, recovery, api, audit).
*   `data/`: Synthetic data generator (`generate_data.py`) and staged test images.
*   `evaluation/`: Evaluation scripts and resulting plots.
*   `demo/`: HTML/CSS/JS for the interactive dashboard.
*   `docs/`: Additional architecture documentation.
*   `models/`: Saved joblib checkpoints for LightGBM and Logistic Regression (generated by training scripts, not checked into Git).

## 👤 Author

*   **Jey Praveen Sivaraj** - Razorpay AI Buildathon 2026
