# 🩺 DermViT — Vision Transformer Skin Disease Diagnostic System

<p align="center">
  <img src="https://img.shields.io/badge/Model-Vision_Transformer_(ViT--B16)-E3A5AF?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/Classes-22_Skin_Pathologies-10b981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Inference-Pure_NumPy_(Zero_GPU)-6366f1?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Dark_%26_Light_Mode-FBBF24?style=for-the-badge" />
</p>

An intelligent, state-of-the-art dermatopathology classification platform powered by Google's **Vision Transformer (ViT-B16)**. Designed with a luxury clinical UI, dynamic dark/light theming, real-time optical visual biomarker extraction, and persistent 25-record scan history.

---

## 🌟 Key Capabilities

- **22 Distinct Skin Disease Pathologies**: Comprehensive classification covering Melanoma, Basal Cell Carcinoma, Actinic Keratosis, Tinea, Eczema, Psoriasis, Cutaneous Lupus, Vitiligo, and more.
- **Pure NumPy ViT Engine**: Forward pass executed with optimized pure NumPy matrix operations (~800ms CPU latency). Zero CUDA, PyTorch, or TensorFlow runtime dependencies needed for inference.
- **4 Optical Visual Biomarkers**: Real-time extraction of:
  - **Vascular Erythema**: Quantitative redness and vascular dilation index.
  - **Border Architecture**: Boundary contrast and edge regularity profiling.
  - **Surface Topology**: Epidermal textural standard deviation and scaling analysis.
  - **Pigment Geometry**: Melanin distribution and chromatic spatial asymmetry.
- **Streamlined Clinical Insights**:
  - **Yeh Kya Hai? (Clinical Overview)**: Concise clinical explanation of the condition.
  - **Kiski Wajah Se Hoti Hai? (Causes & Pathological Mechanism)**: Underlying biological etiology, triggers, and pathophysiological process.
- **Diagnostic Scan History (25 Records)**: Automatically logs the last 25 scans with exact dates, timestamps, and image thumbnails in local storage. One-click reload to review past scans.
- **Dual Luxury Theme (Light & Dark Mode)**:
  - *Light Mode*: Soothing, warm skin-tone palette (Nude `#F5E6E8`, Beige `#E8C7EA`, Soft Rose Gold `#E3A5AF`, Deep Charcoal `#1E293B`).
  - *Dark Mode*: Ultra-sleek Velvet Obsidian (`#090D16`) with dark frosted glass cards and glowing blooms.

---

## 🗂️ 22 Classified Skin Pathologies

| Code | Disease / Pathology | Risk Category |
|---|---|---|
| **MEL** | Malignant Melanoma | Critical Risk |
| **BCC** | Basal Cell Carcinoma | High Risk |
| **AKIEC** | Actinic Keratosis | Moderate Risk |
| **LUPUS** | Cutaneous Lupus | High Risk |
| **VASCULITIS** | Cutaneous Vasculitis | High Risk |
| **BULLOUS** | Bullous Disease (Pemphigoid) | High Risk |
| **TINEA** | Tinea (Ringworm / Dermatophytosis) | Low Risk |
| **CANDIDIASIS** | Cutaneous Candidiasis | Low Risk |
| **ECZEMA** | Atopic Dermatitis / Eczema | Low Risk |
| **PSORIASIS** | Plaque Psoriasis | Moderate Risk |
| **ROSACEA** | Acne Rosacea | Low Risk |
| **ACNE** | Acne Vulgaris | Low Risk |
| **LICHEN** | Lichen Planus | Moderate Risk |
| **NV** | Melanocytic Nevus (Common Mole) | Benign / Low Risk |
| **BKL** | Seborrheic Keratosis | Benign / Low Risk |
| **BENIGN_TUMOR** | Benign Cutaneous Neoplasms | Benign / Low Risk |
| **VASC** | Vascular Tumors & Hemangiomas | Benign / Low Risk |
| **VITILIGO** | Vitiligo (Depigmentation) | Low Risk |
| **WARTS** | Verruca Vulgaris (HPV) | Low Risk |
| **BITES** | Arthropod Bites & Scabies | Low Risk |
| **PHOTO_DAMAGE** | Solar Elastosis & Photoaging | Low Risk |
| **NORMAL** | Normal / Healthy Skin | Normal |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yatish-debug/vit-skin-disease.git
cd vit-skin-disease
```

### 2. Run the Application
Double-click `start_webapp.bat` or run via terminal:

```bash
# Activate your python virtual environment
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

*(Note: If `model.safetensors` is not present locally, the system automatically downloads the model weights from Hugging Face on the first run).*

---

## 📦 Project Structure

```
DermViT/
├── models/
│   └── vit-skin-disease/
│       ├── config.json              # ViT-Base architecture configuration
│       └── preprocessor_config.json # Image normalization parameters
├── webapp/
│   ├── app.py                       # FastAPI application & REST endpoints
│   ├── clinical_knowledge.py        # 22-class pathology knowledge base & biomarker extractor
│   ├── hf_vit_model.py              # Pure NumPy Vision Transformer inference engine
│   └── static/
│       ├── index.html               # Clean single-page application UI
│       ├── style.css                # Luxury skin-tone & Obsidian Dark theme styling
│       └── app.js                   # Client controller, theme toggle, and 25-record history
├── requirements.txt                 # Python dependencies
├── start_webapp.bat                 # One-click Windows starter script
└── README.md
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
