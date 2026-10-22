"""
Hugging Face Vision Transformer (ViT) Inference Engine
Model: LaurianeMD/vit-skin-disease
Pre-trained on 22 distinct skin disease categories.
Implements the full PyTorch ViT-Base forward pass in high-performance NumPy/SciPy.
Zero PyTorch/CUDA overhead.
"""

import os
import json
import numpy as np
import scipy.special
from PIL import Image
from safetensors.numpy import load_file

# Directory where LaurianeMD/vit-skin-disease is stored
MODEL_DIR = os.path.join("models", "vit-skin-disease")

# 22 Skin Condition Labels & Clinical Metadata
DISEASE_METADATA = {
    0: {
        "code": "ACNE",
        "name": "Acne",
        "fullName": "Acne Vulgaris",
        "type": "Inflammatory",
        "severity": "Low Risk",
        "badgeColor": "#3b82f6",
        "description": "Common chronic inflammation of the pilosebaceous units leading to comedones, papules, pustules, and occasional nodules.",
        "action": "Maintain gentle skin hygiene, topical benzoyl peroxide or retinoids. Consult dermatologist if cystic."
    },
    1: {
        "code": "AKIEC",
        "name": "Actinic Keratosis",
        "fullName": "Actinic Keratosis (Pre-Malignant)",
        "type": "Pre-Malignant",
        "severity": "Moderate Risk",
        "badgeColor": "#f59e0b",
        "description": "Rough, scaly precancerous patch induced by long-term ultraviolet (UV) radiation. Has potential to transform into squamous cell carcinoma.",
        "action": "Dermatological evaluation recommended for cryotherapy or topical 5-FU therapy."
    },
    2: {
        "code": "BENIGN_TUMOR",
        "name": "Benign Tumors",
        "fullName": "Benign Cutaneous Neoplasms",
        "type": "Benign",
        "severity": "Low Risk",
        "badgeColor": "#10b981",
        "description": "Non-cancerous tissue proliferations such as skin tags, lipomas, or fibromas with benign histological architecture.",
        "action": "Routine observation. Removal optional for cosmetic or irritation reasons."
    },
    3: {
        "code": "BULLOUS",
        "name": "Bullous Disease",
        "fullName": "Bullous Pemphigoid / Pemphigus",
        "type": "Autoimmune / Blistering",
        "severity": "High Risk",
        "badgeColor": "#ef4444",
        "description": "Rare autoimmune subepidermal blistering condition causing tense, fluid-filled bullae and severe cutaneous erosions.",
        "action": "Immediate clinical dermatologist visit and autoimmune serology/biopsy required."
    },
    4: {
        "code": "CANDIDIASIS",
        "name": "Candidiasis",
        "fullName": "Cutaneous Candidiasis",
        "type": "Fungal Infection",
        "severity": "Low Risk",
        "badgeColor": "#8b5cf6",
        "description": "Superficial fungal infection caused by Candida albicans, typically flourishing in warm, moist intertriginous folds.",
        "action": "Keep area dry; topical antifungal creams (clotrimazole, miconazole) usually effective."
    },
    5: {
        "code": "DRUG_ERUPTION",
        "name": "Drug Eruption",
        "fullName": "Adverse Cutaneous Drug Reaction",
        "type": "Allergic / Toxic",
        "severity": "Moderate Risk",
        "badgeColor": "#f97316",
        "description": "Cutaneous manifestation of an immunological or toxic reaction to ingested or administered pharmaceuticals.",
        "action": "Review recent medication history with your physician. Discontinue suspect drug under medical supervision."
    },
    6: {
        "code": "ECZEMA",
        "name": "Eczema",
        "fullName": "Atopic Dermatitis / Eczema",
        "type": "Inflammatory",
        "severity": "Low Risk",
        "badgeColor": "#06b6d4",
        "description": "Pruritic, erythematous, dry inflammation of the epidermal barrier associated with immune dysregulation.",
        "action": "Intensive skin barrier moisturization, avoid triggers, short-course mild topical corticosteroids if inflamed."
    },
    7: {
        "code": "BITES",
        "name": "Infestations & Bites",
        "fullName": "Arthropod Bite / Scabies Reaction",
        "type": "Parasitic / Bite",
        "severity": "Low Risk",
        "badgeColor": "#84cc16",
        "description": "Localized inflammatory response caused by insect bites, scabies mites, or environmental arthropods.",
        "action": "Antihistamines for pruritus, topical calamine or antiparasitic treatment (permethrin) if scabies suspected."
    },
    8: {
        "code": "LICHEN",
        "name": "Lichen Planus",
        "fullName": "Lichen Planus / Sclerosus",
        "type": "Inflammatory",
        "severity": "Moderate Risk",
        "badgeColor": "#a855f7",
        "description": "Chronic inflammatory autoimmune disease characterized by polygonal, violaceous, flat-topped pruritic papules.",
        "action": "Dermatology assessment for potent topical corticosteroid therapy and mucosal screening."
    },
    9: {
        "code": "LUPUS",
        "name": "Cutaneous Lupus",
        "fullName": "Discoid / Subacute Cutaneous Lupus",
        "type": "Autoimmune",
        "severity": "High Risk",
        "badgeColor": "#ec4899",
        "description": "Autoimmune condition featuring photo-distributed erythema, scaling plaques, and potential scarring or systemic involvement.",
        "action": "Comprehensive rheumatology and dermatology workup; strict photoprotection and antimalarial therapy."
    },
    10: {
        "code": "NV",
        "name": "Melanocytic Moles",
        "fullName": "Melanocytic Nevus (Common Mole)",
        "type": "Benign",
        "severity": "Low Risk",
        "badgeColor": "#10b981",
        "description": "Benign proliferation of cutaneous melanocytes resulting in symmetrical pigmented macules or papules.",
        "action": "Routine periodic self-monitoring using the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution)."
    },
    11: {
        "code": "PSORIASIS",
        "name": "Psoriasis",
        "fullName": "Plaque Psoriasis",
        "type": "Autoimmune",
        "severity": "Moderate Risk",
        "badgeColor": "#f59e0b",
        "description": "T-cell mediated autoimmune dermatosis featuring well-demarcated erythematous plaques covered by silvery micaceous scales.",
        "action": "Targeted topical treatments, vitamin D analogues, phototherapy, or systemic biologics under dermatologist care."
    },
    12: {
        "code": "ROSACEA",
        "name": "Rosacea",
        "fullName": "Acne Rosacea",
        "type": "Vascular / Inflammatory",
        "severity": "Low Risk",
        "badgeColor": "#fb7185",
        "description": "Chronic central facial vascular hyperreactivity causing persistent flushing, telangiectasia, and papulopustules.",
        "action": "Avoid thermal and dietary triggers, use gentle mineral sunscreen, topical metronidazole or ivermectin."
    },
    13: {
        "code": "BKL",
        "name": "Seborrheic Keratosis",
        "fullName": "Seborrheic Keratosis",
        "type": "Benign",
        "severity": "Low Risk",
        "badgeColor": "#10b981",
        "description": "Extremely common benign epidermal tumor presenting as a 'stuck-on', waxy, hyperpigmented or verrucous lesion.",
        "action": "Reassuringly benign. No mandatory intervention required unless mechanically irritated by clothing."
    },
    14: {
        "code": "MEL",
        "name": "Skin Cancer (Melanoma / Carcinoma)",
        "fullName": "Malignant Skin Cancer (Melanoma / BCC / SCC)",
        "type": "Malignant",
        "severity": "Critical Risk",
        "badgeColor": "#dc2626",
        "description": "High-risk malignant skin tumor with invasive or metastatic potential. Requires immediate histological confirmation.",
        "action": "Urgent surgical dermatology biopsy and staging protocol strongly indicated without delay."
    },
    15: {
        "code": "PHOTO_DAMAGE",
        "name": "Sun & UV Damage",
        "fullName": "Solar Elastosis & Photoaging",
        "type": "Environmental / UV",
        "severity": "Low Risk",
        "badgeColor": "#eab308",
        "description": "Degenerative changes in dermal elastin fibers and collagen matrix caused by cumulative solar radiation.",
        "action": "Strict daily broad-spectrum SPF 50+ sunscreen and annual skin cancer surveillance exams."
    },
    16: {
        "code": "TINEA",
        "name": "Tinea (Ringworm)",
        "fullName": "Dermatophytosis (Tinea Corporis/Pedis)",
        "type": "Fungal Infection",
        "severity": "Low Risk",
        "badgeColor": "#14b8a6",
        "description": "Superficial fungal infection of keratinized tissue presenting with characteristic annular scaly plaques with active borders.",
        "action": "Topical allylamine or azole antifungals (terbinafine) for 2 to 4 weeks."
    },
    17: {
        "code": "NORMAL",
        "name": "Normal / Healthy Skin",
        "fullName": "Normal Cutaneous Architecture",
        "type": "Normal",
        "severity": "Healthy",
        "badgeColor": "#22c55e",
        "description": "Intact physiological skin surface showing healthy epidermal stratum corneum without pathological lesion traits.",
        "action": "Maintain routine skin hydration and standard sun protection habits."
    },
    18: {
        "code": "VASC",
        "name": "Vascular Tumors",
        "fullName": "Hemangioma / Vascular Malformation",
        "type": "Vascular",
        "severity": "Low Risk",
        "badgeColor": "#f43f5e",
        "description": "Benign proliferation of blood vessels or capillary malformations presenting as bright red to violaceous papules.",
        "action": "Generally benign. Pulsed dye laser or excision if bleeding or cosmetic concern."
    },
    19: {
        "code": "VASCULITIS",
        "name": "Cutaneous Vasculitis",
        "fullName": "Leukocytoclastic Vasculitis",
        "type": "Autoimmune / Vascular",
        "severity": "High Risk",
        "badgeColor": "#e11d48",
        "description": "Inflammation and necrosis of blood vessel walls manifesting as palpable purpura, petechiae, or cutaneous ulceration.",
        "action": "Immediate medical evaluation to exclude renal or systemic organ involvement; punch biopsy."
    },
    20: {
        "code": "VITILIGO",
        "name": "Vitiligo",
        "fullName": "Vitiligo (Depigmentation)",
        "type": "Autoimmune",
        "severity": "Low Risk",
        "badgeColor": "#6366f1",
        "description": "Selective autoimmune destruction of epidermal melanocytes causing sharply circumscribed depigmented macules.",
        "action": "Topical calcineurin inhibitors, targeted narrow-band UVB phototherapy, and sun protection for pale areas."
    },
    21: {
        "code": "WARTS",
        "name": "Warts (Verruca)",
        "fullName": "Verruca Vulgaris (HPV)",
        "type": "Viral Infection",
        "severity": "Low Risk",
        "badgeColor": "#64748b",
        "description": "Benign hyperkeratotic epithelial proliferation caused by localized infection with Human Papillomavirus.",
        "action": "Over-the-counter salicylic acid, cryotherapy with liquid nitrogen, or immune-modulating agents."
    }
}

class LaurianeViTClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.weights = None
        self.config = None
        self.num_layers = 12
        self.num_heads = 12
        self.head_dim = 64
        self.hidden_dim = 768
        self.intermediate_dim = 3072
        self.num_classes = 22
        self.is_ready = False
        
        self._load_config_and_weights()

    def _download_weights(self, st_path):
        import urllib.request
        url = "https://huggingface.co/LaurianeMD/vit-skin-disease/resolve/main/model.safetensors"
        print(f"[...] ViT model weights not found locally. Auto-downloading from Hugging Face ({url})...")
        os.makedirs(os.path.dirname(st_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, st_path)
            print("[OK] Model weights downloaded successfully!")
        except Exception as e:
            print(f"[!] Auto-download failed: {e}. Please ensure internet connection is available.")

    def _load_config_and_weights(self):
        cfg_path = os.path.join(self.model_dir, "config.json")
        st_path = os.path.join(self.model_dir, "model.safetensors")

        if not os.path.exists(cfg_path):
            print(f"[!] Config file missing in {self.model_dir}.")
            return

        # Auto-download model weights if not present locally (<10KB)
        if not os.path.exists(st_path) or os.path.getsize(st_path) < 10000:
            self._download_weights(st_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        print(f"[...] Loading model/vit-skin-disease safetensors weights...")
        self.weights = load_file(st_path)
        self.is_ready = True
        print(f"[OK] model/vit-skin-disease loaded successfully! ({len(self.weights)} tensors, 22 classes)")

    @staticmethod
    def _layer_norm(x, weight, bias, eps=1e-12):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + scipy.special.erf(x / 1.4142135623730951))

    @staticmethod
    def _softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def preprocess_image(self, pil_img: Image.Image):
        # 1. Resize to (224, 224) and convert to RGB
        img = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        # 2. Normalize with ViTImageProcessor defaults: mean=0.5, std=0.5
        arr = (arr - 0.5) / 0.5
        # 3. Transpose to Channel-First: (3, 224, 224)
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def forward(self, input_tensor):
        """
        Runs the exact 12-layer Vision Transformer forward pass in pure NumPy.
        input_tensor: shape (3, 224, 224)
        """
        if not self.is_ready:
            self._load_config_and_weights()
            if not self.is_ready:
                raise RuntimeError("Model weights are not fully loaded yet.")

        W = self.weights

        # 1. Patch Embeddings: Conv2d(3, 768, kernel_size=16, stride=16)
        # Reshape non-overlapping patches (3, 14, 16, 14, 16) -> (196, 768)
        c, h, w = input_tensor.shape
        patches = input_tensor.reshape(3, 14, 16, 14, 16).transpose(1, 3, 0, 2, 4).reshape(196, 768)
        proj_w = W["vit.embeddings.patch_embeddings.projection.weight"].reshape(768, 768)
        proj_b = W["vit.embeddings.patch_embeddings.projection.bias"]
        patch_embeds = patches @ proj_w.T + proj_b  # shape (196, 768)

        # 2. Add CLS Token & Position Embeddings
        cls_token = W["vit.embeddings.cls_token"].reshape(1, 768)
        pos_embeds = W["vit.embeddings.position_embeddings"].reshape(197, 768)

        x = np.concatenate([cls_token, patch_embeds], axis=0) + pos_embeds  # shape (197, 768)

        # 3. 12 Transformer Encoder Layers
        for i in range(self.num_layers):
            prefix = f"vit.encoder.layer.{i}."

            # -- Self Attention --
            ln1_w = W[prefix + "layernorm_before.weight"]
            ln1_b = W[prefix + "layernorm_before.bias"]
            norm_x = self._layer_norm(x, ln1_w, ln1_b)

            q_w = W[prefix + "attention.attention.query.weight"]
            q_b = W[prefix + "attention.attention.query.bias"]
            k_w = W[prefix + "attention.attention.key.weight"]
            k_b = W[prefix + "attention.attention.key.bias"]
            v_w = W[prefix + "attention.attention.value.weight"]
            v_b = W[prefix + "attention.attention.value.bias"]

            Q = (norm_x @ q_w.T + q_b).reshape(197, 12, 64).transpose(1, 0, 2)
            K = (norm_x @ k_w.T + k_b).reshape(197, 12, 64).transpose(1, 0, 2)
            V = (norm_x @ v_w.T + v_b).reshape(197, 12, 64).transpose(1, 0, 2)

            # Scaled Dot-Product Attention: (12, 197, 197)
            attn_scores = (Q @ K.transpose(0, 2, 1)) * 0.125  # 1 / sqrt(64) = 0.125
            attn_probs = self._softmax(attn_scores, axis=-1)

            context = (attn_probs @ V).transpose(1, 0, 2).reshape(197, 768)

            out_w = W[prefix + "attention.output.dense.weight"]
            out_b = W[prefix + "attention.output.dense.bias"]
            attn_out = context @ out_w.T + out_b

            # Residual connection
            x = x + attn_out

            # -- Feed-Forward MLP --
            ln2_w = W[prefix + "layernorm_after.weight"]
            ln2_b = W[prefix + "layernorm_after.bias"]
            norm_x2 = self._layer_norm(x, ln2_w, ln2_b)

            fc1_w = W[prefix + "intermediate.dense.weight"]
            fc1_b = W[prefix + "intermediate.dense.bias"]
            fc2_w = W[prefix + "output.dense.weight"]
            fc2_b = W[prefix + "output.dense.bias"]

            hidden = self._gelu(norm_x2 @ fc1_w.T + fc1_b)
            mlp_out = hidden @ fc2_w.T + fc2_b

            # Residual connection
            x = x + mlp_out

        # 4. Final LayerNorm
        final_ln_w = W["vit.layernorm.weight"]
        final_ln_b = W["vit.layernorm.bias"]
        x = self._layer_norm(x, final_ln_w, final_ln_b)

        # 5. Classifier Head on CLS Token
        cls_feature = x[0]  # shape (768,)
        clf_w = W["classifier.weight"]  # shape (22, 768)
        clf_b = W["classifier.bias"]    # shape (22,)

        logits = clf_w @ cls_feature + clf_b  # shape (22,)
        probs = self._softmax(logits)

        return probs

    def predict(self, pil_img: Image.Image):
        tensor = self.preprocess_image(pil_img)
        probs = self.forward(tensor)

        results = []
        for idx in range(self.num_classes):
            prob = float(probs[idx])
            meta = DISEASE_METADATA.get(idx, {
                "code": f"CLASS_{idx}",
                "name": self.config.get("id2label", {}).get(str(idx), f"Class {idx}"),
                "fullName": f"Class {idx}",
                "type": "General",
                "severity": "Low Risk",
                "badgeColor": "#64748b",
                "description": "Cutaneous classification category.",
                "action": "Consult a healthcare provider for diagnosis."
            })
            results.append({
                "classId": idx,
                "code": meta["code"],
                "name": meta["name"],
                "fullName": meta["fullName"],
                "type": meta["type"],
                "severity": meta["severity"],
                "badgeColor": meta["badgeColor"],
                "probability": prob,
                "percentage": round(prob * 100, 2),
                "description": meta["description"],
                "action": meta["action"]
            })

        results.sort(key=lambda r: r["probability"], reverse=True)
        return {
            "top": results[0],
            "top5": results[:5],
            "all": results
        }

# Global singleton
_classifier_instance = None

def get_vit_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = LaurianeViTClassifier()
    return _classifier_instance
