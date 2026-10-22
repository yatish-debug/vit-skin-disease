import os
import io
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from webapp.hf_vit_model import get_vit_classifier, DISEASE_METADATA

app = FastAPI(
    title="Skin Disease ViT Diagnostic Platform (LaurianeMD/vit-skin-disease)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

@app.on_event("startup")
def startup_event():
    global classifier
    try:
        classifier = get_vit_classifier()
        print(f"🚀 Loaded LaurianeMD/vit-skin-disease (22 Classes)")
    except Exception as e:
        print(f"⚠️ Error loading ViT classifier: {e}")

@app.get("/")
def read_root():
    return FileResponse("webapp/static/index.html")

@app.get("/api/info")
def get_info():
    global classifier
    if classifier is None:
        classifier = get_vit_classifier()

    classes_list = [DISEASE_METADATA[i] for i in sorted(DISEASE_METADATA.keys())]

    return {
        "status": "ready" if classifier and classifier.is_ready else "loading",
        "modelName": "LaurianeMD/vit-skin-disease",
        "architecture": "Vision Transformer (ViT-Base-Patch16-224)",
        "numClasses": 22,
        "classes": classes_list,
        "inputSize": [224, 224]
    }

@app.get("/api/samples")
def get_samples():
    sample_dir = os.path.join("webapp", "static", "samples")
    samples = []
    if os.path.exists(sample_dir):
        for f in sorted(os.listdir(sample_dir)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                name_clean = os.path.splitext(f)[0].replace("_", " ")
                samples.append({
                    "filename": f,
                    "url": f"/static/samples/{f}",
                    "name": name_clean,
                    "type": "Clinical Sample"
                })
    return {"samples": samples}

@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    global classifier
    if classifier is None:
        classifier = get_vit_classifier()

    start_time = time.time()
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        source_name = file.filename or "Uploaded Image"
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format. Please upload a PNG, JPG, or WEBP image.")

    # Run LaurianeMD/vit-skin-disease forward pass
    pred_results = classifier.predict(pil_image)
    inference_ms = round((time.time() - start_time) * 1000, 1)

    top_pred = pred_results["top"]
    top3 = pred_results["all"][:3]
    top5 = pred_results["all"][:5]

    # Generate exhaustive photo-specific clinical dossier
    from webapp.clinical_knowledge import generate_clinical_dossier
    dossier = generate_clinical_dossier(top_pred, pil_image, pred_results["all"])

    return {
        "success": True,
        "source": source_name,
        "model": "LaurianeMD/vit-skin-disease",
        "architecture": "Google Vision Transformer (ViT-Base)",
        "inferenceTimeMs": inference_ms,
        "topPrediction": top_pred,
        "dossier": dossier,
        "top3": top3,
        "top5": top5,
        "allPredictions": pred_results["all"]
    }

# Mount static folder
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=False)
