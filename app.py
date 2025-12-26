from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

from model.resnet18_custom import ResNet18

# =========================
# CONFIG
# =========================
IMG_SIZE = (160, 160)
WEIGHTS_PATH = "model/resnet18.weights.h5"
CLASSES_PATH = "model/class_indices.json"

model = None
idx_to_class = None

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, idx_to_class

    # 1️⃣ recria arquitetura
    model = ResNet18(num_classes=4, input_shape=(224, 224, 3))

    # 2️⃣ força build
    dummy = tf.zeros((1, *IMG_SIZE, 3))
    _ = model(dummy)

    # 3️⃣ carrega pesos
    model.load_weights(WEIGHTS_PATH)

    # 4️⃣ carrega classes
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    print("✅ Modelo carregado com pesos")

    yield

    print("🛑 API finalizada")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Coffee Leaf Disease Classifier",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PREPROCESS
# =========================
def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagem inválida")

    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Arquivo não é imagem")

    image = preprocess_image(await file.read())

    preds = model.predict(image, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "class": idx_to_class[class_id],
        "confidence": round(confidence, 4)
    }


# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
