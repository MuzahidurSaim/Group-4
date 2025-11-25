from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os
import json
from pathlib import Path

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auto-detect latest model version ---
# Get absolute path to saved_models directory relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = str(BASE_DIR.parent / "saved_models")

def get_latest_model_path(model_dir: str):
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory not found: {model_dir}")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Model path is not a directory: {model_dir}")
    
    checkpoints = []
    for fname in os.listdir(model_dir):
        name, ext = os.path.splitext(fname)
        if ext == ".keras" and name.startswith("model_epoch_"):
            try:
                epoch_num = int(name.split("_")[-1])
                checkpoints.append((epoch_num, fname))
            except ValueError:
                continue
    if not checkpoints:
        raise RuntimeError("No .keras models found in saved_models/")
    # Pick the file with the highest epoch number
    latest_epoch, latest_fname = max(checkpoints, key=lambda x: x[0])
    return os.path.join(model_dir, latest_fname)

# --- Load model and class names at startup ---
model_path = get_latest_model_path(MODEL_DIR)
print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

class_names_path = os.path.join(MODEL_DIR, "class_names.json")
if not os.path.exists(class_names_path):
    raise RuntimeError("class_names.json not found in saved_models/")
with open(class_names_path) as f:
    CLASS_NAMES = json.load(f)

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    # Resize to match training input size
    image = Image.open(BytesIO(data)).resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # model handles normalization internally
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        "class": predicted_class,
        "confidence": confidence
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the latest model version and class names without restarting the server."""
    global model, model_path, CLASS_NAMES
    new_model_path = get_latest_model_path(MODEL_DIR)
    model = tf.keras.models.load_model(new_model_path)
    model_path = new_model_path

    # Reload class names
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        CLASS_NAMES = json.load(f)

    return {"message": f"Model reloaded from {model_path}"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
