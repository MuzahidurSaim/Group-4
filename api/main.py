from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os

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
MODEL_DIR = "../saved_models"

def get_latest_model_path(model_dir: str):
    versions = []
    for fname in os.listdir(model_dir):
        name, ext = os.path.splitext(fname)
        if ext == ".keras" and name.isdigit():
            versions.append(int(name))
    if not versions:
        raise RuntimeError("No .keras models found in saved_models/")
    latest_version = max(versions)
    return os.path.join(model_dir, f"{latest_version}.keras")

# Load the latest model at startup
model_path = get_latest_model_path(MODEL_DIR)
print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

# Define your class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    # No manual resize or normalization here
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # no /255.0
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        "class": predicted_class,
        "confidence": confidence
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the latest model version without restarting the server."""
    global model, model_path
    new_model_path = get_latest_model_path(MODEL_DIR)
    model = tf.keras.models.load_model(new_model_path)
    model_path = new_model_path
    return {"message": f"Model reloaded from {model_path}"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
