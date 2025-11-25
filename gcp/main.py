from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import jsonify, make_response

BUCKET_NAME = "group-4-plant-doc"
class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Blight",
    "Potato - Healthy",
    "Rice - Downy Mildew",
    "Rice - Healthy",
    "Tomato - Healthy",
    "Tomato - Leaf Mold"
]

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

def predict(request):
    global model

    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))

    # Handle GET requests for health check
    if request.method == "GET":
        return add_cors_headers(jsonify({"message": "Predict endpoint alive"}))

    # Handle POST requests for prediction
    if model is None:
        download_blob(
            BUCKET_NAME,
            "saved_models/latest_model.h5",
            "/tmp/latest_model.h5",
        )
        model = tf.keras.models.load_model("/tmp/latest_model.h5")

    if "file" not in request.files:
        return add_cors_headers(jsonify({"error": "No file uploaded"})), 400

    image_file = request.files["file"]

    # Preprocess: resize only, no manual rescaling
    image = np.array(Image.open(image_file).convert("RGB").resize((256, 256)))
    img_array = tf.expand_dims(image, 0)

    # Run prediction
    predictions = model.predict(img_array)
    print(predictions)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))  # fraction, e.g. 0.95

    # Return JSON response with CORS headers
    return add_cors_headers(jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 4)  # frontend multiplies by 100
    }))
 