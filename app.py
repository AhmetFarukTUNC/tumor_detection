import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2

MODEL_PATH = "cnn_pixel_model.h5"
CLASS_NAMES = ["Glioma", "Meningioma", "Notumor", "Pituitary"]

model = tf.keras.models.load_model(MODEL_PATH)
_, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = model.input_shape

app = Flask(__name__)

def preprocess_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
    dst_resized = cv2.resize(dst, (IMG_WIDTH, IMG_HEIGHT))
    if IMG_CHANNELS == 1:
        arr = dst_resized[..., np.newaxis]
    else:
        arr = cv2.cvtColor(dst_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Resim gönderilmedi"}), 400
    try:
        img = preprocess_image(request.files["image"])
        preds = model.predict(img).squeeze()
        if preds.ndim == 0: preds = [preds]
        class_idx = int(np.argmax(preds))
        return jsonify({
            "ok": True,
            "class": CLASS_NAMES[class_idx],
            "confidence": float(preds[class_idx])
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
