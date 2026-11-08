import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import io
import base64

import src.models
from src.models import EXPERIMENTS, ExpDecay

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

BEST_MODEL_PATH = os.path.join(OUT, "best_model.keras")

app = Flask(__name__, static_folder=OUT, static_url_path='/static')

models = {}
for name, (build, make_opt) in EXPERIMENTS.items():
    models[name] = build()
    models[name].compile(optimizer=make_opt(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

if os.path.exists(BEST_MODEL_PATH):
    try:
        best_loaded = tf.keras.models.load_model(
            BEST_MODEL_PATH, 
            compile=False, 
            custom_objects={"ExpDecay": ExpDecay}
        )
        models["best_saved_model"] = best_loaded
    except Exception as e:
        print("Could not load saved best model:", e)

@app.route("/")
def index():
    return render_template("index.html", model_names=list(models.keys()))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(OUT, filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    model_name = data.get("model", list(models.keys())[0])

    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = ImageOps.invert(image)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    new_image = Image.new("L", (28, 28), 0)
    paste_x = (28 - image.width) // 2
    paste_y = (28 - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))

    arr = np.array(new_image, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    results = {}
    if model_name == "all":
        for name, m in models.items():
            preds = m.predict(arr, verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            confidence = float(preds[pred_digit])
            results[name] = {
                "digit": pred_digit,
                "confidence": confidence,
                "probabilities": preds.tolist()
            }
    else:
        m = models.get(model_name, list(models.values())[0])
        preds = m.predict(arr, verbose=0)[0]
        pred_digit = int(np.argmax(preds))
        confidence = float(preds[pred_digit])
        results[model_name] = {
            "digit": pred_digit,
            "confidence": confidence,
            "probabilities": preds.tolist()
        }

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)