import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SegmentationModel

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Resolve weights path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_FILE = os.path.join(WEIGHTS_DIR, "unet_resnet34_road_segmentation.pth")
CHECKPOINT_FILE = os.path.join(WEIGHTS_DIR, "best_unet_checkpoint.pth")

# Load model at startup
seg_model = None


def get_model():
    global seg_model
    if seg_model is not None:
        return seg_model
    # Prefer checkpoint (has optimizer state + better dice), fallback to plain weights
    for path in [CHECKPOINT_FILE, MODEL_FILE]:
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            seg_model = SegmentationModel(path)
            print("Model loaded successfully!")
            return seg_model
    return None


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def segment():
    model = get_model()
    if model is None:
        return jsonify({"error": "Model weights not found. Place .pth files in the weights/ directory."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Supported formats: PNG, JPG, JPEG, BMP, WebP, TIFF"}), 400

    try:
        image_bytes = file.read()
        result = model.predict(image_bytes)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    model = get_model()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(model.device) if model else "n/a"
    })


if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
