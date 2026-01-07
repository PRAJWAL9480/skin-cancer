# flask_api/app.py — server that serves static frontend and /predict
import os, sys, traceback, tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import inference as inference_module
except Exception as e:
    raise RuntimeError(f"Failed to import inference.py: {e}\n{traceback.format_exc()}")

app = Flask(__name__, static_folder="static")
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"no image file provided"}), 400
    f = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1] or ".jpg") as tmp:
        tmp_path = tmp.name
        tmp.write(f.read())
        tmp.flush()
    try:
        # run skin detector
        detector_fn = getattr(inference_module, "is_skin_lesion_model")
        accepted, skin_prob, reason = detector_fn(tmp_path)
        if not accepted:
            return jsonify({"accepted": False, "reject_reason": reason, "skin_score": float(skin_prob)})

        # run hybrid predictor
        predict_fn = getattr(inference_module, "predict_single_image")
        res = predict_fn(tmp_path)
        if not isinstance(res, dict):
            return jsonify({"error":"predict_single_image did not return dict"}), 500

        response = {
            "accepted": True,
            "skin_score": float(skin_prob),
            "predicted_class": res.get("predicted_class"),
            "predicted_prob": float(res.get("predicted_prob", 0.0)),
            "vgg_probs": res.get("vgg_probs"),
            "densenet_probs": res.get("densenet_probs"),
            "hybrid_probs": res.get("hybrid_probs"),
            "raw": res
        }
        return jsonify(response)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error":"server_error", "details": str(e), "traceback": tb}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    print("Starting Flask server. Project root:", PROJECT_ROOT)
    app.run(host="0.0.0.0", port=5000, debug=True)
