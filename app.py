"""
Web UI: face counting + recognition with cached LBPH model.

  python app.py
  open http://127.0.0.1:5050

Env:
  FACE_REC_DATA_DIR       default ``data``
  FACE_MODEL_CACHE        default ``<data>/.lbph_cache``
  FACE_REC_THRESHOLD      LBPH distance threshold
  FACE_DETECT_DOWNSCALE   Haar downscale (e.g. 0.4)
  FACE_REC_FORCE_RETRAIN  1 = ignore cache and retrain
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_count import FaceCounterSystem
from face_rec import SimpleFaceRecognizer, default_cache_dir, resolve_unknown_dir

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATA_DIR = os.environ.get("FACE_REC_DATA_DIR", "data")
KNOWN_FACES = str(Path(DATA_DIR) / "known_faces")
MODEL_CACHE = os.environ.get(
    "FACE_MODEL_CACHE",
    str(default_cache_dir(DATA_DIR)),
)

_counter: FaceCounterSystem | None = None
_recognizer: SimpleFaceRecognizer | None = None
_recognizer_error: str | None = None
_last_train_mode: str | None = None


def get_counter() -> FaceCounterSystem:
    global _counter
    if _counter is None:
        model = os.environ.get("FACE_COUNT_MODEL", "hog")
        _counter = FaceCounterSystem(model=model)
    return _counter


def get_recognizer() -> SimpleFaceRecognizer:
    global _recognizer, _recognizer_error, _last_train_mode
    if _recognizer is None:
        if _recognizer_error:
            raise RuntimeError(_recognizer_error)
        try:
            thr = os.environ.get("FACE_REC_THRESHOLD")
            dsc = os.environ.get("FACE_DETECT_DOWNSCALE")
            r = SimpleFaceRecognizer(
                match_threshold=float(thr) if thr else None,
                detect_downscale=float(dsc) if dsc else None,
            )
            if not Path(KNOWN_FACES).exists():
                raise FileNotFoundError(f"Missing known_faces folder: {KNOWN_FACES}")
            _last_train_mode = r.train_or_load(KNOWN_FACES, cache_dir=MODEL_CACHE)
            _recognizer = r
        except Exception as e:
            _recognizer_error = str(e)
            raise RuntimeError(_recognizer_error) from e
    return _recognizer


def bgr_to_png_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html", data_dir=DATA_DIR, model_cache=MODEL_CACHE)


@app.route("/api/health")
def health():
    rec_ok = Path(KNOWN_FACES).exists()
    cache_ok = Path(MODEL_CACHE).is_dir() and (Path(MODEL_CACHE) / "meta.json").is_file()
    return jsonify(
        {
            "data_dir": DATA_DIR,
            "known_faces": KNOWN_FACES,
            "known_faces_exists": rec_ok,
            "model_cache": MODEL_CACHE,
            "cache_present": cache_ok,
            "unknown_dir": resolve_unknown_dir(DATA_DIR) if rec_ok else None,
            "last_load_mode": _last_train_mode,
        }
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force rebuild LBPH cache (e.g. after adding faces)."""
    global _recognizer, _recognizer_error, _last_train_mode
    _recognizer = None
    _recognizer_error = None
    os.environ["FACE_REC_FORCE_RETRAIN"] = "1"
    try:
        get_recognizer()
        mode = _last_train_mode
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    finally:
        os.environ.pop("FACE_REC_FORCE_RETRAIN", None)
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/count", methods=["POST"])
def api_count():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(f.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": "Use .jpg, .png, or .webp"}), 400

    raw = f.read()
    try:
        counter = get_counter()
        count, bgr = counter.count_faces_in_bytes(raw)
        return jsonify({"count": count, "image": bgr_to_png_data_url(bgr)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(f.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": "Unsupported type"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        rec = get_recognizer()
        detail = rec.predict_from_image_detail(tmp_path)
        if detail is None:
            return jsonify(
                {
                    "face_found": False,
                    "message": "No face detected (try a clearer frontal photo).",
                }
            )

        thumb_url = None
        big_url = None
        if detail.similar_thumb_bgr is not None:
            thumb_url = bgr_to_png_data_url(detail.similar_thumb_bgr)
        ref = rec.reference_display_bgr(detail)
        if ref is not None:
            big_url = bgr_to_png_data_url(ref)

        return jsonify(
            {
                "face_found": True,
                "name": detail.name,
                "distance": detail.distance,
                "is_match": detail.is_match,
                "verdict": detail.verdict,
                "similar_training_file": detail.similar_source,
                "similar_thumb": thumb_url,
                "reference_large": big_url,
            }
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/snapshot_identify", methods=["POST"])
def api_snapshot_identify():
    """
    One frame from the webcam: largest face → most likely identity + reference image.
    No continuous processing.
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400
    data = request.files["image"].read()
    if not data:
        return jsonify({"error": "Empty body"}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        rec = get_recognizer()
        detail = rec.identify_largest_face(frame)
        if detail is None:
            return jsonify(
                {
                    "face_found": False,
                    "message": "No face detected in this frame.",
                }
            )

        ref = rec.reference_display_bgr(detail)
        return jsonify(
            {
                "face_found": True,
                "name": detail.name,
                "distance": detail.distance,
                "is_match": detail.is_match,
                "verdict": detail.verdict,
                "similar_training_file": detail.similar_source,
                "reference_large": bgr_to_png_data_url(ref) if ref is not None else None,
            }
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503


if __name__ == "__main__":
    # Default 5050: macOS often reserves 5000 for AirPlay Receiver.
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=True)
