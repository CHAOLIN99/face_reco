"""
Web UI: face counting + recognition with cached LBPH model,
        plus a person/pedestrian counter for surveillance footage.

  python app.py
  open the URL printed in the terminal (default starts at 5050; next free port if busy).

Env:
  FACE_REC_DATA_DIR       default ``data``
  FACE_MODEL_CACHE        default ``<data>/.lbph_cache``
  FACE_REC_THRESHOLD      LBPH distance threshold (default 72)
  FACE_DETECT_DOWNSCALE   Haar downscale factor (default 0.55)
  FACE_REC_FORCE_RETRAIN  1 = ignore cache and retrain
  FACE_COUNT_MODEL        hog (default) or cnn
  FACE_COUNT_SENSITIVE    1 = higher upsample / larger max-side
  FACE_COUNT_ENHANCE      0 = disable CLAHE contrast enhancement (default on)
  FACE_COUNT_SHARPEN      1 = enable unsharp-mask sharpening (for blurry images)
  FACE_COUNT_MIN_FACE     minimum face size in px (default 20)
  PORT                    if set: bind 0.0.0.0 (cloud); if unset: 127.0.0.1 + port scan
  FLASK_DEBUG             see README (defaults differ with/without PORT)
"""

from __future__ import annotations

import base64
import os
import socket
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_count import FaceCounterSystem, PersonCounterSystem
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
_person_counter: PersonCounterSystem | None = None
_recognizer: SimpleFaceRecognizer | None = None
_recognizer_error: str | None = None
_last_train_mode: str | None = None


def get_counter() -> FaceCounterSystem:
    global _counter
    if _counter is None:
        model    = os.environ.get("FACE_COUNT_MODEL", "hog")
        sens     = os.environ.get("FACE_COUNT_SENSITIVE", "0").lower()
        enhance  = os.environ.get("FACE_COUNT_ENHANCE", "1").lower() not in ("0", "false", "no")
        sharpen  = os.environ.get("FACE_COUNT_SHARPEN", "0").lower() not in ("0", "false", "no")
        min_face = int(os.environ.get("FACE_COUNT_MIN_FACE", "20"))
        _counter = FaceCounterSystem(
            model=model,
            sensitive_counting=sens not in ("0", "false", "no"),
            enhance=enhance,
            sharpen=sharpen,
            min_face_size=min_face,
        )
    return _counter


def get_person_counter() -> PersonCounterSystem:
    """Lazy singleton for the HOG pedestrian counter (stateless between requests)."""
    global _person_counter
    if _person_counter is None:
        _person_counter = PersonCounterSystem()
    return _person_counter


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


def bgr_to_jpeg_data_url(bgr: np.ndarray, quality: int = 88) -> str:
    """Encode a BGR frame as a JPEG data URL (3-5× smaller than PNG for photos)."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _decode_upload(file_storage) -> np.ndarray | None:
    """Read a Flask FileStorage upload and return a decoded BGR frame, or None."""
    arr = np.frombuffer(file_storage.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", data_dir=DATA_DIR, model_cache=MODEL_CACHE)


@app.route("/api/health")
def health():
    rec_ok   = Path(KNOWN_FACES).exists()
    cache_ok = Path(MODEL_CACHE).is_dir() and (Path(MODEL_CACHE) / "meta.json").is_file()
    return jsonify(
        {
            "data_dir":            DATA_DIR,
            "known_faces":         KNOWN_FACES,
            "known_faces_exists":  rec_ok,
            "model_cache":         MODEL_CACHE,
            "cache_present":       cache_ok,
            "unknown_dir":         resolve_unknown_dir(DATA_DIR) if rec_ok else None,
            "last_load_mode":      _last_train_mode,
        }
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force LBPH retrain and refresh the cache (e.g. after adding/changing faces)."""
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
        return jsonify({"count": count, "image": bgr_to_jpeg_data_url(bgr)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/count_people", methods=["POST"])
def api_count_people():
    """Person/pedestrian counter using OpenCV HOG.

    Designed for surveillance or overhead-angle footage where frontal
    face detection fails.  Accepts the same ``image`` file field as
    ``/api/count`` and returns an identical response shape.
    """
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
        pc = get_person_counter()
        count, bgr = pc.count_people_in_bytes(raw)
        return jsonify({"count": count, "image": bgr_to_jpeg_data_url(bgr)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(f.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": "Unsupported image type"}), 400

    # Decode directly from memory — no temp file needed
    frame = _decode_upload(f)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        rec    = get_recognizer()
        detail = rec.predict_from_bgr_detail(frame)
        if detail is None:
            return jsonify(
                {
                    "face_found": False,
                    "message": "No face detected (try a clearer frontal photo).",
                }
            )

        thumb_url = None
        big_url   = None
        if detail.similar_thumb_bgr is not None:
            thumb_url = bgr_to_jpeg_data_url(detail.similar_thumb_bgr)
        ref = rec.reference_display_bgr(detail)
        if ref is not None:
            big_url = bgr_to_jpeg_data_url(ref)

        return jsonify(
            {
                "face_found":            True,
                "name":                  detail.name,
                "distance":              detail.distance,
                "is_match":              detail.is_match,
                "verdict":               detail.verdict,
                "similar_training_file": detail.similar_source,
                "similar_thumb":         thumb_url,
                "reference_large":       big_url,
            }
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/snapshot_identify", methods=["POST"])
def api_snapshot_identify():
    """One webcam frame → largest face → most likely identity + reference image.
    No continuous processing — single-shot only."""
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400

    frame = _decode_upload(request.files["image"])
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        rec    = get_recognizer()
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
                "face_found":            True,
                "name":                  detail.name,
                "distance":              detail.distance,
                "is_match":              detail.is_match,
                "verdict":               detail.verdict,
                "similar_training_file": detail.similar_source,
                "reference_large":       bgr_to_jpeg_data_url(ref) if ref is not None else None,
            }
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503


# ---------------------------------------------------------------------------
# Server startup helpers
# ---------------------------------------------------------------------------

def _resolve_run_port() -> int:
    """Scan 5050–5079 for an available port; fall back to an OS-assigned one."""
    preferred = 5050
    for p in range(preferred, preferred + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
        except OSError as e:
            raise RuntimeError(
                "Could not bind to 127.0.0.1 (ports 5050–5079 and ephemeral all failed)."
                " Set PORT to a free port."
            ) from e


def _bind_config() -> tuple[int, str, bool]:
    """Return (port, host, debug).

    Cloud (PORT set): all interfaces, debug off by default.
    Local (PORT unset): loopback, debug on by default, port scan.
    """
    if "PORT" in os.environ:
        port  = int(os.environ["PORT"])
        host  = "0.0.0.0"
        debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
        return port, host, debug
    port  = _resolve_run_port()
    host  = "127.0.0.1"
    debug = os.environ.get("FLASK_DEBUG", "1").lower() not in ("0", "false", "no")
    return port, host, debug


if __name__ == "__main__":
    port, host, debug = _bind_config()
    if host == "0.0.0.0":
        print(f"\n  Face Studio → listening on 0.0.0.0:{port}/ (set FLASK_DEBUG=1 for debug)\n")
    else:
        print(f"\n  Face Studio → http://127.0.0.1:{port}/\n")
    app.run(host=host, port=port, debug=debug)
