"""
Web UI: face counting + recognition with cached LBPH model,
        plus a person/pedestrian counter for surveillance footage.

  python app.py
  open the URL printed in the terminal (default starts at 5050; next free port if busy).

Env:
  FACE_REC_DATA_DIR       default ``data``
  FACE_MODEL_CACHE        default ``<data>/.lbph_cache``
  FACE_REC_THRESHOLD      embedding L2 distance threshold (default 0.55; lower = stricter)
  FACE_DETECT_DOWNSCALE   kept for API compat; not used internally
  FACE_REC_FORCE_RETRAIN  1 = ignore cache and retrain
  FACE_COUNT_MODEL        hog (default) or cnn
  FACE_COUNT_SENSITIVE    1 = higher upsample / larger max-side
  FACE_COUNT_ENHANCE      0 = disable CLAHE contrast enhancement (default on)
  FACE_COUNT_SHARPEN      1 = enable unsharp-mask sharpening (for blurry images)
  FACE_COUNT_MIN_FACE     minimum face size in px (default 20)
  FACE_COUNT_NMS_IOU      NMS IoU for face duplicate merge (default 0.35)
  FACE_EVAL_IMAGES_DIR    default still-image folder for /api/eval_face (image_data)
  FACE_EVAL_FRAMES_DIR    default frames folder for /api/eval_person (frames)
  FACE_EVAL_FACE_LABELS   default CSV path for face eval labels (output_image_data_test/image_data_counts.csv)
  FACE_EVAL_API           set to 1/true to enable POST /api/eval_face and /api/eval_person
  FACE_EVAL_DEFAULT_LIMIT sample size default for eval API (default 200)
  FACE_EVAL_MAX           hard cap on eval limit per request (default 2000)
  PORT                    if set: bind 0.0.0.0 (cloud); if unset: 127.0.0.1 + port scan
  FLASK_DEBUG             see README (defaults differ with/without PORT)
"""

from __future__ import annotations

import base64
import os
import socket
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_count import (
    FaceCounterSystem,
    PersonCounterSystem,
    evaluate_face_counter_against_csv,
    evaluate_person_counter_against_csv,
)
from face_rec import SimpleFaceRecognizer, default_cache_dir, resolve_unknown_dir

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATA_DIR = os.environ.get("FACE_REC_DATA_DIR", "data")
KNOWN_FACES = str(Path(DATA_DIR) / "known_faces")
MODEL_CACHE = os.environ.get(
    "FACE_MODEL_CACHE",
    str(default_cache_dir(DATA_DIR)),
)

# Project root: server-side eval paths must stay under this directory.
PROJECT_ROOT = Path(__file__).resolve().parent


def _eval_api_enabled() -> bool:
    return os.environ.get("FACE_EVAL_API", "").lower() in ("1", "true", "yes")


def _safe_path_under_project(path_str: str) -> Path:
    """Resolve ``path_str`` to an absolute path confined under ``PROJECT_ROOT``."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError as e:
        raise ValueError(f"Path must be under project root: {path_str}") from e
    return p


def _eval_face_defaults() -> tuple[str, str | None]:
    """``(images_dir, labels_csv)`` with optional labels path if that file exists."""
    img = os.environ.get("FACE_EVAL_IMAGES_DIR", "image_data")
    lbl = os.environ.get("FACE_EVAL_FACE_LABELS", "")
    if lbl:
        return img, lbl
    fallback = PROJECT_ROOT / "output_image_data_test" / "image_data_counts.csv"
    return img, str(fallback) if fallback.is_file() else None


def _eval_person_defaults() -> str:
    return os.environ.get("FACE_EVAL_FRAMES_DIR", "frames")

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
        nms_iou = float(os.environ.get("FACE_COUNT_NMS_IOU", "0.35"))
        _counter = FaceCounterSystem(
            model=model,
            sensitive_counting=sens not in ("0", "false", "no"),
            enhance=enhance,
            sharpen=sharpen,
            min_face_size=min_face,
            nms_iou=nms_iou,
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
    eval_img_rel = os.environ.get("FACE_EVAL_IMAGES_DIR", "image_data")
    eval_fr_rel = os.environ.get("FACE_EVAL_FRAMES_DIR", "frames")
    eval_img = PROJECT_ROOT / eval_img_rel
    eval_fr = PROJECT_ROOT / eval_fr_rel
    _, face_lbl_default = _eval_face_defaults()
    face_lbl_path = Path(face_lbl_default) if face_lbl_default else None
    return jsonify(
        {
            "data_dir":            DATA_DIR,
            "known_faces":         KNOWN_FACES,
            "known_faces_exists":  rec_ok,
            "model_cache":         MODEL_CACHE,
            "cache_present":       cache_ok,
            "unknown_dir":         resolve_unknown_dir(DATA_DIR) if rec_ok else None,
            "last_load_mode":      _last_train_mode,
            "eval": {
                "api_enabled":           _eval_api_enabled(),
                "face_images_dir":       str(eval_img),
                "face_images_exists":    eval_img.is_dir(),
                "frames_dir":            str(eval_fr),
                "frames_exists":         eval_fr.is_dir(),
                "face_labels_csv":       str(face_lbl_path) if face_lbl_path else None,
                "face_labels_exists":    face_lbl_path.is_file() if face_lbl_path else False,
                "default_eval_limit":    int(os.environ.get("FACE_EVAL_DEFAULT_LIMIT", "200")),
                "max_eval_limit":        int(os.environ.get("FACE_EVAL_MAX", "2000")),
            },
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


def _eval_json_body() -> dict:
    if not request.is_json:
        return {}
    raw = request.get_json(silent=True)
    return raw if isinstance(raw, dict) else {}


@app.route("/api/eval_face", methods=["POST"])
def api_eval_face():
    """Train/test metrics for still images vs a CSV (opt-in via ``FACE_EVAL_API``)."""
    if not _eval_api_enabled():
        return jsonify(
            {"error": "Server-side eval is disabled. Set FACE_EVAL_API=1 to enable POST /api/eval_face."}
        ), 403

    data = _eval_json_body()
    if not data and request.form:
        data = {k: request.form.get(k) for k in request.form}

    img_arg = data.get("images_dir")
    lbl_arg = data.get("labels_csv")
    default_img, default_lbl = _eval_face_defaults()
    img_rel = img_arg if img_arg is not None else default_img
    lbl_rel = lbl_arg if lbl_arg is not None else default_lbl
    if not lbl_rel:
        return jsonify(
            {
                "error": "No labels CSV. Pass labels_csv or set FACE_EVAL_FACE_LABELS, "
                         "or add output_image_data_test/image_data_counts.csv under the project.",
            }
        ), 400

    try:
        img_path = _safe_path_under_project(str(img_rel))
        lbl_path = _safe_path_under_project(str(lbl_rel))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    default_limit = int(os.environ.get("FACE_EVAL_DEFAULT_LIMIT", "200"))
    hard_max = int(os.environ.get("FACE_EVAL_MAX", "2000"))
    try:
        lim_raw = data.get("limit", default_limit)
        limit = int(lim_raw) if lim_raw not in (None, "") else default_limit
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid limit"}), 400
    limit = max(1, min(limit, hard_max))

    try:
        train_ratio = float(data.get("train_ratio", 0.8))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid train_ratio"}), 400

    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid seed"}), 400

    split = str(data.get("split", "random"))
    if split not in ("random", "chronological"):
        return jsonify({"error": "split must be \"random\" or \"chronological\""}), 400

    tune = str(data.get("tune_hparams", "false")).lower() in ("1", "true", "yes")

    c = get_counter()
    try:
        report = evaluate_face_counter_against_csv(
            images_dir=str(img_path),
            labels_csv=str(lbl_path),
            train_ratio=train_ratio,
            seed=seed,
            model=c.model,
            enhance=c.enhance,
            sharpen=c.sharpen,
            nms_iou=c.nms_iou,
            tune_hparams=tune,
            limit=limit,
            chronological_split=(split == "chronological"),
            sensitive_counting=c.sensitive_counting,
            min_face_size=c.min_face_size,
        )
    except (OSError, ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(asdict(report))


@app.route("/api/eval_person", methods=["POST"])
def api_eval_person():
    """Train/test metrics for a frame folder vs a pedestrian ground-truth CSV."""
    if not _eval_api_enabled():
        return jsonify(
            {"error": "Server-side eval is disabled. Set FACE_EVAL_API=1 to enable POST /api/eval_person."}
        ), 403

    data = _eval_json_body()
    if not data and request.form:
        data = {k: request.form.get(k) for k in request.form}

    fr_rel = data.get("frames_dir") if data.get("frames_dir") is not None else _eval_person_defaults()
    lbl_rel = data.get("labels_csv")
    if not lbl_rel:
        return jsonify(
            {"error": "labels_csv is required (true pedestrian counts per frame filename)."}
        ), 400

    try:
        fr_path = _safe_path_under_project(str(fr_rel))
        lbl_path = _safe_path_under_project(str(lbl_rel))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    default_limit = int(os.environ.get("FACE_EVAL_DEFAULT_LIMIT", "200"))
    hard_max = int(os.environ.get("FACE_EVAL_MAX", "2000"))
    try:
        lim_raw = data.get("limit", default_limit)
        limit = int(lim_raw) if lim_raw not in (None, "") else default_limit
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid limit"}), 400
    limit = max(1, min(limit, hard_max))

    try:
        train_ratio = float(data.get("train_ratio", 0.8))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid train_ratio"}), 400

    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid seed"}), 400

    split = str(data.get("split", "chronological"))
    if split not in ("random", "chronological"):
        return jsonify({"error": "split must be \"random\" or \"chronological\""}), 400

    method = str(data.get("method", "hog"))
    if method not in ("hog", "mog2", "combined"):
        return jsonify({"error": "method must be hog, mog2, or combined"}), 400

    try:
        pe = int(data.get("progress_every", 200))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid progress_every"}), 400

    try:
        report = evaluate_person_counter_against_csv(
            frames_dir=str(fr_path),
            labels_csv=str(lbl_path),
            train_ratio=train_ratio,
            seed=seed,
            method=method,
            limit=limit,
            chronological_split=(split == "chronological"),
            progress_every=max(1, pe),
        )
    except (OSError, ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(asdict(report))


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
