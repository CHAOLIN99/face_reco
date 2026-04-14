# Face Studio

A local toolkit for **counting faces** in images and **recognizing** who someone is against your own photo gallery. Includes a **Flask web UI** and **command-line** scripts — no cloud API, no third-party face service.

> **Local mode:** uploads go only to your Flask process on `127.0.0.1`.  
> **Deployed mode:** uploads reach that server — HTTPS, access control, and data policies are your responsibility.

---

## What it does

| Module | Role |
|--------|------|
| `face_rec.py` | Trains a **128-d deep embedding** recognizer (dlib ResNet via `face_recognition`) on folders of named people, predicts the most likely match, and caches embeddings so retraining is skipped when data is unchanged. |
| `face_count.py` | Two counters in one module: `FaceCounterSystem` (dlib HOG/CNN frontal-face detector with CLAHE + dual-pass NMS) and `PersonCounterSystem` (OpenCV HOG pedestrian + MOG2 background subtraction for CCTV/overhead footage). |
| `app.py` | Flask **web UI** and JSON API for face count, person count, recognize, one-shot webcam identify, forced retrain, and optional train/test evaluation. |

---

## Requirements

- **Python 3.10+** (3.11+ recommended)
- **macOS / Linux / Windows** (paths below use macOS/Linux style)
- **Webcam** optional — only needed for the snapshot-identify flow and CLI webcam modes

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pulls in:

| Package | Purpose |
|---------|---------|
| `face-recognition` | dlib 128-d face embeddings (detection + recognition). Requires [CMake](https://cmake.org/) and a C++ compiler on some systems — see [Troubleshooting](#troubleshooting). |
| `opencv-contrib-python` | OpenCV + extra modules for HOG pedestrian detection, MOG2 background subtraction, and image preprocessing. |
| `numpy` | Vectorized distance computation and array ops. |
| `pillow` | Image I/O used by `face_recognition`. |
| `flask` + `gunicorn` | Web server (dev and production). |

---

## Folder layout

```text
face_rec/
├── data/
│   ├── known_faces/          ← required for recognition
│   │   ├── Alice/
│   │   │   ├── photo1.jpg
│   │   │   └── photo2.png
│   │   └── Bob/
│   │       └── img01.jpg
│   ├── unknown_faces/        ← optional; used by CLI batch mode
│   │   └── test1.jpg
│   └── .lbph_cache/          ← created automatically (embeddings + thumbnails)
├── image_data/               ← still images for --eval-face
├── frames/                   ← frame sequences for --eval-person
└── output/                   ← annotated images and CSV saved here
```

**Rules for `known_faces/`:**

- One subfolder per person; the folder name becomes the label.
- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Each image must contain a **detectable frontal face**. Images with no detected face are skipped with a warning.
- For CLI batch testing, the resolver looks for `unknown_faces`, `unknown_face`, `unknown`, or `test` under `data/`.

**The `.lbph_cache/` folder** is the model cache directory. The name is kept for backward compatibility; the recognizer now uses deep embeddings (not LBPH). It holds:

| File | Contents |
|------|----------|
| `embeddings.npz` | `(N, 128)` encoding matrix + `(N,)` label array |
| `meta.json` | SHA-256 fingerprint, pipeline version, label-to-name map, threshold |
| `thumb_<id>.png` | Representative 96×96 thumbnail per person |

---

## Web application

### Start

```bash
cd /path/to/face_rec
source .venv/bin/activate
python app.py
```

Open the URL printed in the terminal (defaults to **`http://127.0.0.1:5050`**; auto-scans 5050–5079 for a free port). macOS reserves port 5000 for AirPlay Receiver — this project avoids it.

### What you can do

1. **Count faces** — upload a photo; get a count and an annotated preview (dlib HOG/CNN frontal detector + CLAHE).
2. **Count people** — upload a surveillance/CCTV frame; person bounding boxes via OpenCV HOG pedestrian detector.
3. **Recognize** — upload a portrait; get the most likely match from `known_faces`, embedding distance, confidence verdict, and reference thumbnails.
4. **Webcam snapshot** — allow camera access, preview locally, press **Capture** (single frame — no streaming loop).
5. **Rebuild model cache** — click after adding/removing/changing training images to recompute embeddings.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `FACE_REC_DATA_DIR` | `data` | Root folder containing `known_faces/`. |
| `FACE_MODEL_CACHE` | `<data>/.lbph_cache` | Where embeddings, `meta.json`, and thumbnails are stored. |
| `FACE_REC_THRESHOLD` | `0.55` | Embedding L2 distance ≤ threshold = confident match. Range ~0–1; lower = stricter. |
| `FACE_DETECT_DOWNSCALE` | — | Kept for API compatibility; not used by the deep-embedding recognizer. |
| `FACE_REC_FORCE_RETRAIN` | unset | Set to `1`/`true` to ignore cache on next load. |
| `FACE_COUNT_MODEL` | `hog` | `hog` (fast, CPU) or `cnn` (slower, more accurate). |
| `FACE_COUNT_SENSITIVE` | `0` | `1`/`true`: 2× upsample + larger max-side for small/distant faces. |
| `FACE_COUNT_ENHANCE` | `1` | `0`/`false`: disable CLAHE contrast enhancement. |
| `FACE_COUNT_SHARPEN` | `0` | `1`/`true`: unsharp-mask sharpening (helps blurry images). |
| `FACE_COUNT_MIN_FACE` | `20` | Minimum face height/width in px; smaller boxes are discarded. |
| `FACE_COUNT_NMS_IOU` | `0.35` | IoU threshold for duplicate-detection merging (NMS). |
| `FACE_EVAL_API` | unset | Set to `1`/`true` to enable `POST /api/eval_face` and `/api/eval_person`. |
| `FACE_EVAL_IMAGES_DIR` | `image_data` | Default still-image folder for `/api/eval_face`. |
| `FACE_EVAL_FRAMES_DIR` | `frames` | Default frame folder for `/api/eval_person`. |
| `FACE_EVAL_FACE_LABELS` | auto | Default CSV path for face-count ground-truth labels. |
| `FACE_EVAL_DEFAULT_LIMIT` | `200` | Default sample size for eval API calls. |
| `FACE_EVAL_MAX` | `2000` | Hard cap on eval sample size per request. |
| `PORT` | unset | If unset: `127.0.0.1:5050–5079`. If set: `0.0.0.0:<PORT>` (cloud). |
| `FLASK_DEBUG` | auto | Without `PORT`: **on**. With `PORT`: **off** unless `FLASK_DEBUG=1`. |

**Example:**

```bash
export FACE_REC_DATA_DIR=/path/to/my_dataset
export FACE_REC_THRESHOLD=0.5   # stricter than default
python app.py
```

### HTTP API

All `POST` endpoints accept multipart form field **`image`** unless noted.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | JSON: paths, cache presence, eval config |
| `POST` | `/api/count` | Frontal **face** count + base64 JPEG overlay |
| `POST` | `/api/count_people` | **Person/pedestrian** count + overlay (OpenCV HOG) |
| `POST` | `/api/recognize` | Recognize against `known_faces`; returns name, distance, thumbnails |
| `POST` | `/api/snapshot_identify` | One JPEG frame → largest face → most likely identity |
| `POST` | `/api/retrain` | Force re-embedding and refresh cache |
| `POST` | `/api/eval_face` | Train/test metrics for still images vs CSV (requires `FACE_EVAL_API=1`) |
| `POST` | `/api/eval_person` | Train/test metrics for frame sequences vs CSV (requires `FACE_EVAL_API=1`) |

**Recognition response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `face_found` | bool | `false` if no face was detected |
| `name` | string | Most likely person |
| `distance` | float | L2 embedding distance (0–1+; lower = more similar) |
| `is_match` | bool | `distance ≤ FACE_REC_THRESHOLD` |
| `verdict` | string | Human-readable result string |
| `reference_large` | string | 256×256 JPEG data URL of the representative training crop |
| `similar_thumb` | string | 96×96 JPEG data URL |
| `similar_training_file` | string | Filename of the matched training image |

---

## Command line

### Face recognition (`face_rec.py`)

```bash
python face_rec.py
```

Interactive menu:

```
1. Batch predict — all images in the resolved unknown_* folder
2. Single image  — enter a path, get verdict + distance
3. Webcam        — press SPACE to capture once; result window appears
4. Evaluate      — accuracy metrics where expected label comes from filename
5. Force retrain — ignore cache, recompute embeddings, save
6. Exit
```

The first run (or after data changes) computes face embeddings and writes `data/.lbph_cache/`. Subsequent runs load from cache instantly when the file fingerprint is unchanged.

### Face & person counting (`face_count.py`)

```bash
# Frontal-face detector — portraits and well-lit photos
python face_count.py [--model hog|cnn] [--sensitive] [--sharpen] [--no-enhance]

# Person/pedestrian detector — CCTV / overhead footage
python face_count.py --detector person [--method hog|mog2|combined]

# Evaluation against a ground-truth CSV
python face_count.py --eval-face  --images-dir image_data --labels-csv labels.csv
python face_count.py --eval-person --frames-dir frames    --labels-csv labels.csv
```

**Face detector** interactive menu: single image → `output/`, folder batch, live webcam.

**Person detector** interactive menu:

| Choice | What it does |
|--------|-------------|
| 1. Single image | HOG pedestrian detection; saves annotated copy to `output/`. |
| 2. Frame sequence | Process a sorted folder in temporal order. Annotated frames → `output/annotated/`; per-frame counts → `output/counts.csv`. |
| 3. Webcam | Live HOG person counting with temporal smoothing (press **q** or **Esc** to quit). |

**Detection methods for frame sequences (`--method`):**

| Method | How it works | When to use |
|--------|-------------|-------------|
| `hog` | Per-frame HOG pedestrian detector; no warmup. | Any footage; fastest. |
| `mog2` | MOG2 background subtraction + blob analysis; 30-frame warmup. | Static camera, slow-moving crowd. |
| `combined` | HOG detections filtered by MOG2 foreground mask, plus uncovered blob detections. | Sequential surveillance footage; best accuracy. |

**Evaluation flags:**

| Flag | Purpose |
|------|---------|
| `--eval-face` | Train/test MAE, RMSE, exact-match rate for still images vs a `file,count` CSV. |
| `--eval-person` | Same metrics for a frame-sequence folder. |
| `--tune-hparams` | Grid-search `sensitive_counting` + `min_face_size` to minimise train MAE. |
| `--train-ratio N` | Fraction used for training metrics (default `0.8`). |
| `--split random\|chronological` | How to split train/test (default `random`). |
| `--limit N` | Cap the number of labeled samples (useful for smoke tests). |
| `--eval-report FILE` | Write the JSON metrics report to a file. |

---

## How recognition works

1. **Training** — For each person folder under `known_faces/`, dlib's HOG detector locates faces. A 128-dimensional embedding vector is computed per face using a ResNet trained on millions of faces. All vectors are stored with their person label.

2. **Prediction** — The same embedding is computed for the query image. The closest match is found via vectorized L2 distance across the full training matrix — nearest neighbour in 128-d space.

3. **Threshold** — If `distance ≤ FACE_REC_THRESHOLD` (default `0.55`), the result is reported as a confident match. A *most likely* name is always returned even when the distance exceeds the threshold.

4. **Cache** — After training, embeddings are written to `.lbph_cache/embeddings.npz` along with `meta.json` and per-person thumbnails. The cache is keyed by a SHA-256 fingerprint of all training image paths, sizes, and mtimes. It is invalidated automatically when images are added, removed, or modified.

**Tuning the threshold:**

- Everyone shows "best guess only" → **raise** (e.g. `0.65`).
- Too many false matches → **lower** (e.g. `0.45`).

---

## Deploying

**GitHub Pages** serves static files only and cannot run this Python server.

| Approach | Steps |
|----------|-------|
| **Container (recommended)** | The repo includes a `Dockerfile`. Build, push to a registry (e.g. GHCR), and run on Render, Fly.io, Railway, or your own VM. |
| **Native Python** | Install dependencies on the host, set `PORT`, run `gunicorn` (see `Procfile`). Ensure `data/known_faces/` is present on the server. |

```bash
# Build and run locally with Docker
docker build -t face-studio .
docker run --rm -p 8080:8080 -e PORT=8080 face-studio
# open http://localhost:8080/
```

**Production defaults:** When `PORT` is set, the server listens on `0.0.0.0` and debug mode is **off** unless `FLASK_DEBUG=1`.

**Training data in Docker:** `data/known_faces/` is copied into the image; `unknown_faces/` is not. Adjust the `Dockerfile` `COPY` lines if you need additional data at runtime, or mount a volume:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 \
  -v /path/to/my_data:/app/data \
  face-studio
```

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `No module named 'flask'` | Run `pip install -r requirements.txt` inside your venv. |
| `face_recognition` / dlib install fails | Install build tools first: Xcode CLI Tools on Mac (`xcode-select --install`), Visual Studio Build Tools + CMake on Windows. See [dlib docs](http://dlib.net/). |
| `opencv-contrib-python` vs `opencv-python` conflict | Run `pip uninstall opencv-python opencv-contrib-python` then `pip install opencv-contrib-python`. |
| `Missing known_faces folder` | Create `data/known_faces/<Name>/` and add at least one frontal-face image per person. |
| First training is slow | Normal — dlib encodes every training image. Subsequent runs load from cache in milliseconds if data is unchanged. |
| Browser camera blocked | Grant permission for `http://127.0.0.1:5050` in your browser's site settings. |
| Port 5000 already in use (Mac) | Disable **AirPlay Receiver** in System Settings → General → AirDrop & Handoff, or set `PORT=5051 python app.py`. |
| Training image skipped | The console prints `[skip] No face detected` — ensure the image contains a clear, well-lit frontal face and is not blurry. |
| Wrong person predicted | Add more varied photos per person (different lighting, angles); adjust `FACE_REC_THRESHOLD`; rebuild the cache with `/api/retrain` or menu option 5. |
| Count is wrong / misses distant faces | Enable `FACE_COUNT_SENSITIVE=1` (2× upsample) and/or `FACE_COUNT_ENHANCE=1` (CLAHE). |

---

## Privacy

- All image processing runs **locally** in your Flask process on `127.0.0.1`.
- No image data is sent to any third-party face API.
- Training embeddings and the cache live entirely on disk under `data/` and `FACE_MODEL_CACHE`.

---

## License / credits

Application code in this repository is original work. Third-party libraries (`face_recognition`, OpenCV, Flask, dlib, etc.) retain their own licenses — see their respective projects for details.
