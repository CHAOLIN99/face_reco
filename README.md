# Face Studio

A local toolkit for **counting faces** in images and **recognizing** who someone is against your own photo gallery. Includes a **Flask web UI** (Face Studio) and **command-line** scripts — no cloud API, no third-party face service.

> When you run the app **locally**, uploads go only to your Flask process on **loopback** (`127.0.0.1`).  
> When **deployed** to a public host, uploads go to that server — HTTPS, access control, and data policies are your responsibility.

---

## What it does

| File | Role |
|------|------|
| **`face_count.py`** | Two counters in one module: `FaceCounterSystem` (dlib HOG/CNN frontal faces, CLAHE + NMS) and `PersonCounterSystem` (OpenCV HOG pedestrian + MOG2 background subtraction — for CCTV / overhead footage). |
| **`face_rec.py`** | Trains a **deep embedding** recognizer (dlib 128-d ResNet via `face_recognition`) on folders of named people, predicts the **most likely** match, and caches embeddings so retraining is skipped when data is unchanged. |
| **`app.py`** | Flask **web UI** plus JSON APIs for face count, person count, recognize, one-shot webcam identify, and forced retrain. |

---

## Requirements

- **Python 3.10+** (3.11+ recommended).
- **macOS / Linux / Windows** (paths below use macOS/Linux style).
- **Webcam** optional (only for the "snapshot identify" webcam flow and CLI webcam modes).

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes:

- **`face-recognition`** — dlib-based 128-d face embeddings used for both detection and recognition. Requires [CMake](https://cmake.org/) and a C++ compiler to build dlib on some systems (see Troubleshooting).
- **`opencv-contrib-python`** — OpenCV with extra modules for image processing and the person/pedestrian counter.
- **`numpy`**, **`pillow`**, **`flask`**, **`gunicorn`**

---

## Folder layout

Put training and test images under a `data` directory (or override with `FACE_REC_DATA_DIR`).

```text
face_rec/
  data/
    known_faces/           ← required for recognition
      Alice/
        photo1.jpg
        photo2.png
      Bob/
        img01.jpg
    unknown_faces/         ← optional; used by CLI batch mode
      test1.jpg
  data/.lbph_cache/        ← created automatically (embeddings + thumbnails)
```

**Rules:**

- One subfolder per person under `known_faces/`; the folder name becomes the label.
- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Each training image should contain a **detectable frontal face**. Images with no detected face are skipped with a printed warning.
- For batch "unknown" testing, the CLI resolver looks for: `unknown_faces`, `unknown_face`, `unknown`, or `test` under `data/`.

---

## Web application (recommended)

### Start the server

```bash
cd /path/to/face_rec
source .venv/bin/activate
python app.py
```

Open the URL printed in the terminal (defaults to **5050**; auto-scans to the next free port). macOS uses **5000** for AirPlay Receiver — this project intentionally avoids it.

### What you can do in the UI

1. **Count faces** — upload a photo; get a count and an annotated preview (dlib frontal-face detector).
2. **Count people** — upload a surveillance / CCTV frame; person bounding boxes via OpenCV HOG pedestrian detector.
3. **Recognize** — upload a portrait; get the most likely person from `known_faces`, embedding distance, and reference thumbnails.
4. **Webcam snapshot** — allow camera access, preview locally, then **Capture & identify** (single frame — no streaming loop).
5. **Rebuild model cache** — click after adding/removing/changing training images to recompute embeddings.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `FACE_REC_DATA_DIR` | `data` | Root folder containing `known_faces/`. |
| `FACE_MODEL_CACHE` | `<data>/.lbph_cache` | Where embeddings (`embeddings.npz`), `meta.json`, and thumbnails are stored. |
| `FACE_REC_THRESHOLD` | `0.55` | Embedding distance ≤ threshold = confident match. Range is roughly 0–1; lower = stricter. |
| `FACE_DETECT_DOWNSCALE` | `0.55` | Kept for API compatibility; not used by the deep embedding recognizer. |
| `FACE_REC_FORCE_RETRAIN` | unset | Set to `1`/`true` to ignore cache on next load. |
| `FACE_COUNT_MODEL` | `hog` | `hog` (CPU-friendly) or `cnn` (slower, more accurate). |
| `FACE_COUNT_SENSITIVE` | `0` | `1`/`true`: 2× upsample + larger max-side for small/distant faces. |
| `FACE_COUNT_ENHANCE` | `1` | `0`/`false`: disable CLAHE contrast enhancement. |
| `FACE_COUNT_SHARPEN` | `0` | `1`/`true`: enable unsharp-mask sharpening (helps blurry images). |
| `FACE_COUNT_MIN_FACE` | `20` | Minimum face height/width in px; smaller boxes are discarded. |
| `PORT` | unset | If unset: Flask binds `127.0.0.1:5050–5079`. If set (cloud): binds `0.0.0.0`. |
| `FLASK_DEBUG` | depends | Without `PORT`: default **on**. With `PORT`: default **off**. |

Example:

```bash
export FACE_REC_DATA_DIR=/Users/me/my_dataset
export FACE_REC_THRESHOLD=0.5   # stricter than default 0.55
python app.py
```

### HTTP API

All POST bodies use multipart form field **`image`** unless noted.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | JSON: paths, cache presence, `known_faces` exists |
| POST | `/api/count` | Frontal **face** count + base64 JPEG overlay (dlib HOG/CNN) |
| POST | `/api/count_people` | **Person/pedestrian** count + overlay (OpenCV HOG — for CCTV/overhead footage) |
| POST | `/api/recognize` | Recognize with optional `reference_large` / `similar_thumb` data URLs |
| POST | `/api/snapshot_identify` | One JPEG frame → largest face → most likely identity |
| POST | `/api/retrain` | Force re-embedding and refresh cache |

Successful recognize-style responses include: `face_found`, `name`, `distance`, `is_match`, `verdict`, `reference_large`, `similar_training_file`, etc.

`distance` is an L2 embedding distance (0–1+ range; lower = more similar). A value ≤ `FACE_REC_THRESHOLD` (default `0.55`) is reported as `is_match: true`.

---

## Command line

### Face recognition (`face_rec.py`)

```bash
python face_rec.py
```

Interactive menu:

1. Batch predict all images in the resolved `unknown_*` folder
2. Predict a single image path
3. Webcam: press **SPACE** to capture once, result window appears
4. Force retrain and save cache
5. Exit

The first run (or after data changes) computes face embeddings and writes `data/.lbph_cache`. Subsequent runs load the cache instantly if the file fingerprint is unchanged.

### Face counting (`face_count.py`)

```bash
# Frontal-face detector (dlib HOG/CNN) — portraits and well-lit photos
python face_count.py [--model hog|cnn] [--sensitive] [--sharpen] [--no-enhance]

# Person/pedestrian detector (OpenCV HOG + optional MOG2) — CCTV / overhead footage
python face_count.py --detector person [--method hog|mog2|combined] [--no-save-frames] [--no-csv]
```

**Face detector** menu: single image → `output/`, folder batch, live webcam.

**Person detector** menu:
1. **Single image** — HOG pedestrian detection; saves annotated copy to `output/`.
2. **Frame sequence** — process a sorted folder (e.g. `frames/`) in order. Supports three methods:
   - `hog` — per-frame only, no warmup needed.
   - `mog2` — background subtraction blobs; 30-frame warmup, then very fast.
   - `combined` *(default)* — HOG filtered by MOG2 foreground; best accuracy on surveillance footage.
   Saves annotated frames to `output/annotated/` and a per-frame `output/counts.csv`.
3. **Webcam** — live HOG person counting with temporal smoothing.

---

## How recognition works

1. **Training** — For each person's folder under `known_faces/`, dlib's HOG detector finds faces in each image. A 128-dimensional embedding vector is computed per face using a ResNet trained on millions of faces. All embeddings are stored alongside the person label.
2. **Prediction** — The same embedding is computed for the query image. The closest match is found via L2 (Euclidean) distance across all training embeddings — nearest neighbour in 128-d space.
3. **Threshold** — If distance ≤ `FACE_REC_THRESHOLD` (default `0.55`), the UI treats it as a confident match. You always get a *most likely* name even if the distance is above threshold.
4. **Cache** — Embeddings are saved to `data/.lbph_cache/embeddings.npz` alongside `meta.json` and representative thumbnails. The cache is keyed by a SHA-256 fingerprint of all training images; it is invalidated automatically when images are added, removed, or modified.

Tune `FACE_REC_THRESHOLD` if:
- Everyone shows "best guess only" → try **raising** the threshold (e.g. `0.65`).
- You're getting too many false matches → try **lowering** it (e.g. `0.45`).

---

## Deploying

**GitHub Pages** serves static files only — it cannot run this Python server.

| Approach | Steps |
|----------|-------|
| **Container (recommended)** | The repo includes a `Dockerfile`. Build and push to GitHub Container Registry (GHCR), then run on Render, Fly.io, Railway, or your own VM. |
| **Native Python** | Install dependencies on the host, set `PORT`, run `gunicorn` (see `Procfile`). Ensure `data/known_faces` is present on the server. |

```bash
# Pull from GHCR (replace OWNER/REPO):
docker pull ghcr.io/OWNER/face_rec:latest
docker run --rm -p 8080:8080 -e PORT=8080 ghcr.io/OWNER/face_rec:latest
```

Then open `http://localhost:8080/`.

**Production defaults:** When `PORT` is set, the app listens on `0.0.0.0` and debug mode is **off** unless `FLASK_DEBUG=1`.

**Docker image size:** `data/known_faces` is copied into the image but `unknown_faces` is not. Adjust the `Dockerfile` if you need extra data at runtime.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `No module named 'flask'` | `pip install -r requirements.txt` inside your venv. |
| `cv2.face module not found` | Reinstall **`opencv-contrib-python`**; remove conflicting `opencv-python` if present. (Note: `cv2.face` is no longer used for recognition, but `opencv-contrib-python` is still required for other features.) |
| `Missing known_faces folder` | Create `data/known_faces/<Name>/` and add at least one image per person. |
| First training is slow | Normal for large galleries; later runs load from cache if unchanged. |
| Browser camera blocked | Grant permission for your local URL (`http://127.0.0.1:5050`). |
| `Address already in use` on 5000 | On Mac, disable **AirPlay Receiver** in System Settings → General → AirDrop & Handoff, or use a different port (`PORT=5050 python app.py`). |
| `face_recognition` / dlib install fails | Install build tools (Xcode CLI on Mac, Visual Studio Build Tools on Windows) and CMake; see dlib docs. |
| Wrong person predicted | Add more varied photos per person; improve lighting; adjust `FACE_REC_THRESHOLD` (default `0.55`); rebuild the cache. |
| Training image skipped | The image printed "[skip] No face detected" — ensure it contains a clear frontal face and is not blurry/dark. |

---

## Privacy

- The web UI communicates with your **local** Flask process on `127.0.0.1`.
- No third-party face API is used.
- Training data and the embedding cache live entirely on disk under your `data` and `FACE_MODEL_CACHE` paths.

---

## License / credits

Application code in this repository is original work. Third-party libraries (`face_recognition`, OpenCV, Flask, dlib, etc.) have their own licenses — see their respective projects for details.
