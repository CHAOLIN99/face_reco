# Face Studio

A small local toolkit for **counting faces** in images and **recognizing** who someone most likely is, using your own photo gallery. It includes a **Flask web UI** (Face Studio) and **command-line** scripts.

Everything runs on your computer. Images you upload in the browser are sent only to your local Flask server (`127.0.0.1`), not to the cloud.

---

## What it does

| Piece | Role |
|--------|------|
| **`face_count.py`** | Counts faces using the [`face_recognition`](https://github.com/ageitgey/face_recognition) library (dlib HOG by default; optional CNN). |
| **`face_rec.py`** | Trains an OpenCV **Haar cascade + LBPH** recognizer on folders of named people, predicts the **most likely** match, and caches the model so you do not retrain on every run. |
| **`app.py`** | **Web UI** plus JSON APIs for count, recognize, one-shot webcam identify, and forced retrain. |

---

## Requirements

- **Python 3.10+** (3.11+ recommended; check what your `venv` uses).
- **macOS / Linux / Windows** (paths below use macOS/Linux style).
- **Webcam** optional (only for the browser “snapshot identify” flow and CLI webcam modes).

### Python packages

Install from the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes:

- **`opencv-contrib-python`** — needed for `cv2.face` / LBPH (do not use plain `opencv-python` alone for recognition).
- **`face-recognition`** — wraps dlib for counting (may need [CMake](https://cmake.org/) and a compiler to build dlib on some systems).
- **`numpy`**, **`pillow`**, **`flask`**

If `import cv2` works but `cv2.face` is missing, reinstall **`opencv-contrib-python`** and ensure you did not overwrite it with `opencv-python` only.

---

## Folder layout (data)

Put training and test images under a `data` directory (or set `FACE_REC_DATA_DIR`).

```text
face_rec/
  data/
    known_faces/           # required for recognition
      Alice/
        photo1.jpg
        photo2.png
      Bob/
        img01.jpg
    unknown_faces/         # optional; used by CLI batch mode (name may vary)
      test1.jpg
  data/.lbph_cache/        # created automatically — LBPH model + thumbnails
```

**Rules:**

- **`known_faces/<Person Name>/`** — one subfolder per person; folder name becomes the label returned by the app.
- Supported extensions: **`.jpg`**, **`.jpeg`**, **`.png`**, **`.webp`**.
- Each training image should contain a **detectable frontal face** (OpenCV Haar during training). Images with no detected face are skipped.

For batch “unknown” testing in the CLI, the resolver looks for (in order): `unknown_faces`, `unknown_face`, `unknown`, or `test` under `data/`.

---

## Web application (recommended)

### Start the server

```bash
cd /path/to/face_rec
source .venv/bin/activate
python app.py
```

Open the URL printed in the terminal (defaults to **5050**; if that port is busy, the app picks the next free one). macOS often uses **5000** for AirPlay Receiver, so this project avoids 5000 by default.

### What you can do in the UI

1. **Count faces** — upload an image; get a count and a preview with overlays.
2. **Recognize** — upload a portrait; get the **most likely** person from `known_faces`, LBPH distance, and reference thumbnails.
3. **Webcam snapshot** — allow camera access, preview locally, then **Capture & identify** once (single frame to the server, no live streaming loop).
4. **Rebuild model cache** — after adding/removing/changing training images, click this (or use the API below) so LBPH retrains and saves a new cache.

### Environment variables (web app)

| Variable | Default | Meaning |
|----------|---------|---------|
| `FACE_REC_DATA_DIR` | `data` | Root folder containing `known_faces`. |
| `FACE_MODEL_CACHE` | `<data>/.lbph_cache` | Where the LBPH model and `meta.json` are stored. |
| `FACE_REC_THRESHOLD` | `72` (in code) | LBPH distance at or below = “confident” match; higher = looser. |
| `FACE_DETECT_DOWNSCALE` | `0.45` | Haar runs on a resized frame for speed (0.3–0.6 typical). |
| `FACE_REC_FORCE_RETRAIN` | unset | Set to `1` / `true` to ignore cache on next load (also used internally by retrain API). |
| `FACE_COUNT_MODEL` | `hog` | `hog` (CPU-friendly) or `cnn` (heavier; better quality if you have GPU support). |
| `FACE_COUNT_SENSITIVE` | `0` | `1` / `true`: larger image + 2× upsample for small/distant faces (still dlib-only; fewer false positives than older Haar merge). |
| `PORT` | (unset) | If unset, Flask tries 5050–5079, then an ephemeral port on loopback. Set explicitly (e.g. `export PORT=8000`) to force one port (fails if in use). |

Example:

```bash
export FACE_REC_DATA_DIR=/Users/me/my_dataset
export FACE_REC_THRESHOLD=80
python app.py
```

### HTTP API (for scripts or tools)

All POST bodies use multipart form field **`image`** unless noted.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | JSON: paths, cache presence, `known_faces` exists |
| POST | `/api/count` | Face count + base64 PNG overlay |
| POST | `/api/recognize` | Full recognize with optional `reference_large` / `similar_thumb` data URLs |
| POST | `/api/snapshot_identify` | One JPEG frame → largest face → most likely identity |
| POST | `/api/retrain` | Force LBPH retrain and refresh cache |

Successful recognize-style responses include fields such as: `face_found`, `name`, `distance`, `is_match`, `verdict`, `reference_large`, `similar_training_file`, etc.

---

## Command line

### Face recognition (`face_rec.py`)

```bash
python face_rec.py
```

Interactive menu:

1. Batch predict all images in the resolved `unknown_*` folder  
2. Predict a single image path  
3. Webcam: **SPACE** to capture once, then a result window  
4. Force retrain and save cache  
5. Exit  

The first run (or after data changes) builds or loads **`data/.lbph_cache`**: if the fingerprint of files under `known_faces` matches `meta.json`, the cached model loads instantly.

### Face counting (`face_count.py`)

```bash
python face_count.py
```

Options: single image (writes under `output/`), folder batch, or live webcam count.

---

## How recognition works (short)

1. **Training:** Haar detects a face in each training image; the crop is resized to 200×200 grayscale and fed to **LBPH** with the folder name as the label.  
2. **Prediction:** Same pipeline on a query image or frame; LBPH returns the **closest** label and a **distance** (lower = more similar).  
3. **Threshold:** If distance ≤ `match_threshold`, the UI treats it as a stronger match; you still always get a **most likely** name.

Tune `FACE_REC_THRESHOLD` if everyone is “unknown” (try a higher value) or wrong people match (try lower).

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `No module named 'flask'` | `pip install flask` (or `pip install -r requirements.txt`) inside your venv. |
| `cv2.face module not found` | Install **`opencv-contrib-python`**; remove conflicting `opencv-python` if needed. |
| `Missing known_faces folder` | Create `data/known_faces/<Name>/` and add images. |
| Training is slow the first time | Normal for large galleries; later runs use **cache** if files unchanged. |
| Browser camera blocked | Grant permission for your local URL (e.g. `http://127.0.0.1:5050`). |
| `Address already in use` on port 5000 | On Mac, turn off **AirPlay Receiver** (System Settings → General → AirDrop & Handoff), or run with `PORT=5050` (already the default in this project). |
| `face_recognition` / dlib install fails | Install build tools (Xcode CLI tools on Mac, Visual Studio Build Tools on Windows) and CMake; see dlib / face_recognition docs. |
| Wrong person predicted | Add more varied photos per person; improve lighting; adjust threshold; retrain cache. |

---

## Privacy

- The web UI talks to **your** Flask process on **localhost**.  
- No third-party face API is used by this project.  
- Training data and the LBPH cache live on disk under your **`data`** and **`FACE_MODEL_CACHE`** paths.

---

## License / credits

This repository bundles application code. Third-party libraries (`face_recognition`, OpenCV, Flask, dlib, etc.) have their own licenses. See their respective projects for details.
