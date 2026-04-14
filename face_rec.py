from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bump this when changing crop/detection/training pipeline so the cache is
# invalidated automatically.
RECOGNIZER_PIPELINE_VERSION = "2026-04-14-deepembedding-v1"


def fingerprint_known_faces(known_faces_dir: str) -> str:
    """SHA-256 hash of all training image paths + sizes + mtimes.
    Used to skip redundant retraining when data is unchanged."""
    root = Path(known_faces_dir)
    if not root.is_dir():
        return ""
    lines: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        lines.append(f"{rel}\t{st.st_size}\t{st.st_mtime_ns}")
    lines.append(f"__pipeline__\t{RECOGNIZER_PIPELINE_VERSION}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _natural_sort_key(name: str) -> list[object]:
    """Sane ordering for filenames containing numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: _natural_sort_key(p.name),
    )


def expected_label_from_filename(filename: str) -> str:
    """Ground-truth label from an unknown-face filename.

    Supports common patterns:
    - ``Name_12.jpg``   -> ``Name``
    - ``Name-12.png``   -> ``Name``
    - ``Name 12.webp``  -> ``Name``
    - otherwise uses the stem as-is.
    """
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[_\-\s]\d+$", "", stem).strip()
    return stem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceMatchDetail:
    """Deep-embedding prediction result for a single face."""

    name: str
    distance: float       # L2 embedding distance; lower = more similar (0–1+ range)
    is_match: bool
    verdict: str
    similar_thumb_bgr: np.ndarray | None
    similar_source: str | None


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class SimpleFaceRecognizer:
    """
    Face recognizer using 128-d dlib deep embeddings via the face_recognition library.

    Training images must be organized as:
        data/known_faces/<Person Name>/*.jpg|jpeg|png|webp

    A model cache (default ``data/.lbph_cache``) is written after training
    so subsequent runs skip retraining when the data hasn't changed.

    Distance is the L2 Euclidean distance between 128-d face embeddings.
    A distance ≤ DEFAULT_MATCH_THRESHOLD (0.55) is considered a confident match.
    """

    DEFAULT_MATCH_THRESHOLD: float = 0.55   # 0.6 is the face_recognition default; 0.55 is tighter
    THUMB_SIZE: int = 96
    DISPLAY_MATCH_SIZE: int = 256
    DEFAULT_DETECT_DOWNSCALE: float = 0.55  # kept for API compat; not used internally
    DEFAULT_CROP_MARGIN_FRAC: float = 0.12  # kept for API compat; not used internally

    def __init__(
        self,
        cascade_path: str | None = None,    # kept for API compat; not used
        match_threshold: float | None = None,
        detect_downscale: float | None = None,
    ) -> None:
        try:
            import face_recognition as _fr
            self._fr = _fr
        except ImportError as e:
            raise RuntimeError(
                "face_recognition not installed. Run: pip install face-recognition"
            ) from e

        self.match_threshold = (
            float(match_threshold) if match_threshold is not None
            else self.DEFAULT_MATCH_THRESHOLD
        )
        self.detect_downscale = (
            float(detect_downscale) if detect_downscale is not None
            else self.DEFAULT_DETECT_DOWNSCALE
        )

        self.label_to_name: dict[int, str] = {}
        self.trained: bool = False

        # Flat parallel lists — index i corresponds to the same training sample.
        self._enc_list: list[np.ndarray] = []   # each shape (128,)
        self._lbl_list: list[int] = []

        # Representative thumbnail per label (first detected face per person)
        self._rep_thumb_by_label: dict[int, tuple[np.ndarray, str]] = {}
        self._cache_loaded: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _make_thumb_from_location(
        self, img_bgr: np.ndarray, location: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop face bbox to a THUMB_SIZE × THUMB_SIZE BGR thumbnail."""
        top, right, bottom, left = location
        h_img, w_img = img_bgr.shape[:2]
        top    = max(0, top)
        left   = max(0, left)
        bottom = min(h_img, bottom)
        right  = min(w_img, right)
        crop = img_bgr[top:bottom, left:right]
        if crop.size == 0:
            return np.zeros((self.THUMB_SIZE, self.THUMB_SIZE, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.THUMB_SIZE, self.THUMB_SIZE))

    @staticmethod
    def _face_area(loc: tuple[int, int, int, int]) -> int:
        """Return pixel area of a (top, right, bottom, left) face location."""
        top, right, bottom, left = loc
        return max(0, (bottom - top) * (right - left))

    def _detect_all(
        self, img_bgr: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """Detect all faces in a BGR image.
        Returns a list of (location, 128-d encoding) pairs sorted largest-face-first."""
        rgb = self._bgr_to_rgb(img_bgr)
        locations = self._fr.face_locations(rgb, model="hog")
        if not locations:
            return []
        encodings = self._fr.face_encodings(rgb, known_face_locations=locations)
        pairs = list(zip(locations, encodings))
        pairs.sort(key=lambda p: self._face_area(p[0]), reverse=True)
        return pairs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _collect_images_from_folder(self, labeled_faces_root_dir: str) -> None:
        """Walk known_faces/<Person>/ subdirectories, compute face embeddings,
        and populate internal state."""
        dataset_path = Path(labeled_faces_root_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"known_faces folder does not exist: {labeled_faces_root_dir}"
            )

        label_id = 0
        self.label_to_name.clear()
        self._enc_list.clear()
        self._lbl_list.clear()
        self._rep_thumb_by_label.clear()

        for person_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
            person_name = person_dir.name
            found_any = False

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [skip] Cannot read {img_path.name}")
                    continue

                pairs = self._detect_all(img)
                if not pairs:
                    print(f"  [skip] No face detected in {img_path.name}")
                    continue

                if not found_any:
                    self.label_to_name[label_id] = person_name
                    found_any = True

                # Use the largest detected face
                loc, enc = pairs[0]

                self._enc_list.append(enc)
                self._lbl_list.append(label_id)

                if label_id not in self._rep_thumb_by_label:
                    thumb = self._make_thumb_from_location(img, loc)
                    self._rep_thumb_by_label[label_id] = (thumb, img_path.name)

            if found_any:
                label_id += 1
            else:
                print(f"  [warn] No usable images for '{person_name}' — skipped.")

        if not self._enc_list:
            raise RuntimeError(
                "No training faces found. "
                "Check that known_faces/<Name>/ contains images with detectable frontal faces."
            )

    def train_from_known_faces(self, known_faces_dir: str) -> None:
        self._collect_images_from_folder(known_faces_dir)
        self.trained = True
        self._cache_loaded = False

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def save_cache(self, cache_dir: str | Path, fingerprint: str) -> None:
        """Write embeddings (.npz), meta.json, and per-label reference thumbnails."""
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)

        if self._enc_list:
            enc_array = np.stack(self._enc_list, axis=0)          # (N, 128)
            lbl_array = np.array(self._lbl_list, dtype=np.int32)  # (N,)
        else:
            enc_array = np.empty((0, 128), dtype=np.float64)
            lbl_array = np.empty((0,), dtype=np.int32)

        np.savez_compressed(str(d / "embeddings.npz"), encodings=enc_array, labels=lbl_array)

        thumb_sources: dict[str, str] = {}
        for lid, (thumb, src) in self._rep_thumb_by_label.items():
            thumb_sources[str(lid)] = src
            cv2.imwrite(str(d / f"thumb_{lid}.png"), thumb)

        meta = {
            "fingerprint": fingerprint,
            "pipeline_version": RECOGNIZER_PIPELINE_VERSION,
            "label_to_name": {str(k): v for k, v in sorted(self.label_to_name.items())},
            "thumb_source": thumb_sources,
            "match_threshold": self.match_threshold,
            "detect_downscale": self.detect_downscale,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_cache(self, cache_dir: str | Path, expected_fingerprint: str) -> bool:
        """Load cached embeddings if the fingerprint matches. Returns True on success."""
        d = Path(cache_dir)
        meta_path = d / "meta.json"
        emb_path  = d / "embeddings.npz"
        if not meta_path.is_file() or not emb_path.is_file():
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        if meta.get("fingerprint") != expected_fingerprint:
            return False
        if meta.get("pipeline_version") != RECOGNIZER_PIPELINE_VERSION:
            return False

        try:
            data = np.load(str(emb_path))
            enc_array = data["encodings"]
            lbl_array = data["labels"]
        except Exception:
            return False

        self.label_to_name = {int(k): v for k, v in meta["label_to_name"].items()}
        self.match_threshold = float(meta.get("match_threshold", self.DEFAULT_MATCH_THRESHOLD))
        self.detect_downscale = float(meta.get("detect_downscale", self.DEFAULT_DETECT_DOWNSCALE))

        self._enc_list = [enc_array[i] for i in range(len(enc_array))]
        self._lbl_list = list(lbl_array.astype(int))

        self._rep_thumb_by_label.clear()
        thumb_src_map: dict[str, str] = meta.get("thumb_source") or {}
        for lid in self.label_to_name:
            tp = d / f"thumb_{lid}.png"
            if tp.is_file():
                im = cv2.imread(str(tp))
                if im is not None:
                    self._rep_thumb_by_label[lid] = (im, thumb_src_map.get(str(lid), ""))

        self.trained = True
        self._cache_loaded = True
        return True

    def train_or_load(
        self,
        known_faces_dir: str,
        cache_dir: str | Path | None = None,
        *,
        force_retrain: bool = False,
    ) -> str:
        """Train only when cache is missing, stale, or ``force_retrain=True``.

        Returns ``"trained"`` or ``"loaded_cache"``.
        """
        kf = str(Path(known_faces_dir))
        fp = fingerprint_known_faces(kf)
        if not fp:
            raise FileNotFoundError(f"No training images found under: {kf}")

        if cache_dir is None:
            cache_dir = Path(kf).parent / ".lbph_cache"

        cdir = Path(cache_dir)

        if (
            not force_retrain
            and os.environ.get("FACE_REC_FORCE_RETRAIN", "").lower()
            not in ("1", "true", "yes")
            and self.load_cache(cdir, fp)
        ):
            return "loaded_cache"

        self.train_from_known_faces(kf)
        self.save_cache(cdir, fp)
        return "trained"

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _match_encoding(self, encoding: np.ndarray) -> FaceMatchDetail:
        """Find the closest known face to the given 128-d embedding."""
        if not self.trained or not self._enc_list:
            raise RuntimeError("Model is not trained. Call train_or_load() first.")

        known = np.stack(self._enc_list, axis=0)           # (N, 128)
        distances = self._fr.face_distance(known, encoding) # (N,)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        label_id  = self._lbl_list[best_idx]
        name      = self.label_to_name.get(label_id, "unknown")
        is_match  = best_dist <= self.match_threshold

        rep = self._rep_thumb_by_label.get(label_id)
        thumb, src = (rep[0], rep[1]) if rep else (None, None)

        return FaceMatchDetail(
            name=name,
            distance=best_dist,
            is_match=is_match,
            verdict=self._verdict_most_likely(name, best_dist, is_match),
            similar_thumb_bgr=thumb,
            similar_source=src,
        )

    def _verdict_most_likely(self, name: str, distance: float, is_match: bool) -> str:
        base = f"Most likely: {name} (distance {distance:.3f})"
        if is_match:
            return f"{base} — above confidence threshold"
        return f"{base} — best guess only (raise threshold if too loose)"

    # ------------------------------------------------------------------
    # High-level predict API  (same signatures as before)
    # ------------------------------------------------------------------

    def predict_from_bgr_detail(
        self, frame_bgr: np.ndarray
    ) -> FaceMatchDetail | None:
        """Run recognition on a decoded BGR frame.
        Returns the best-matching identity, or None if no face is detected."""
        if not self.trained:
            raise RuntimeError("Model is not trained.")

        pairs = self._detect_all(frame_bgr)
        if not pairs:
            return None

        if len(pairs) == 1:
            return self._match_encoding(pairs[0][1])

        # Multiple faces: return the one with the lowest distance to any known person
        best: FaceMatchDetail | None = None
        for _, enc in pairs:
            detail = self._match_encoding(enc)
            if best is None or detail.distance < best.distance:
                best = detail
        return best

    def identify_largest_face(
        self, frame_bgr: np.ndarray
    ) -> FaceMatchDetail | None:
        """Single-shot: largest face → embedding → identity (fast path for webcam)."""
        pairs = self._detect_all(frame_bgr)
        if not pairs:
            return None
        # _detect_all already sorts largest-first
        return self._match_encoding(pairs[0][1])

    def reference_display_bgr(self, detail: FaceMatchDetail) -> np.ndarray | None:
        """Return a 256×256 reference thumbnail for the UI."""
        if detail.similar_thumb_bgr is None:
            return None
        return cv2.resize(
            detail.similar_thumb_bgr,
            (self.DISPLAY_MATCH_SIZE, self.DISPLAY_MATCH_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

    # ------------------------------------------------------------------
    # Image / folder helpers
    # ------------------------------------------------------------------

    def predict_from_image_detail(
        self, image_path: str
    ) -> FaceMatchDetail | None:
        """Load an image from disk and run recognition."""
        if not self.trained:
            raise RuntimeError("Model is not trained.")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.predict_from_bgr_detail(img)

    def predict_from_image(
        self, image_path: str
    ) -> tuple[str | None, float | None]:
        """Convenience wrapper; returns ``(name, distance)`` or ``(None, None)``."""
        detail = self.predict_from_image_detail(image_path)
        if detail is None:
            return None, None
        return detail.name, detail.distance

    def predict_from_unknown_folder(
        self, unknown_face_dir: str
    ) -> dict[str, tuple[str | None, float | None]]:
        unknown_path = Path(unknown_face_dir)
        if not unknown_path.exists():
            raise FileNotFoundError(
                f"Unknown-faces folder does not exist: {unknown_face_dir}"
            )
        image_paths = _iter_images(unknown_path)
        if not image_paths:
            raise RuntimeError(f"No images found in: {unknown_face_dir}")
        return {p.name: self.predict_from_image(str(p)) for p in image_paths}

    def evaluate_unknown_folder_by_filename(
        self,
        unknown_face_dir: str,
        *,
        limit: int | None = None,
        show_mismatches: int = 20,
    ) -> dict[str, object]:
        """Evaluate recognition where expected label is derived from the filename."""
        unknown_path = Path(unknown_face_dir)
        if not unknown_path.is_dir():
            raise FileNotFoundError(
                f"Unknown-faces folder does not exist: {unknown_face_dir}"
            )

        paths = _iter_images(unknown_path)
        if limit is not None:
            paths = paths[: max(1, int(limit))]
        if not paths:
            raise RuntimeError(f"No images found in: {unknown_face_dir}")

        total = 0
        correct = 0
        no_face = 0
        mismatches: list[dict[str, object]] = []
        dist_sum = 0.0
        dist_n = 0

        for p in paths:
            total += 1
            expected = expected_label_from_filename(p.name)
            detail = self.predict_from_image_detail(str(p))
            if detail is None:
                no_face += 1
                if len(mismatches) < show_mismatches:
                    mismatches.append(
                        {
                            "file": p.name,
                            "expected": expected,
                            "predicted": None,
                            "distance": None,
                            "reason": "no_face",
                        }
                    )
                continue

            predicted = detail.name
            dist_sum += float(detail.distance)
            dist_n += 1

            if predicted == expected:
                correct += 1
            else:
                if len(mismatches) < show_mismatches:
                    mismatches.append(
                        {
                            "file": p.name,
                            "expected": expected,
                            "predicted": predicted,
                            "distance": float(detail.distance),
                            "is_match": bool(detail.is_match),
                        }
                    )

        evaluated = total - no_face
        acc = (correct / evaluated) if evaluated > 0 else 0.0
        return {
            "unknown_dir": str(unknown_path),
            "total": total,
            "no_face": no_face,
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": acc,
            "mean_distance": (dist_sum / dist_n) if dist_n else None,
            "threshold": self.match_threshold,
            "mismatches": mismatches,
        }

    # ------------------------------------------------------------------
    # Legacy / compat methods (kept so external callers don't break)
    # ------------------------------------------------------------------

    def match_face_gray(
        self, face_gray_200: np.ndarray, *, fast_thumbnail: bool = False
    ) -> FaceMatchDetail:
        """Legacy LBPH-era method. Converts the gray crop back to BGR and re-encodes."""
        bgr = cv2.cvtColor(face_gray_200, cv2.COLOR_GRAY2BGR)
        result = self.predict_from_bgr_detail(bgr)
        if result is not None:
            return result
        return FaceMatchDetail(
            name="unknown",
            distance=999.0,
            is_match=False,
            verdict="No face found in gray→BGR conversion",
            similar_thumb_bgr=None,
            similar_source=None,
        )

    def largest_face_gray_200(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Legacy: detect the largest face and return a 200×200 gray crop."""
        pairs = self._detect_all(frame_bgr)
        if not pairs:
            return None
        top, right, bottom, left = pairs[0][0]
        h, w = frame_bgr.shape[:2]
        top = max(0, top); left = max(0, left)
        bottom = min(h, bottom); right = min(w, right)
        crop = frame_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (200, 200))

    def best_face_gray_200(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Legacy: returns the largest face gray crop (best-match logic is now in encoding space)."""
        return self.largest_face_gray_200(frame_bgr)

    # ------------------------------------------------------------------
    # Webcam interactive
    # ------------------------------------------------------------------

    def capture_and_identify_webcam(
        self,
        camera_index: int = 0,
        window_preview: str = "Preview — SPACE capture, ESC quit",
        window_result: str = "Result — any key to close",
    ) -> FaceMatchDetail | None:
        """Open camera preview; press SPACE to capture once, show the result."""
        if not self.trained:
            raise RuntimeError("Model is not trained.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        cv2.namedWindow(window_preview, cv2.WINDOW_NORMAL)
        print("SPACE = capture & identify, ESC = quit without capture.")

        detail: FaceMatchDetail | None = None
        last_frame: np.ndarray | None = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
            cv2.putText(
                frame,
                "SPACE: capture   ESC: quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_preview, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" ") and last_frame is not None:
                detail = self.identify_largest_face(last_frame)
                break

        cap.release()
        cv2.destroyWindow(window_preview)

        if detail is None:
            return None

        board = np.full((420, 720, 3), 40, dtype=np.uint8)
        ref = self.reference_display_bgr(detail)
        if ref is not None:
            board[40 : 40 + ref.shape[0], 40 : 40 + ref.shape[1]] = ref

        color = (0, 220, 0) if detail.is_match else (0, 180, 255)
        cv2.putText(
            board, detail.verdict[:70], (320, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            board, detail.name, (320, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA,
        )
        if detail.similar_source:
            cv2.putText(
                board, detail.similar_source[:50], (320, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
            )

        cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
        cv2.imshow(window_result, board)
        cv2.waitKey(0)
        cv2.destroyWindow(window_result)

        return detail


# ---------------------------------------------------------------------------
# Module-level helpers used by app.py
# ---------------------------------------------------------------------------

def resolve_unknown_dir(base_data_dir: str) -> str:
    """Return the first of the standard unknown-face folder names that exists,
    or fall back to ``data/unknown_faces``."""
    base_path = Path(base_data_dir)
    for name in ("unknown_faces", "unknown_face", "unknown", "test"):
        p = base_path / name
        if p.exists():
            return str(p)
    return str(base_path / "unknown_faces")


def default_cache_dir(data_dir: str) -> Path:
    return Path(data_dir) / ".lbph_cache"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n--- Face Recognition (deep embeddings) ---")
    base_data_dir = input("Path to 'data' folder [data]: ").strip() or "data"
    known_faces_dir = str(Path(base_data_dir) / "known_faces")
    unknown_face_dir = resolve_unknown_dir(base_data_dir)
    cache_dir = default_cache_dir(base_data_dir)

    rec = SimpleFaceRecognizer()
    mode = rec.train_or_load(known_faces_dir, cache_dir=cache_dir)
    print(f"Model: {mode}  (cache: {cache_dir})")

    while True:
        print("\n1. Batch unknown folder  2. Image file  3. Webcam snapshot (SPACE)")
        print("4. Evaluate unknowns (filename=truth)  5. Force retrain & save cache  6. Exit")
        choice = input("Choice [1-6]: ").strip()

        if choice == "1":
            print(f"Scanning {unknown_face_dir} …")
            results = rec.predict_from_unknown_folder(unknown_face_dir)
            for filename, (name, conf) in results.items():
                if name is None:
                    print(f"  {filename}: no face")
                else:
                    print(f"  {filename}: {name} (d={conf:.3f})")
        elif choice == "2":
            img_path = input("Image path: ").strip()
            d = rec.predict_from_image_detail(img_path)
            if d is None:
                print("No face detected.")
            else:
                print(d.verdict)
        elif choice == "3":
            rec.capture_and_identify_webcam()
        elif choice == "4":
            print(f"Evaluating {unknown_face_dir} …")
            rep = rec.evaluate_unknown_folder_by_filename(unknown_face_dir)
            print(
                f"Accuracy: {rep['accuracy']:.3f}  "
                f"(correct={rep['correct']}/{rep['evaluated']}, no_face={rep['no_face']})"
            )
            if rep.get("mean_distance") is not None:
                print(
                    f"Mean distance: {rep['mean_distance']:.3f}  "
                    f"(threshold={rep['threshold']:.3f})"
                )
            for m in rep["mismatches"]:
                if m.get("reason") == "no_face":
                    print(f"  [no face] {m['file']}  expected={m['expected']}")
                else:
                    print(
                        f"  [wrong] {m['file']}  expected={m['expected']}  "
                        f"predicted={m['predicted']}  d={m['distance']:.3f}"
                    )
        elif choice == "5":
            m = rec.train_or_load(known_faces_dir, cache_dir=cache_dir, force_retrain=True)
            print(f"Retrained ({m}).")
        else:
            print("Bye.")
            break


if __name__ == "__main__":
    main()
