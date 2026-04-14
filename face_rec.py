from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale image — improves detection and recognition
    in low-contrast, dark, or hazy scenes."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


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
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceMatchDetail:
    """LBPH prediction result for a single face crop."""

    name: str
    distance: float
    is_match: bool
    verdict: str
    similar_thumb_bgr: np.ndarray | None
    similar_source: str | None


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class SimpleFaceRecognizer:
    """
    OpenCV Haar + LBPH face recognizer.

    Training images must be organized as:
        data/known_faces/<Person Name>/*.jpg|jpeg|png|webp

    A model cache (default ``data/.lbph_cache``) is written after training
    so subsequent runs skip retraining when the data hasn't changed.
    """

    DEFAULT_MATCH_THRESHOLD: float = 72.0
    THUMB_SIZE: int = 96
    DISPLAY_MATCH_SIZE: int = 256
    # 0.55 gives a good balance between speed and small-face detection.
    DEFAULT_DETECT_DOWNSCALE: float = 0.55

    def __init__(
        self,
        cascade_path: str | None = None,
        match_threshold: float | None = None,
        detect_downscale: float | None = None,
    ) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face module not found. "
                "Install 'opencv-contrib-python' instead of 'opencv-python'."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.match_threshold = (
            float(match_threshold)
            if match_threshold is not None
            else self.DEFAULT_MATCH_THRESHOLD
        )
        self.detect_downscale = (
            float(detect_downscale)
            if detect_downscale is not None
            else self.DEFAULT_DETECT_DOWNSCALE
        )

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.label_to_name: dict[int, str] = {}
        self.trained: bool = False
        self._training_samples: list[tuple[int, np.ndarray, np.ndarray, str]] = []
        self._rep_thumb_by_label: dict[int, tuple[np.ndarray, str]] = {}
        self._cache_loaded: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_thumb_bgr(
        self, img_bgr: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        crop = img_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return np.zeros((self.THUMB_SIZE, self.THUMB_SIZE, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.THUMB_SIZE, self.THUMB_SIZE))

    def _detect_faces_xywh(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Detect faces and return (x, y, w, h) boxes in original image coordinates."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = _clahe_gray(gray)

        sc = self.detect_downscale
        if sc < 1.0:
            small = cv2.resize(gray, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        min_px = max(24, int(48 * sc))
        rects = self.face_cascade.detectMultiScale(
            small,
            scaleFactor=1.1,   # finer pyramid — catches more face sizes than 1.2
            minNeighbors=4,    # balanced: few false positives, few missed faces
            minSize=(min_px, min_px),
        )

        if sc < 1.0:
            inv = 1.0 / sc
            return [
                (
                    int(round(x * inv)),
                    int(round(y * inv)),
                    int(round(w * inv)),
                    int(round(h * inv)),
                )
                for (x, y, w, h) in rects
            ]
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _collect_images_from_folder(
        self,
        labeled_faces_root_dir: str,
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Walk ``known_faces/<Person>/`` subdirectories, detect one face per image,
        and build LBPH training arrays.

        Label IDs are assigned only to persons that have at least one detectable
        face, so IDs are always contiguous and match the cache exactly.
        """
        images: list[np.ndarray] = []
        labels: list[int] = []

        dataset_path = Path(labeled_faces_root_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"known_faces folder does not exist: {labeled_faces_root_dir}"
            )

        label_id = 0
        self.label_to_name.clear()
        self._training_samples.clear()
        self._rep_thumb_by_label.clear()

        for person_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
            person_name = person_dir.name
            found_any = False

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [skip] Cannot read {img_path.name}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = _clahe_gray(gray)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                )
                if len(faces) == 0:
                    print(f"  [skip] No face detected in {img_path.name}")
                    continue

                if not found_any:
                    # Only assign a label ID once we know this person has data
                    self.label_to_name[label_id] = person_name
                    found_any = True

                (x, y, w, h) = faces[0]
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, (200, 200))
                thumb = self._make_thumb_bgr(img, x, y, w, h)

                images.append(face_resized)
                labels.append(label_id)
                self._training_samples.append(
                    (label_id, face_resized, thumb, img_path.name)
                )
                if label_id not in self._rep_thumb_by_label:
                    self._rep_thumb_by_label[label_id] = (thumb, img_path.name)

            if found_any:
                label_id += 1
            else:
                print(f"  [warn] No usable images for '{person_name}' — skipped.")

        if not images:
            raise RuntimeError(
                "No training faces found. "
                "Check that known_faces/<Name>/ contains images with detectable frontal faces."
            )

        return images, labels

    def train_from_known_faces(self, known_faces_dir: str) -> None:
        images, labels = self._collect_images_from_folder(known_faces_dir)
        self.recognizer.train(images, np.array(labels))
        self.trained = True
        self._cache_loaded = False

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def save_cache(self, cache_dir: str | Path, fingerprint: str) -> None:
        """Write LBPH model XML, meta.json, and per-label reference thumbnails."""
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)

        self.recognizer.write(str(d / "lbph_model.xml"))

        thumb_sources: dict[str, str] = {}
        for lid, (thumb, src) in self._rep_thumb_by_label.items():
            thumb_sources[str(lid)] = src
            cv2.imwrite(str(d / f"thumb_{lid}.png"), thumb)

        meta = {
            "fingerprint": fingerprint,
            "label_to_name": {str(k): v for k, v in sorted(self.label_to_name.items())},
            "thumb_source": thumb_sources,
            "match_threshold": self.match_threshold,
            "detect_downscale": self.detect_downscale,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_cache(self, cache_dir: str | Path, expected_fingerprint: str) -> bool:
        """Load cached model if the fingerprint matches.  Returns True on success."""
        d = Path(cache_dir)
        meta_path = d / "meta.json"
        model_path = d / "lbph_model.xml"
        if not meta_path.is_file() or not model_path.is_file():
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        if meta.get("fingerprint") != expected_fingerprint:
            return False

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(model_path))

        self.label_to_name = {int(k): v for k, v in meta["label_to_name"].items()}
        self.match_threshold = float(
            meta.get("match_threshold", self.DEFAULT_MATCH_THRESHOLD)
        )
        self.detect_downscale = float(
            meta.get("detect_downscale", self.DEFAULT_DETECT_DOWNSCALE)
        )

        self._training_samples.clear()
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
    # Prediction helpers
    # ------------------------------------------------------------------

    def _best_similar_training_thumb(
        self, face_gray_200: np.ndarray, predicted_label: int
    ) -> tuple[np.ndarray | None, str | None]:
        """Return the training thumbnail whose pixel values are closest (MSE)
        to the query face.  Falls back to the representative thumbnail when
        training samples are not in memory (cache-loaded mode)."""
        if not self._training_samples:
            rep = self._rep_thumb_by_label.get(predicted_label)
            return (rep[0], rep[1]) if rep else (None, None)

        candidates = [
            (g, t, src)
            for lid, g, t, src in self._training_samples
            if lid == predicted_label
        ]
        if not candidates:
            rep = self._rep_thumb_by_label.get(predicted_label)
            return (rep[0], rep[1]) if rep else (None, None)

        best_mse = float("inf")
        best_thumb: np.ndarray | None = None
        best_src: str | None = None
        q = face_gray_200.astype(np.float32)
        for gray, thumb, src in candidates:
            mse = float(np.mean((gray.astype(np.float32) - q) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_thumb = thumb
                best_src = src
        return best_thumb, best_src

    def _verdict_most_likely(self, name: str, distance: float, is_match: bool) -> str:
        base = f"Most likely: {name} (distance {distance:.1f})"
        if is_match:
            return f"{base} — above confidence threshold"
        return f"{base} — best guess only (raise threshold if too loose)"

    # ------------------------------------------------------------------
    # Core match API
    # ------------------------------------------------------------------

    def match_face_gray(
        self,
        face_gray_200: np.ndarray,
        *,
        fast_thumbnail: bool = False,
    ) -> FaceMatchDetail:
        """Run LBPH prediction on a 200×200 grayscale face crop."""
        if not self.trained:
            raise RuntimeError("Model is not trained. Call train_or_load() first.")

        label_id, distance = self.recognizer.predict(face_gray_200)
        name = self.label_to_name.get(label_id, "unknown")
        is_match = float(distance) <= self.match_threshold

        if fast_thumbnail or self._cache_loaded:
            rep = self._rep_thumb_by_label.get(label_id)
            thumb, src = (rep[0], rep[1]) if rep else (None, None)
        else:
            thumb, src = self._best_similar_training_thumb(face_gray_200, label_id)

        return FaceMatchDetail(
            name=name,
            distance=float(distance),
            is_match=is_match,
            verdict=self._verdict_most_likely(name, float(distance), is_match),
            similar_thumb_bgr=thumb,
            similar_source=src,
        )

    def largest_face_gray_200(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Detect the largest face in a BGR frame; return a 200×200 CLAHE grayscale
        crop ready for LBPH prediction.

        CLAHE is applied here to match the training pipeline in
        ``_collect_images_from_folder``, which also extracts ROIs from the
        CLAHE-enhanced gray image.
        """
        faces = self._detect_faces_xywh(frame_bgr)
        if not faces:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = _clahe_gray(gray)  # must match training pipeline
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = gray[y : y + h, x : x + w]
        return cv2.resize(face_roi, (200, 200))

    def identify_largest_face(
        self, frame_bgr: np.ndarray
    ) -> FaceMatchDetail | None:
        """Single-shot: largest face → LBPH → most likely person (fast path)."""
        crop = self.largest_face_gray_200(frame_bgr)
        if crop is None:
            return None
        return self.match_face_gray(crop, fast_thumbnail=True)

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
    # Predict from image / BGR frame
    # ------------------------------------------------------------------

    def predict_from_bgr_detail(
        self, frame_bgr: np.ndarray
    ) -> FaceMatchDetail | None:
        """Run recognition on a decoded BGR frame.
        Avoids temp-file I/O when the caller already holds the image in memory."""
        if not self.trained:
            raise RuntimeError("Model is not trained.")
        crop = self.largest_face_gray_200(frame_bgr)
        if crop is None:
            return None
        return self.match_face_gray(crop, fast_thumbnail=bool(self._cache_loaded))

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

    # ------------------------------------------------------------------
    # Batch / webcam
    # ------------------------------------------------------------------

    def predict_from_unknown_folder(
        self, unknown_face_dir: str
    ) -> dict[str, tuple[str | None, float | None]]:
        unknown_path = Path(unknown_face_dir)
        if not unknown_path.exists():
            raise FileNotFoundError(
                f"Unknown-faces folder does not exist: {unknown_face_dir}"
            )

        image_paths = [
            p
            for p in sorted(unknown_path.iterdir())
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        if not image_paths:
            raise RuntimeError(f"No images found in: {unknown_face_dir}")

        return {p.name: self.predict_from_image(str(p)) for p in image_paths}

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

        # Result board
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
    print("\n--- Face Recognition ---")
    base_data_dir = input("Path to 'data' folder [data]: ").strip() or "data"
    known_faces_dir = str(Path(base_data_dir) / "known_faces")
    unknown_face_dir = resolve_unknown_dir(base_data_dir)
    cache_dir = default_cache_dir(base_data_dir)

    rec = SimpleFaceRecognizer()
    mode = rec.train_or_load(known_faces_dir, cache_dir=cache_dir)
    print(f"Model: {mode}  (cache: {cache_dir})")

    while True:
        print("\n1. Batch unknown folder  2. Image file  3. Webcam snapshot (SPACE)")
        print("4. Force retrain & save cache  5. Exit")
        choice = input("Choice [1-5]: ").strip()

        if choice == "1":
            print(f"Scanning {unknown_face_dir} …")
            results = rec.predict_from_unknown_folder(unknown_face_dir)
            for filename, (name, conf) in results.items():
                if name is None:
                    print(f"  {filename}: no face")
                else:
                    print(f"  {filename}: {name} (d={conf:.1f})")
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
            m = rec.train_or_load(known_faces_dir, cache_dir=cache_dir, force_retrain=True)
            print(f"Retrained ({m}).")
        else:
            print("Bye.")
            break


if __name__ == "__main__":
    main()
