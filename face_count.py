from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import face_recognition
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union for two (top, right, bottom, left) boxes."""
    top    = max(a[0], b[0])
    right  = min(a[1], b[1])
    bottom = min(a[2], b[2])
    left   = max(a[3], b[3])
    inter_h = max(0, bottom - top)
    inter_w = max(0, right - left)
    inter = inter_h * inter_w
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[1] - a[3])
    area_b = (b[2] - b[0]) * (b[1] - b[3])
    return inter / float(area_a + area_b - inter)


def _nms(
    locations: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """Non-Maximum Suppression: keep the largest box from each overlapping cluster.

    Merges duplicate detections that arise when running multiple upsample passes
    on the same image.
    """
    if not locations:
        return []
    # Sort by area descending so the largest (most reliable) box wins
    by_area = sorted(
        locations,
        key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
        reverse=True,
    )
    kept: list[tuple[int, int, int, int]] = []
    for box in by_area:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _clahe_enhance(rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel (LAB) to boost local contrast.
    Helps detect faces in low-light, hazy, or low-contrast images."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _sharpen(rgb: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask — recovers detail in soft or distant faces."""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)


def _iou_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU for (x, y, w, h) format rectangles."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax1 + aw, bx1 + bw);  iy2 = min(ay1 + ah, by1 + bh)
    iw = max(0, ix2 - ix1);  ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def _nms_xywh(
    rects: np.ndarray,
    weights: np.ndarray,
    overlap_thresh: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression for HOG-style (x, y, w, h) detections.

    Keeps the highest-weight box from each overlapping cluster.
    """
    if len(rects) == 0:
        return rects, weights
    weights = np.asarray(weights, dtype=float).ravel()
    x1 = rects[:, 0].astype(float);  y1 = rects[:, 1].astype(float)
    x2 = (rects[:, 0] + rects[:, 2]).astype(float)
    y2 = (rects[:, 1] + rects[:, 3]).astype(float)
    areas = (x2 - x1) * (y2 - y1)
    order = weights.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0]);  keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < overlap_thresh]
    idx = np.array(keep, dtype=int)
    return rects[idx], weights[idx]


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _natural_sort_key(name: str) -> list:
    """Split ``name`` into alternating text and integer tokens for sane ordering
    (``img2`` before ``img10``; ``seq_000010`` still sorts with zero-padded names)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _natural_sort_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: _natural_sort_key(p.name))


def list_image_files(folder: Path, *, recursive: bool = False) -> list[Path]:
    """Return image paths under ``folder``, ordered with :func:`_natural_sort_paths`."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    if recursive:
        raw = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
    else:
        raw = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
    return _natural_sort_paths(raw)


def load_count_labels_csv(csv_path: str | Path) -> dict[str, int]:
    """Load ``filename -> count`` from a CSV.

    Accepts headers such as ``file`` / ``filename`` / ``frame`` for the key column
    and ``count`` / ``truth`` / ``label`` for the integer; extra columns are ignored.
    Opens with UTF-8-BOM support for Excel exports.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(fh, dialect)
        rows = list(reader)

    if not rows:
        return {}

    header = [c.strip().lower() for c in rows[0]]
    key_aliases = {"file", "filename", "frame", "name", "image"}
    val_aliases = {"count", "truth", "label", "gt", "faces", "people"}

    def find_col(aliases: set[str], cells: list[str]) -> int | None:
        for i, h in enumerate(cells):
            if h in aliases:
                return i
        return None

    hk = find_col(key_aliases, header)
    vk = find_col(val_aliases, header)

    out: dict[str, int] = {}
    if hk is not None and vk is not None:
        for row in rows[1:]:
            if len(row) <= max(hk, vk):
                continue
            key = row[hk].strip()
            if not key:
                continue
            try:
                out[key] = int(float(row[vk].strip()))
            except ValueError:
                continue
    else:
        for row in rows:
            if len(row) < 2:
                continue
            key = row[0].strip()
            if not key or key.lower() in key_aliases:
                continue
            try:
                out[key] = int(float(row[1]))
            except ValueError:
                continue

    if not out:
        raise ValueError(
            f"No valid label rows in {path} (expected header with file/count "
            "or two columns: filename, integer count)."
        )
    return out


def _mae_rmse_exact(
    pairs: list[tuple[int, int]],
) -> tuple[float, float, float, int]:
    """``pairs`` are ``(predicted, truth)``. Returns MAE, RMSE, exact-match rate, n."""
    if not pairs:
        return 0.0, 0.0, 0.0, 0
    n = len(pairs)
    abs_err = sum(abs(a - b) for a, b in pairs)
    mae = abs_err / n
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / n)
    exact = sum(1 for a, b in pairs if a == b) / n
    return mae, rmse, exact, n


@dataclass
class CountEvalReport:
    """Metrics from evaluating a counter against ground-truth CSV labels."""

    train_mae: float
    train_rmse: float
    train_exact_rate: float
    train_n: int
    test_mae: float
    test_rmse: float
    test_exact_rate: float
    test_n: int
    labels_in_csv: int
    matched_disk: int
    missing_files: int
    orphan_files: int
    load_failures: int
    train_ratio: float
    seed: int
    best_config: str | None = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FaceCounterSystem:
    """
    Face counting with ``face_recognition`` (dlib HOG by default; optional CNN).

    Key features:
    - CLAHE preprocessing for distant / low-contrast faces.
    - Optional unsharp-mask sharpening for blurry images.
    - Dual-pass upsample merged with NMS to catch small faces without duplicates.
    - Minimum face-size filter to drop tiny false positives.
    - Webcam mode with temporal smoothing to reduce count flicker.
    """

    DEFAULT_MAX_SIDE_SENSITIVE = 4096
    DEFAULT_MAX_SIDE_FAST = 2400   # keeps detail without killing speed
    HARD_MAX_SIDE = 8000

    DEFAULT_UPSAMPLE_SENSITIVE = 2
    DEFAULT_UPSAMPLE_FAST = 1

    def __init__(
        self,
        model: str = "hog",
        *,
        sensitive_counting: bool = False,
        enhance: bool = True,       # CLAHE contrast enhancement
        sharpen: bool = False,      # unsharp-mask (enable for blurry images)
        min_face_size: int = 20,    # minimum face height/width in pixels
        nms_iou: float = 0.35,      # IoU threshold for duplicate removal
    ) -> None:
        if model not in ("hog", "cnn"):
            raise ValueError("model must be 'hog' or 'cnn'")
        self.model = model
        self.sensitive_counting = sensitive_counting
        self.enhance = enhance
        self.sharpen = sharpen
        self.min_face_size = min_face_size
        self.nms_iou = nms_iou

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        if self.enhance:
            rgb = _clahe_enhance(rgb)
        if self.sharpen:
            rgb = _sharpen(rgb)
        return rgb

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        return face_recognition.load_image_file(path)

    @staticmethod
    def _load_rgb_safe(path: str) -> np.ndarray | None:
        """Load RGB array or ``None`` on missing/corrupt/empty images (PIL errors)."""
        try:
            rgb = face_recognition.load_image_file(path)
        except (OSError, ValueError):
            return None
        if rgb is None or rgb.size == 0:
            return None
        return rgb

    def count_faces_only(
        self,
        image_path: str,
        *,
        upsample: int | None = None,
        max_side: int | None = None,
    ) -> int | None:
        """Return face count, or ``None`` if the image could not be decoded."""
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side
        rgb = self._load_rgb_safe(image_path)
        if rgb is None:
            return None
        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        return len(locs)

    def _raw_locations(
        self,
        rgb: np.ndarray,
        upsample: int,
    ) -> list[tuple[int, int, int, int]]:
        return list(
            face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=upsample,
                model=self.model,
            )
        )

    def _locations(
        self,
        rgb: np.ndarray,
        *,
        upsample: int,
    ) -> list[tuple[int, int, int, int]]:
        """Detect at the requested upsample level and one level higher (when
        upsample < 2) to catch small/distant faces.  Results are merged with
        NMS and filtered by ``min_face_size``."""
        locs = self._raw_locations(rgb, upsample)

        # Extra pass catches faces that only appear at higher resolution
        if upsample < 2:
            locs = locs + self._raw_locations(rgb, upsample + 1)

        locs = _nms(locs, self.nms_iou)

        if self.min_face_size > 0:
            locs = [
                (top, right, bottom, left)
                for (top, right, bottom, left) in locs
                if (bottom - top) >= self.min_face_size
                and (right - left) >= self.min_face_size
            ]
        return locs

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate_bgr(
        self,
        rgb: np.ndarray,
        face_locations: list[tuple[int, int, int, int]],
        *,
        draw_block: bool,
        block_alpha: float,
    ) -> tuple[int, np.ndarray]:
        count = len(face_locations)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()

        for idx, (top, right, bottom, left) in enumerate(face_locations, start=1):
            if draw_block:
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 180, 0), 2)
            else:
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            label_y = max(top - 6, 14)
            cv2.putText(
                bgr, str(idx), (left + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        if draw_block and count > 0:
            alpha = max(0.0, min(1.0, float(block_alpha)))
            bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

        # HUD banner
        label = f"People detected: {count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(bgr, (10, 10), (20 + tw, 18 + th + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(bgr, label, (16, 18 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return count, bgr

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _defaults_for_count(self) -> tuple[int, int]:
        if self.sensitive_counting:
            return self.DEFAULT_UPSAMPLE_SENSITIVE, self.DEFAULT_MAX_SIDE_SENSITIVE
        return self.DEFAULT_UPSAMPLE_FAST, self.DEFAULT_MAX_SIDE_FAST

    @staticmethod
    def _maybe_downscale_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
        """Downscale ``rgb`` so its longest edge does not exceed ``max_side``.
        Hard cap of ``HARD_MAX_SIDE`` (8000 px) is always enforced."""
        h, w = rgb.shape[:2]
        limit = max_side if max_side > 0 else FaceCounterSystem.HARD_MAX_SIDE
        limit = min(limit, FaceCounterSystem.HARD_MAX_SIDE)
        m = max(h, w)
        if m <= limit:
            return rgb
        scale = limit / m
        return cv2.resize(
            rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_faces_in_image(
        self,
        image_path: str,
        output_dir: str = "output",
        upsample: int | None = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: int | None = None,
    ) -> int:
        """Detect faces in ``image_path``, save an annotated copy to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)
        d_up, d_side = self._defaults_for_count()
        up   = d_up   if upsample  is None else upsample
        side = d_side if max_side  is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        count, bgr = self._annotate_bgr(
            rgb, locs, draw_block=draw_block, block_alpha=block_alpha
        )
        output_path = os.path.join(output_dir, f"count_{Path(image_path).name}")
        cv2.imwrite(output_path, bgr)
        print(f"{Path(image_path).name}: {count} face(s). Saved: {output_path}")
        return count

    def count_faces_in_image_annotated(
        self,
        image_path: str,
        upsample: int | None = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: int | None = None,
    ) -> tuple[int, np.ndarray]:
        """Like ``count_faces_in_image`` but returns ``(count, annotated_bgr)``
        instead of saving to disk."""
        d_up, d_side = self._defaults_for_count()
        up   = d_up   if upsample  is None else upsample
        side = d_side if max_side  is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_rgb(
        self,
        rgb: np.ndarray,
        *,
        upsample: int | None = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: int | None = None,
    ) -> tuple[int, np.ndarray]:
        """Detect faces in an RGB array; returns ``(count, annotated_bgr)``."""
        d_up, d_side = self._defaults_for_count()
        up   = d_up   if upsample  is None else upsample
        side = d_side if max_side  is None else max_side

        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_bytes(
        self,
        image_bytes: bytes,
        *,
        upsample: int | None = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: int | None = None,
    ) -> tuple[int, np.ndarray]:
        """Decode raw image bytes, detect faces, return ``(count, annotated_bgr)``."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.count_faces_in_rgb(
            rgb,
            upsample=upsample,
            draw_block=draw_block,
            block_alpha=block_alpha,
            max_side=max_side,
        )

    def count_faces_in_folder(
        self,
        folder_path: str,
        output_dir: str = "output",
        upsample: int | None = None,
        draw_block: bool = True,
        max_side: int | None = None,
    ) -> dict[str, int]:
        """Process every supported image in ``folder_path``; return a
        ``{filename: count}`` mapping."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        results: dict[str, int] = {}
        for p in list_image_files(folder):
            results[p.name] = self.count_faces_in_image(
                str(p),
                output_dir=output_dir,
                upsample=upsample,
                draw_block=draw_block,
                max_side=max_side,
            )
        return results

    def count_from_webcam(
        self,
        camera_index: int = 0,
        upsample: int | None = None,
        window_name: str = "Face Counter",
        smooth_frames: int = 3,
    ) -> None:
        """Live webcam face counting with temporal smoothing.

        Detection runs on a 0.75× downscale for speed while retaining enough
        resolution for distant faces.  The displayed count is the rolling
        average over ``smooth_frames`` frames to reduce flicker.
        """
        d_up, _ = self._defaults_for_count()
        up = d_up if upsample is None else upsample

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\nWebcam: q or Esc to quit.")

        DETECT_SCALE = 0.75
        recent_counts: list[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small = self._preprocess(rgb_small)
            raw_locs = self._locations(rgb_small, upsample=up)

            # Scale boxes back to full-frame coordinates
            inv = 1.0 / DETECT_SCALE
            face_locations = [
                (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
                for (t, r, b, l) in raw_locs
            ]

            # Rolling average for a stable display count
            recent_counts.append(len(face_locations))
            if len(recent_counts) > smooth_frames:
                recent_counts.pop(0)
            display_count = round(sum(recent_counts) / len(recent_counts))

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            label = f"People: {display_count}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 8), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (16, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyWindow(window_name)


# ---------------------------------------------------------------------------
# PersonCounterSystem — HOG pedestrian + MOG2 background subtraction
# ---------------------------------------------------------------------------

class PersonCounterSystem:
    """
    Person/pedestrian counter for surveillance or overhead-angle footage.

    Designed for scenes where face detectors fail — top-down CCTV, angled
    cameras, small/distant figures.  Supports three detection strategies:

    ``"hog"``
        OpenCV HOG pedestrian detector.  Works on any single frame
        independently; no warmup required.

    ``"mog2"``
        MOG2 background subtraction + blob analysis.  Requires a sequence
        of frames so the background model can warm up.  Best for a static
        camera with moving people.

    ``"combined"``
        HOG detections refined by the MOG2 foreground mask.  HOG boxes
        whose centre lands inside a foreground region are kept; large
        foreground blobs not covered by any HOG box are added as extra
        detections.  Most accurate for sequential surveillance footage.
    """

    # HOG sliding-window parameters
    _HOG_WIN_STRIDE: tuple[int, int] = (8, 8)
    _HOG_PADDING:    tuple[int, int] = (4, 4)

    # Morphological kernel for MOG2 mask cleaning
    _MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __init__(
        self,
        *,
        hog_scale: float = 1.05,
        hog_weight_threshold: float = 0.4,
        hog_nms_overlap: float = 0.65,
        mog2_var_threshold: float = 40.0,
        mog2_warmup: int = 30,
        blob_min_area: int = 600,
        blob_max_area: int = 25_000,
        smooth_frames: int = 5,
    ) -> None:
        # HOG detector (shared; stateless)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hog_scale            = hog_scale
        self._hog_weight_threshold = hog_weight_threshold
        self._hog_nms_overlap      = hog_nms_overlap

        # MOG2 parameters (model created fresh per sequence)
        self._mog2_var_threshold = mog2_var_threshold
        self._mog2_warmup        = mog2_warmup

        # Blob filtering
        self._blob_min_area = blob_min_area
        self._blob_max_area = blob_max_area

        self._smooth_frames = smooth_frames

    # ------------------------------------------------------------------
    # Detection primitives
    # ------------------------------------------------------------------

    def _detect_hog(
        self, bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """HOG pedestrian detector + NMS.  Returns (x, y, w, h) boxes."""
        rects, weights = self._hog.detectMultiScale(
            bgr,
            winStride=self._HOG_WIN_STRIDE,
            padding=self._HOG_PADDING,
            scale=self._hog_scale,
        )
        if len(rects) == 0:
            return []
        weights = np.asarray(weights, dtype=float).ravel()
        mask = weights >= self._hog_weight_threshold
        rects, weights = rects[mask], weights[mask]
        if len(rects) == 0:
            return []
        rects, _ = _nms_xywh(rects, weights, self._hog_nms_overlap)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in rects]

    def _detect_blobs(
        self, fg_mask: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Extract person-sized blobs from a MOG2 foreground mask.
        Returns (x, y, w, h) bounding rects filtered by area."""
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._MORPH_KERNEL)
        cleaned = cv2.dilate(cleaned, self._MORPH_KERNEL, iterations=2)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if self._blob_min_area <= area <= self._blob_max_area:
                boxes.append(tuple(int(v) for v in cv2.boundingRect(c)))  # type: ignore[arg-type]
        return boxes

    def _detect_combined(
        self,
        bgr: np.ndarray,
        fg_mask: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """HOG detections filtered by foreground, plus uncovered blob boxes.

        Steps:
        1. Keep HOG boxes whose centre patch has >15 % foreground pixels.
        2. Add blob bounding rects that don't overlap (IoU > 0.2) with any
           kept HOG box — these capture people that HOG missed.
        """
        hog_boxes  = self._detect_hog(bgr)
        blob_boxes = self._detect_blobs(fg_mask)
        h_img, w_img = bgr.shape[:2]

        kept: list[tuple[int, int, int, int]] = []
        for box in hog_boxes:
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            r = max(4, min(w, h) // 4)
            py1 = max(0, cy - r);  py2 = min(h_img, cy + r)
            px1 = max(0, cx - r);  px2 = min(w_img, cx + r)
            patch = fg_mask[py1:py2, px1:px2]
            # Keep if foreground occupies at least 15 % of the centre patch
            if patch.size > 0 and float(patch.mean()) / 255.0 >= 0.15:
                kept.append(box)

        for blob in blob_boxes:
            if not any(_iou_xywh(blob, k) > 0.2 for k in kept):
                kept.append(blob)

        return kept

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate_bgr(
        self,
        bgr: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        method: str,
    ) -> tuple[int, np.ndarray]:
        """Draw boxes and a HUD banner; return (count, annotated_copy)."""
        out = bgr.copy()
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_y = max(y - 6, 14)
            cv2.putText(
                out, str(idx), (x + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
        count = len(boxes)
        hud = f"People [{method}]: {count}"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (10, 10), (20 + tw, 18 + th + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(out, hud, (16, 18 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return count, out

    # ------------------------------------------------------------------
    # Public API — single image / bytes
    # ------------------------------------------------------------------

    def count_people_in_image(
        self,
        image_path: str,
        output_dir: str = "output",
    ) -> int:
        """Detect people in a single image using HOG; save annotated copy."""
        os.makedirs(output_dir, exist_ok=True)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        boxes = self._detect_hog(bgr)
        count, annotated = self._annotate_bgr(bgr, boxes, "hog")
        out_path = os.path.join(output_dir, f"people_{Path(image_path).name}")
        cv2.imwrite(out_path, annotated)
        print(f"{Path(image_path).name}: {count} person(s). Saved: {out_path}")
        return count

    def count_people_in_bytes(
        self,
        image_bytes: bytes,
    ) -> tuple[int, np.ndarray]:
        """Decode raw image bytes, detect people with HOG, return (count, annotated_bgr)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
        boxes = self._detect_hog(bgr)
        return self._annotate_bgr(bgr, boxes, "hog")

    def count_people_hog_only(self, image_path: str) -> int | None:
        """HOG person count, or ``None`` if the image could not be read."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        return len(self._detect_hog(bgr))

    def _person_boxes_for_sequence_frame(
        self,
        bgr: np.ndarray,
        idx: int,
        method: str,
        mog2: cv2.BackgroundSubtractorMOG2,
    ) -> tuple[list[tuple[int, int, int, int]], str]:
        """Shared counting logic for sequential footage (HOG / MOG2 / combined)."""
        fg_mask: np.ndarray | None = None
        if method in ("mog2", "combined"):
            fg_mask = mog2.apply(bgr)

        in_warmup = (method in ("mog2", "combined")) and (idx < self._mog2_warmup)

        if method == "hog" or in_warmup:
            return self._detect_hog(bgr), "hog"
        if method == "mog2":
            return self._detect_blobs(fg_mask), "mog2"
        return self._detect_combined(bgr, fg_mask), "combined"

    def sequence_counts(
        self,
        folder_path: str,
        method: str = "hog",
        *,
        progress_every: int = 100,
    ) -> dict[str, int]:
        """Run the same detection path as :meth:`process_frame_sequence` but only
        return ``{filename -> count}`` (no files written). Skipped unreadable
        frames are omitted from the mapping.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_paths = list_image_files(folder)
        if not image_paths:
            raise RuntimeError(f"No images found in: {folder_path}")

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=self._mog2_var_threshold,
            detectShadows=False,
        )

        results: dict[str, int] = {}
        total = len(image_paths)
        seq_start = time.perf_counter()

        print(f"\n[PersonCounter.sequence_counts] {total} frames  method={method}")
        if method in ("mog2", "combined"):
            print(f"  MOG2 warmup: first {self._mog2_warmup} frames (HOG during warmup)")

        for idx, p in enumerate(image_paths):
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"  [skip] Cannot read {p.name}")
                continue

            boxes, _ = self._person_boxes_for_sequence_frame(bgr, idx, method, mog2)
            results[p.name] = len(boxes)

            if (idx + 1) % progress_every == 0 or idx == total - 1:
                t = time.perf_counter() - seq_start
                fps = (idx + 1) / t if t > 0 else 0.0
                width = len(str(total))
                print(f"  [{idx+1:>{width}}/{total}]  count={results[p.name]:>2}  fps={fps:.1f}")

        print(f"  Done: {len(results)}/{total} frames with predictions.")
        return results

    def sequence_counts_subset(
        self,
        folder_path: str,
        *,
        method: str,
        wanted_files: set[str],
        progress_every: int = 200,
    ) -> dict[str, int]:
        """Like :meth:`sequence_counts`, but only returns counts for ``wanted_files``.

        For sequential methods (mog2/combined), this still processes frames in order
        (to keep the background model correct), but stops once it has processed up to
        the last needed frame in filename order.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_paths = list_image_files(folder)
        if not image_paths:
            raise RuntimeError(f"No images found in: {folder_path}")

        # Determine how far we must process in the ordered list.
        idx_by_name = {p.name: i for i, p in enumerate(image_paths)}
        present = wanted_files & set(idx_by_name)
        if not present:
            return {}
        last_idx = max(idx_by_name[n] for n in present)

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=self._mog2_var_threshold,
            detectShadows=False,
        )

        results: dict[str, int] = {}
        total = last_idx + 1
        seq_start = time.perf_counter()

        print(f"\n[PersonCounter.sequence_counts_subset] up_to={total}  method={method}  wanted={len(present)}")
        if method in ("mog2", "combined"):
            print(f"  MOG2 warmup: first {self._mog2_warmup} frames (HOG during warmup)")

        for idx, p in enumerate(image_paths[: total]):
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue

            boxes, _ = self._person_boxes_for_sequence_frame(bgr, idx, method, mog2)
            if p.name in wanted_files:
                results[p.name] = len(boxes)

            if (idx + 1) % progress_every == 0 or idx == total - 1:
                t = time.perf_counter() - seq_start
                fps = (idx + 1) / t if t > 0 else 0.0
                width = len(str(total))
                print(f"  [{idx+1:>{width}}/{total}]  collected={len(results):>3}  fps={fps:.1f}")

        return results

    # ------------------------------------------------------------------
    # Public API — sequential frame processing
    # ------------------------------------------------------------------

    def process_frame_sequence(
        self,
        folder_path: str,
        output_dir: str = "output",
        method: str = "hog",
        save_frames: bool = True,
        save_csv: bool = True,
        progress_every: int = 100,
    ) -> dict[str, int]:
        """Process a sorted folder of frames in sequence order.

        Parameters
        ----------
        folder_path:
            Directory containing image files (sorted alphabetically = temporal order).
        output_dir:
            Root output directory.  Annotated frames go to ``output_dir/annotated/``;
            the CSV is written to ``output_dir/counts.csv``.
        method:
            ``"hog"``      — HOG only; independent per frame, no warmup.
            ``"mog2"``     — Blob-based; requires sequential warmup frames.
            ``"combined"`` — HOG + MOG2 cross-validation; best for sequences.
        save_frames:
            Write an annotated copy of every frame.
        save_csv:
            Write ``counts.csv`` with per-frame counts and timing.
        progress_every:
            Print a progress line every N frames.

        Returns
        -------
        dict mapping filename → person count for every processed frame.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_paths = list_image_files(folder)
        if not image_paths:
            raise RuntimeError(f"No images found in: {folder_path}")

        frames_dir = Path(output_dir) / "annotated"
        if save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        # Fresh background model for this sequence
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=self._mog2_var_threshold,
            detectShadows=False,
        )

        results: dict[str, int] = {}
        csv_rows: list[dict] = []
        total = len(image_paths)
        seq_start = time.perf_counter()

        print(f"\n[PersonCounter] {total} frames  |  method={method}  "
              f"save_frames={save_frames}  save_csv={save_csv}")
        if method in ("mog2", "combined"):
            print(f"  MOG2 warmup: first {self._mog2_warmup} frames "
                  f"(HOG fallback during warmup)")

        for idx, p in enumerate(image_paths):
            t0 = time.perf_counter()
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"  [skip] Cannot read {p.name}")
                continue

            boxes, used_method = self._person_boxes_for_sequence_frame(
                bgr, idx, method, mog2
            )

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            count, annotated = self._annotate_bgr(bgr, boxes, used_method)
            results[p.name] = count

            if save_frames:
                cv2.imwrite(str(frames_dir / p.name), annotated)

            csv_rows.append({
                "frame":      p.name,
                "count":      count,
                "method":     used_method,
                "elapsed_ms": elapsed_ms,
            })

            if (idx + 1) % progress_every == 0 or idx == total - 1:
                elapsed_total = time.perf_counter() - seq_start
                fps = (idx + 1) / elapsed_total
                width = len(str(total))
                print(f"  [{idx+1:>{width}}/{total}]  "
                      f"last_count={count:>2}  "
                      f"fps={fps:.1f}")

        # Write CSV
        if save_csv:
            csv_path = Path(output_dir) / "counts.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["frame", "count", "method", "elapsed_ms"]
                )
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n  CSV  → {csv_path}")

        if save_frames:
            print(f"  Frames → {frames_dir}/")

        # Summary statistics
        counts = list(results.values())
        if counts:
            min_c = min(counts);  max_c = max(counts)
            mean_c = sum(counts) / len(counts)
            min_f  = next(f for f, c in results.items() if c == min_c)
            max_f  = next(f for f, c in results.items() if c == max_c)
            total_elapsed = time.perf_counter() - seq_start
            print(f"\n  === Summary ===")
            print(f"  Frames processed : {len(counts)}")
            print(f"  Method           : {method}")
            print(f"  Mean per frame   : {mean_c:.1f}")
            print(f"  Min              : {min_c}  ({min_f})")
            print(f"  Max              : {max_c}  ({max_f})")
            print(f"  Total time       : {total_elapsed:.1f}s  "
                  f"({len(counts) / total_elapsed:.1f} fps avg)")

        return results

    # ------------------------------------------------------------------
    # Public API — live webcam
    # ------------------------------------------------------------------

    def count_from_webcam(
        self,
        camera_index: int = 0,
        window_name: str = "Person Counter",
        smooth_frames: int = 5,
    ) -> None:
        """Live webcam person counting with HOG and temporal smoothing."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\nWebcam (person/HOG): q or Esc to quit.")

        recent_counts: list[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = self._detect_hog(frame)

            recent_counts.append(len(boxes))
            if len(recent_counts) > smooth_frames:
                recent_counts.pop(0)
            display_count = round(sum(recent_counts) / len(recent_counts))

            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"People (HOG): {display_count}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 8), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (16, 16 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyWindow(window_name)


# ---------------------------------------------------------------------------
# Train / test evaluation vs ground-truth CSV labels
# ---------------------------------------------------------------------------

def evaluate_face_counter_against_csv(
    *,
    images_dir: str | Path,
    labels_csv: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    model: str = "hog",
    enhance: bool = True,
    sharpen: bool = False,
    nms_iou: float = 0.35,
    tune_hparams: bool = False,
    limit: int | None = None,
    chronological_split: bool = False,
    sensitive_counting: bool | None = None,
    min_face_size: int | None = None,
) -> CountEvalReport:
    """Evaluate :class:`FaceCounterSystem` on still images with a ``file,count`` CSV.

    Splits labeled images that exist on disk into train/test sets (random by
    default; use ``chronological_split`` for a stable filename-order split).
    Optional ``tune_hparams`` picks the lowest train MAE among a small grid of
    ``sensitive_counting`` / ``min_face_size`` presets, then scores test data.
    """
    folder = Path(images_dir)
    labels = load_count_labels_csv(labels_csv)
    labeled_names = set(labels)
    rng = random.Random(seed)

    disk_images = {
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    }
    missing_files = len(labeled_names - disk_images)
    orphan_files = len(disk_images - labeled_names)

    paths = _natural_sort_paths(
        [folder / n for n in labeled_names if n in disk_images]
    )
    if not paths:
        raise ValueError(
            f"No CSV labels matched image files under {folder}. "
            f"(missing on disk: {missing_files}, orphans: {orphan_files})"
        )

    if limit is not None:
        if chronological_split:
            paths = paths[: min(limit, len(paths))]
        else:
            p2 = paths.copy()
            rng.shuffle(p2)
            paths = sorted(p2[: min(limit, len(paths))], key=lambda x: _natural_sort_key(x.name))

    configs: list[tuple[str, dict[str, object]]] = [
        ("default_min20", {"sensitive_counting": False, "min_face_size": 20}),
        ("sensitive_min20", {"sensitive_counting": True, "min_face_size": 20}),
        ("sensitive_min15", {"sensitive_counting": True, "min_face_size": 15}),
        ("sensitive_min25", {"sensitive_counting": True, "min_face_size": 25}),
        ("default_min15", {"sensitive_counting": False, "min_face_size": 15}),
    ]

    if not tune_hparams and (
        sensitive_counting is not None or min_face_size is not None
    ):
        sens = False if sensitive_counting is None else sensitive_counting
        mf = 20 if min_face_size is None else int(min_face_size)
        configs = [("env_matched", {"sensitive_counting": sens, "min_face_size": mf})]


    base_kw: dict[str, object] = {
        "model": model,
        "enhance": enhance,
        "sharpen": sharpen,
        "nms_iou": nms_iou,
    }

    def predict_items(
        counter: FaceCounterSystem,
        subset: list[tuple[Path, int]],
    ) -> tuple[list[tuple[int, int]], int]:
        pairs: list[tuple[int, int]] = []
        fails = 0
        for p, truth in subset:
            pred = counter.count_faces_only(str(p))
            if pred is None:
                fails += 1
                continue
            pairs.append((pred, truth))
        return pairs, fails

    items: list[tuple[Path, int]] = [(p, labels[p.name]) for p in paths]
    work = items.copy()
    if not chronological_split:
        rng.shuffle(work)

    n = len(work)
    if n == 0:
        raise ValueError("No items to evaluate after filtering.")

    if n == 1:
        train_items, test_items = work, []
    else:
        k = int(n * train_ratio)
        k = max(1, min(k, n - 1))
        train_items, test_items = work[:k], work[k:]

    def _mae(pairs: list[tuple[int, int]]) -> float:
        if not pairs:
            return float("inf")
        return sum(abs(a - b) for a, b in pairs) / len(pairs)

    chosen_name, chosen_ov = configs[0]

    if tune_hparams:
        best_mae = float("inf")
        for name, ov in configs:
            c = FaceCounterSystem(**base_kw, **ov)  # type: ignore[arg-type]
            tr_pairs, _ = predict_items(c, train_items)
            m = _mae(tr_pairs)
            if m < best_mae:
                best_mae = m
                chosen_name, chosen_ov = name, ov

    final = FaceCounterSystem(**base_kw, **chosen_ov)  # type: ignore[arg-type]
    train_pairs, lf_tr = predict_items(final, train_items)
    test_pairs, lf_te = predict_items(final, test_items)

    tm, trmse, tex, tn = _mae_rmse_exact(train_pairs)
    vm, vrmse, vex, vn = _mae_rmse_exact(test_pairs)

    return CountEvalReport(
        train_mae=tm,
        train_rmse=trmse,
        train_exact_rate=tex,
        train_n=tn,
        test_mae=vm,
        test_rmse=vrmse,
        test_exact_rate=vex,
        test_n=vn,
        labels_in_csv=len(labels),
        matched_disk=len(paths),
        missing_files=missing_files,
        orphan_files=orphan_files,
        load_failures=lf_tr + lf_te,
        train_ratio=train_ratio,
        seed=seed,
        best_config=f"{chosen_name} ({chosen_ov})" if tune_hparams else f"{chosen_name}",
    )


def evaluate_person_counter_against_csv(
    *,
    frames_dir: str | Path,
    labels_csv: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    method: str = "hog",
    limit: int | None = None,
    chronological_split: bool = True,
    progress_every: int = 200,
) -> CountEvalReport:
    """Score pedestrian counts vs a ground-truth ``file,count`` CSV."""
    labels = load_count_labels_csv(labels_csv)
    folder = Path(frames_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    disk = {p.name for p in list_image_files(folder)}
    labeled_names = set(labels)
    missing_files = len(labeled_names - disk)
    orphan_files = len(disk - labeled_names)
    common = sorted(labeled_names & disk, key=_natural_sort_key)
    if limit is not None:
        if chronological_split:
            common = common[: min(limit, len(common))]
        else:
            rng = random.Random(seed)
            tmp = common.copy()
            rng.shuffle(tmp)
            common = tmp[: min(limit, len(tmp))]
    if not common:
        raise ValueError(
            "No overlap between CSV labels and image files "
            f"in {folder}. (missing_files={missing_files}, orphans={orphan_files})"
        )

    pcs = PersonCounterSystem()

    ordered_pairs: list[tuple[int, int]] = []
    load_failures = 0

    if method == "hog":
        # Frames are independent → only read/process the requested subset.
        for name in common:
            pred = pcs.count_people_hog_only(str(folder / name))
            if pred is None:
                load_failures += 1
                continue
            ordered_pairs.append((pred, labels[name]))
    else:
        # Sequential methods need ordered processing up to the last needed frame.
        wanted = set(common)
        all_counts = pcs.sequence_counts_subset(
            str(folder),
            method=method,
            wanted_files=wanted,
            progress_every=progress_every,
        )
        for name in common:
            if name not in all_counts:
                load_failures += 1
                continue
            ordered_pairs.append((all_counts[name], labels[name]))

    if not ordered_pairs:
        raise RuntimeError(
            "No prediction/label pairs produced. "
            "Check that all labeled frames exist and are readable."
        )

    if chronological_split:
        order = ordered_pairs.copy()
    else:
        rng = random.Random(seed)
        order = ordered_pairs.copy()
        rng.shuffle(order)

    pn = len(order)
    if pn == 1:
        train_pairs, test_pairs = order, []
    else:
        pk = int(pn * train_ratio)
        pk = max(1, min(pk, pn - 1))
        train_pairs, test_pairs = order[:pk], order[pk:]

    tm, trmse, tex, tn = _mae_rmse_exact(train_pairs)
    vm, vrmse, vex, vn = _mae_rmse_exact(test_pairs)

    return CountEvalReport(
        train_mae=tm,
        train_rmse=trmse,
        train_exact_rate=tex,
        train_n=tn,
        test_mae=vm,
        test_rmse=vrmse,
        test_exact_rate=vex,
        test_n=vn,
        labels_in_csv=len(labels),
        matched_disk=len(common),
        missing_files=missing_files,
        orphan_files=orphan_files,
        load_failures=load_failures,
        train_ratio=train_ratio,
        seed=seed,
        best_config=f"method={method}",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Count faces (dlib HOG/CNN) or people (OpenCV HOG + MOG2).\n\n"
            "Detector guide:\n"
            "  face   → dlib frontal-face detector; portraits and well-lit photos.\n"
            "  person → OpenCV HOG pedestrian + optional MOG2 background subtraction;\n"
            "           ideal for surveillance / overhead / CCTV footage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--detector", choices=["face", "person"], default="face",
        help="'face': dlib frontal-face detector. "
             "'person': HOG pedestrian + MOG2 for surveillance/overhead footage.",
    )
    # Face-detector flags
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog",
                        help="[face] dlib model.")
    parser.add_argument("--sensitive", action="store_true",
                        help="[face] Higher upsample + larger max-side.")
    parser.add_argument("--sharpen", action="store_true",
                        help="[face] Apply unsharp-mask sharpening.")
    parser.add_argument("--no-enhance", action="store_true",
                        help="[face] Disable CLAHE contrast enhancement.")
    # Person-detector flags
    parser.add_argument(
        "--method", choices=["hog", "mog2", "combined"], default="hog",
        help="[person] hog=single-frame; mog2=background subtraction; "
             "combined=HOG filtered by MOG2 foreground (best for sequences).",
    )
    parser.add_argument("--no-save-frames", action="store_true",
                        help="[person, sequence] Skip saving annotated frames.")
    parser.add_argument("--no-csv", action="store_true",
                        help="[person, sequence] Skip writing counts.csv.")
    parser.add_argument(
        "--eval-face", action="store_true",
        help="Train/test metrics vs a CSV of true face counts (see --images-dir, --labels-csv).",
    )
    parser.add_argument(
        "--eval-person", action="store_true",
        help="Train/test metrics vs a CSV of true pedestrian counts for a frame folder.",
    )
    parser.add_argument(
        "--images-dir", default="image_data",
        help="[eval-face] Still image directory (default: image_data).",
    )
    parser.add_argument(
        "--frames-dir", default="frames",
        help="[eval-person] Frame directory (default: frames).",
    )
    parser.add_argument(
        "--labels-csv", default=None,
        help="Ground-truth CSV with columns file,count. If omitted for --eval-face, "
             "uses output_image_data_test/image_data_counts.csv when that file exists.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of labeled items used for training metrics (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffled train/test split.")
    parser.add_argument(
        "--tune-hparams", action="store_true",
        help="[eval-face] Pick sensitive/min_face preset with lowest train MAE.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of labeled files to use (smoke tests).",
    )
    parser.add_argument(
        "--split", choices=("random", "chronological"), default="random",
        help="Train/test assignment: random shuffle vs file-name order.",
    )
    parser.add_argument(
        "--eval-report", default=None,
        help="Write evaluation metrics JSON to this file.",
    )
    args, _ = parser.parse_known_args()

    if args.eval_face:
        csv_path = args.labels_csv
        if csv_path is None:
            fallback = Path("output_image_data_test/image_data_counts.csv")
            csv_path = str(fallback) if fallback.is_file() else None
        if not csv_path:
            raise SystemExit(
                "Provide --labels-csv FILE (or place "
                "output_image_data_test/image_data_counts.csv)."
            )
        report = evaluate_face_counter_against_csv(
            images_dir=args.images_dir,
            labels_csv=csv_path,
            train_ratio=args.train_ratio,
            seed=args.seed,
            model=args.model,
            enhance=not args.no_enhance,
            sharpen=args.sharpen,
            tune_hparams=args.tune_hparams,
            limit=args.limit,
            chronological_split=(args.split == "chronological"),
            sensitive_counting=bool(args.sensitive),
        )
        out = json.dumps(asdict(report), indent=2)
        print(out)
        if args.eval_report:
            Path(args.eval_report).write_text(out + "\n", encoding="utf-8")
        return

    if args.eval_person:
        if not args.labels_csv:
            raise SystemExit("--labels-csv is required for --eval-person (true pedestrian counts).")
        report = evaluate_person_counter_against_csv(
            frames_dir=args.frames_dir,
            labels_csv=args.labels_csv,
            train_ratio=args.train_ratio,
            seed=args.seed,
            method=args.method,
            limit=args.limit,
            chronological_split=(args.split == "chronological"),
        )
        out = json.dumps(asdict(report), indent=2)
        print(out)
        if args.eval_report:
            Path(args.eval_report).write_text(out + "\n", encoding="utf-8")
        return

    if args.detector == "person":
        system = PersonCounterSystem()
        print(f"\n--- Person Counter (method={args.method}) ---")
        print("1. Single image  2. Frame sequence  3. Webcam  4. Exit")
        choice = input("Choice [1-4]: ").strip()

        if choice == "1":
            image_path = input("Image path: ").strip()
            system.count_people_in_image(image_path, output_dir="output")
        elif choice == "2":
            folder_path = input("Frame sequence folder: ").strip()
            system.process_frame_sequence(
                folder_path,
                output_dir="output",
                method=args.method,
                save_frames=not args.no_save_frames,
                save_csv=not args.no_csv,
            )
        elif choice == "3":
            system.count_from_webcam()
        else:
            print("Bye.")

    else:  # face detector (original behaviour)
        system = FaceCounterSystem(
            model=args.model,
            sensitive_counting=args.sensitive,
            enhance=not args.no_enhance,
            sharpen=args.sharpen,
        )
        print(f"\n--- Face Counter (model={args.model}, sensitive={args.sensitive}) ---")
        print("1. One image  2. Folder  3. Webcam  4. Exit")
        choice = input("Choice [1-4]: ").strip()

        if choice == "1":
            image_path = input("Image path: ").strip()
            system.count_faces_in_image(image_path, output_dir="output")
        elif choice == "2":
            folder_path = input("Folder path: ").strip()
            results = system.count_faces_in_folder(folder_path, output_dir="output")
            print(f"Total faces (sum): {sum(results.values())}")
        elif choice == "3":
            system.count_from_webcam(camera_index=0)
        else:
            print("Bye.")


if __name__ == "__main__":
    main()
