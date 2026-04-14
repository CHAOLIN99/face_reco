from __future__ import annotations

import csv
import os
import time
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
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
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

        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
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

            # Feed frame to MOG2 regardless of warmup — keeps model current
            fg_mask: np.ndarray | None = None
            if method in ("mog2", "combined"):
                fg_mask = mog2.apply(bgr)

            in_warmup = (method in ("mog2", "combined")) and (idx < self._mog2_warmup)

            if method == "hog" or in_warmup:
                boxes      = self._detect_hog(bgr)
                used_method = "hog"
            elif method == "mog2":
                boxes      = self._detect_blobs(fg_mask)   # type: ignore[arg-type]
                used_method = "mog2"
            else:                                           # combined (post-warmup)
                boxes      = self._detect_combined(bgr, fg_mask)  # type: ignore[arg-type]
                used_method = "combined"

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
    args, _ = parser.parse_known_args()

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
