import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import face_recognition
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for two (top, right, bottom, left) boxes."""
    top = max(a[0], b[0])
    right = min(a[1], b[1])
    bottom = min(a[2], b[2])
    left = max(a[3], b[3])
    inter_h = max(0, bottom - top)
    inter_w = max(0, right - left)
    inter = inter_h * inter_w
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[1] - a[3])
    area_b = (b[2] - b[0]) * (b[1] - b[3])
    return inter / float(area_a + area_b - inter)


def _nms(
    locations: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    """
    Non-Maximum Suppression: merge overlapping boxes that likely refer to the
    same face.  Keeps the largest box from each overlapping cluster.
    """
    if not locations:
        return []
    # Sort by box area descending — keep the largest representative
    by_area = sorted(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for box in by_area:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _clahe_enhance(rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L channel (LAB) to boost local contrast.
    Helps detect faces in low-light, hazy, or low-contrast images.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _sharpen(rgb: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask sharpening — recovers detail in soft/distant faces."""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FaceCounterSystem:
    """
    Face counting with ``face_recognition`` (dlib HOG or CNN).

    Key improvements over the baseline:
    - CLAHE + optional sharpening preprocessing for distant / blurry faces.
    - Multi-scale upsample passes merged with NMS to catch small faces without
      producing duplicates.
    - Minimum face-size filter to drop tiny false positives.
    - Smarter defaults: larger max_side so downscaling doesn't destroy detail.
    - Webcam: detects on full frame (no 0.5× shrink) with temporal smoothing.
    """

    # Pixel limits
    DEFAULT_MAX_SIDE_SENSITIVE = 4096
    DEFAULT_MAX_SIDE_FAST = 2400       # raised from 1600 — keeps more detail
    HARD_MAX_SIDE = 8000

    # Upsample levels
    DEFAULT_UPSAMPLE_SENSITIVE = 2
    DEFAULT_UPSAMPLE_FAST = 1

    def __init__(
        self,
        model: str = "hog",
        *,
        sensitive_counting: bool = False,
        enhance: bool = True,          # CLAHE contrast enhancement
        sharpen: bool = False,         # unsharp-mask (enable for blurry images)
        min_face_size: int = 20,       # minimum face height/width in pixels
        nms_iou: float = 0.35,        # IoU threshold for duplicate removal
    ):
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
    ) -> List[Tuple[int, int, int, int]]:
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
    ) -> List[Tuple[int, int, int, int]]:
        """
        Run detection at the requested upsample level *and* one level higher
        (to catch small/distant faces), then merge with NMS and filter by size.
        """
        locs = self._raw_locations(rgb, upsample)

        # Also run one extra upsample pass if the primary level is 0 or 1,
        # so we don't miss faces that only appear at higher resolution.
        if upsample < 2:
            extra = self._raw_locations(rgb, upsample + 1)
            locs = locs + extra

        # Remove duplicates introduced by the merged passes
        locs = _nms(locs, self.nms_iou)

        # Drop boxes smaller than min_face_size in either dimension
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
        face_locations: Sequence[Tuple[int, int, int, int]],
        *,
        draw_block: bool,
        block_alpha: float,
    ) -> Tuple[int, np.ndarray]:
        face_locations = list(face_locations)
        count = len(face_locations)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()

        for idx, (top, right, bottom, left) in enumerate(face_locations, start=1):
            if draw_block:
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 180, 0), 2)
            else:
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Small face-index label above each box
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

    def _defaults_for_count(self) -> Tuple[int, int]:
        if self.sensitive_counting:
            return self.DEFAULT_UPSAMPLE_SENSITIVE, self.DEFAULT_MAX_SIDE_SENSITIVE
        return self.DEFAULT_UPSAMPLE_FAST, self.DEFAULT_MAX_SIDE_FAST

    @staticmethod
    def _maybe_downscale_rgb(
        rgb: np.ndarray,
        max_side: int,
        hard_cap: int = HARD_MAX_SIDE,
    ) -> np.ndarray:
        h, w = rgb.shape[:2]
        m = max(h, w)
        limit = max_side if max_side > 0 else hard_cap
        if m <= limit:
            return rgb
        scale = limit / m
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_faces_in_image(
        self,
        image_path: str,
        output_dir: str = "output",
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: Optional[int] = None,
    ) -> int:
        os.makedirs(output_dir, exist_ok=True)
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        count, bgr = self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)
        output_path = os.path.join(output_dir, f"count_{Path(image_path).name}")
        cv2.imwrite(output_path, bgr)
        print(f"{Path(image_path).name}: {count} face(s). Saved: {output_path}")
        return count

    def count_faces_in_image_annotated(
        self,
        image_path: str,
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_rgb(
        self,
        rgb: np.ndarray,
        *,
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._maybe_downscale_rgb(rgb, side)
        rgb = self._preprocess(rgb)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_bytes(
        self,
        image_bytes: bytes,
        *,
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.45,
        max_side: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
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
        upsample: Optional[int] = None,
        draw_block: bool = True,
        max_side: Optional[int] = None,
    ) -> dict[str, int]:
        results: dict[str, int] = {}
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

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
        upsample: Optional[int] = None,
        window_name: str = "Face Counter",
        smooth_frames: int = 3,
    ) -> None:
        """
        Live webcam face counting.

        - Detects on a modestly downscaled frame (0.75×) for speed while
          keeping enough resolution for distant faces.
        - ``smooth_frames``: number of recent counts to average for a stable
          display number (reduces flicker from missed detections).
        """
        d_up, _ = self._defaults_for_count()
        up = d_up if upsample is None else upsample

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\nWebcam: q or Esc to quit.")

        # Detection scale: 0.75 is a good balance — much faster than full-res
        # yet keeps significantly more detail than the old 0.5×.
        DETECT_SCALE = 0.75
        recent_counts: List[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale only for detection
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

            # Temporal smoothing: show rolling-average count
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
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Face Counter")
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog",
                        help="Detection model. 'cnn' is more accurate but slower.")
    parser.add_argument("--sensitive", action="store_true",
                        help="Sensitive mode: higher upsample + larger max-side.")
    parser.add_argument("--sharpen", action="store_true",
                        help="Apply unsharp-mask sharpening (helps blurry images).")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable CLAHE contrast enhancement.")
    args, _ = parser.parse_known_args()

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
