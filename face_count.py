import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import face_recognition
import numpy as np


class FaceCounterSystem:
    """
    Face counting with ``face_recognition`` (dlib HOG or CNN) only — no Haar merge,
    to avoid counting non-face patterns.

    **Default (strict):** moderate resize + single upsample — fewer false positives.
    **Sensitive mode** (``sensitive_counting=True`` or ``FACE_COUNT_SENSITIVE=1``):
    keeps more resolution and uses higher upsampling for small / distant faces;
    still dlib-only (may miss some faces vs CNN).
    """

    DEFAULT_MAX_SIDE_SENSITIVE = 4096
    HARD_MAX_SIDE = 8000
    DEFAULT_UPSAMPLE_SENSITIVE = 2
    DEFAULT_MAX_SIDE_FAST = 1600
    DEFAULT_UPSAMPLE_FAST = 1

    def __init__(self, model: str = "hog", *, sensitive_counting: bool = False):
        if model not in ("hog", "cnn"):
            raise ValueError("model must be 'hog' or 'cnn'")
        self.model = model
        self.sensitive_counting = sensitive_counting

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        return face_recognition.load_image_file(path)

    def _locations_hog_cnn(
        self,
        rgb: np.ndarray,
        *,
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
        return self._locations_hog_cnn(rgb, upsample=upsample)

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

        for (top, right, bottom, left) in face_locations:
            if draw_block:
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 150, 0), 2)
            else:
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        if draw_block and count > 0:
            alpha = max(0.0, min(1.0, float(block_alpha)))
            bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

        label = f"People detected: {count}"
        cv2.rectangle(bgr, (10, 10), (10 + 340, 10 + 42), (0, 0, 0), cv2.FILLED)
        cv2.putText(bgr, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return count, bgr

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
        """
        Shrink only when the longest side exceeds max_side.
        ``max_side <= 0`` means skip downscale except ``hard_cap`` safety limit.
        """
        h, w = rgb.shape[:2]
        m = max(h, w)
        limit = max_side if max_side > 0 else hard_cap
        if m <= limit:
            return rgb
        scale = limit / m
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    def count_faces_in_image(
        self,
        image_path: str,
        output_dir: str = "output",
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.55,
        max_side: Optional[int] = None,
    ) -> int:
        os.makedirs(output_dir, exist_ok=True)
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
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
        block_alpha: float = 0.55,
        max_side: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._load_rgb(image_path)
        rgb = self._maybe_downscale_rgb(rgb, side)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_rgb(
        self,
        rgb: np.ndarray,
        *,
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.55,
        max_side: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
        d_up, d_side = self._defaults_for_count()
        up = d_up if upsample is None else upsample
        side = d_side if max_side is None else max_side

        rgb = self._maybe_downscale_rgb(rgb, side)
        locs = self._locations(rgb, upsample=up)
        return self._annotate_bgr(rgb, locs, draw_block=draw_block, block_alpha=block_alpha)

    def count_faces_in_bytes(
        self,
        image_bytes: bytes,
        *,
        upsample: Optional[int] = None,
        draw_block: bool = True,
        block_alpha: float = 0.55,
        max_side: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
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
        downscale: float = 0.5,
        upsample: Optional[int] = None,
        window_name: str = "Face Counter",
    ) -> None:
        d_up, _ = self._defaults_for_count()
        up = d_up if upsample is None else upsample

        video_capture = cv2.VideoCapture(camera_index)
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\nWebcam: q or Esc to quit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            small_frame = (
                cv2.resize(frame, (0, 0), fx=downscale, fy=downscale)
                if downscale != 1.0
                else frame
            )
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = self._locations(rgb_small, upsample=up)

            for (top, right, bottom, left) in face_locations:
                if downscale != 1.0:
                    inv = 1.0 / downscale
                    top = int(top * inv)
                    right = int(right * inv)
                    bottom = int(bottom * inv)
                    left = int(left * inv)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            count = len(face_locations)
            cv2.rectangle(frame, (10, 10), (10 + 320, 10 + 40), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"People: {count}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyWindow(window_name)


def main() -> None:
    system = FaceCounterSystem(model="hog")

    print("\n--- Face Counting ---")
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
