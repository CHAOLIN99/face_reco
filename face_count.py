import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path

class FaceCounterSystem:
    def __init__(self, model: str = "cnn"):
        """
        model:
          - 'hog' (default): fast on CPU
          - 'cnn': more accurate but needs a good GPU / slower on CPU
        """
        self.model = model

    def count_faces_in_image(
        self,
        image_path: str,
        output_dir: str = "output",
        upsample: int = 1,
        draw_block: bool = True,
        block_alpha: float = 0.55,
    ) -> int:
        """Detect faces in a single image and return the count.

        Writes an annotated image to `output_dir`:
        - If `draw_block` is True: draws a semi-transparent filled block over each face
        - Otherwise: draws a rectangle outline
        """
        os.makedirs(output_dir, exist_ok=True)

        rgb = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=upsample,
            model=self.model,
        )
        count = len(face_locations)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        for (top, right, bottom, left) in face_locations:
            if draw_block:
                # Filled block (more obvious than outline)
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 150, 0), 2)
            else:
                cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        if draw_block and count > 0:
            # Blend overlay so the block doesn't fully hide the image
            alpha = float(block_alpha)
            alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
            bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

        label = f"People detected: {count}"
        cv2.rectangle(bgr, (10, 10), (10 + 320, 10 + 40), (0, 0, 0), cv2.FILLED)
        cv2.putText(bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        output_path = os.path.join(output_dir, f"count_{Path(image_path).name}")
        cv2.imwrite(output_path, bgr)
        print(f"{Path(image_path).name}: {count} face(s). Saved: {output_path}")
        return count

    def count_faces_in_folder(
        self,
        folder_path: str,
        output_dir: str = "output",
        upsample: int = 1,
        draw_block: bool = True,
    ) -> dict[str, int]:
        """Count faces for every image in a folder. Returns {filename: count}."""
        results: dict[str, int] = {}
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            results[p.name] = self.count_faces_in_image(
                str(p),
                output_dir=output_dir,
                upsample=upsample,
                draw_block=draw_block,
            )
        return results

    def count_from_webcam(
        self,
        camera_index: int = 0,
        downscale: float = 0.5,
        upsample: int = 1,
        window_name: str = "Face Counter",
    ):
        """Real-time face counting from webcam.

        Tips:
        - Increase `downscale` (e.g. 0.75 or 1.0) for better detection (slower).
        - Increase `upsample` for smaller/farther faces (slower).
        - Use model='cnn' for better accuracy (much slower on CPU).
        """
        video_capture = cv2.VideoCapture(camera_index)
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\nStarting webcam... Press 'q' or 'Esc' to quit (click the video window first).")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if downscale != 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=downscale, fy=downscale)
            else:
                small_frame = frame

            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                number_of_times_to_upsample=upsample,
                model=self.model,
            )

            for (top, right, bottom, left) in face_locations:
                # Scale back up
                if downscale != 1.0:
                    inv = 1.0 / downscale
                    top = int(top * inv)
                    right = int(right * inv)
                    bottom = int(bottom * inv)
                    left = int(left * inv)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            count = len(face_locations)
            cv2.rectangle(frame, (10, 10), (10 + 320, 10 + 40), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"People: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

            # If the user closes the window, exit cleanly.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyWindow(window_name)

def main():
    system = FaceCounterSystem(model="hog")

    print("\n--- Face Counting System ---")
    print("1. Count people in one image")
    print("2. Count people in a folder of images")
    print("3. Real-time webcam counting")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        image_path = input("Enter path to image: ").strip()
        system.count_faces_in_image(image_path, output_dir="output", upsample=1, draw_block=True)
    elif choice == "2":
        folder_path = input("Enter path to folder: ").strip()
        results = system.count_faces_in_folder(folder_path, output_dir="output", upsample=1, draw_block=True)
        total = sum(results.values())
        print(f"\nProcessed {len(results)} image(s). Total faces across all images: {total}")
    elif choice == "3":
        # For better detection (occlusion / small faces), try:
        # - model: FaceCounterSystem(model="cnn")
        # - downscale: 0.75 or 1.0 (slower but more accurate)
        # - upsample: 1 or 2 (slower but detects smaller faces)
        system.count_from_webcam(camera_index=0, downscale=0.5, upsample=1)
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()