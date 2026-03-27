import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class SimpleFaceRecognizer:
    """
    Train a small face recognizer from images on disk and predict names from new images.

    Dataset layout it expects:

        data/
          known_faces/
            alice/
              img1.jpg
              img2.png
            bob/
              001.jpg
              selfie.png
          unknown_face/
            test1.jpg
            test2.png

    - Each subfolder under `known_faces` is treated as the person's label.
    - All images inside are used as training samples.
    """

    def __init__(
        self,
        cascade_path: str | None = None,
    ):
        # You need opencv-contrib-python installed for the cv2.face module.
        # pip install opencv-contrib-python
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face module not found. Install 'opencv-contrib-python' instead of 'opencv-python'."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        if cascade_path is None:
            # Use OpenCV's built-in frontal face detector
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cannot find Haar cascade file: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.label_to_name: Dict[int, str] = {}
        self.trained: bool = False

    def _collect_images_from_folder(
        self,
        labeled_faces_root_dir: str,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Walk through labeled_faces_root_dir, read images, detect a face in each,
        and return aligned face crops + integer labels.
        """
        images: List[np.ndarray] = []
        labels: List[int] = []

        dataset_path = Path(labeled_faces_root_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"known_faces folder does not exist: {labeled_faces_root_dir}"
            )

        label_id = 0
        self.label_to_name.clear()

        for person_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
            person_name = person_dir.name
            self.label_to_name[label_id] = person_name

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
                )
                if len(faces) == 0:
                    continue

                # Use the first detected face in the image
                (x, y, w, h) = faces[0]
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, (200, 200))

                images.append(face_resized)
                labels.append(label_id)

            label_id += 1

        if not images:
            raise RuntimeError(
                "No training faces found. "
                "Make sure your dataset has images and faces can be detected."
            )

        return images, labels

    def train_from_known_faces(self, known_faces_dir: str) -> None:
        """
        Train the recognizer from a `known_faces` folder.
        """
        images, labels = self._collect_images_from_folder(known_faces_dir)
        self.recognizer.train(images, np.array(labels))
        self.trained = True

    def predict_from_image(self, image_path: str) -> Tuple[str | None, float | None]:
        """
        Predict the name of the most prominent face in an image.

        Returns (name, confidence) or (None, None) if no face / not trained.
        Lower confidence is better (LBPH uses distance, not probability).
        """
        if not self.trained:
            raise RuntimeError("Model is not trained. Call train_from_folder(...) first.")

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None, None

        (x, y, w, h) = faces[0]
        face_roi = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_roi, (200, 200))

        label_id, confidence = self.recognizer.predict(face_resized)
        name = self.label_to_name.get(label_id, "unknown")
        return name, confidence

    def predict_from_webcam(self, camera_index: int = 0, threshold: float = 80.0) -> None:
        """
        Real-time test from webcam.

        - Shows the predicted name above each detected face.
        - If confidence is above 'threshold', label will be 'unknown'.
        """
        if not self.trained:
            raise RuntimeError("Model is not trained. Call train_from_folder(...) first.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        print("Webcam started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, (200, 200))

                label_id, confidence = self.recognizer.predict(face_resized)
                name = self.label_to_name.get(label_id, "unknown")

                display_name = name
                if confidence > threshold:
                    display_name = "unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{display_name} ({confidence:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("SimpleFaceRecognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def predict_from_unknown_folder(self, unknown_face_dir: str) -> Dict[str, Tuple[str | None, float | None]]:
        """
        Predict names for each image in unknown_face_dir.
        Returns {filename: (predicted_name_or_None, confidence_or_None)}.
        """
        unknown_path = Path(unknown_face_dir)
        if not unknown_path.exists():
            raise FileNotFoundError(f"unknown_face folder does not exist: {unknown_face_dir}")

        results: Dict[str, Tuple[str | None, float | None]] = {}
        image_paths = [
            p
            for p in sorted(unknown_path.iterdir())
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if not image_paths:
            raise RuntimeError(
                f"No images found in unknown_face folder: {unknown_face_dir}"
            )

        for p in image_paths:
            name, conf = self.predict_from_image(str(p))
            results[p.name] = (name, conf)

        return results


def main() -> None:
    """
    Small CLI to train and test.

    Expects:
      - data/known_faces/<person_name>/...
      - data/unknown_face/...
    """
    recognizer = SimpleFaceRecognizer()

    print("\n--- Simple Face Recognition ---")
    base_data_dir = input("Enter path to 'data' folder (e.g. 'data'): ").strip()
    if base_data_dir == "":
        base_data_dir = "data"

    known_faces_dir = str(Path(base_data_dir) / "known_faces")
    # Common dataset naming uses 'unknown_faces' (plural).
    # We'll auto-detect a few common variants to be forgiving.
    base_path = Path(base_data_dir)
    unknown_candidates = [
        base_path / "unknown_faces",
        base_path / "unknown_face",
        base_path / "unknown",
        base_path / "test",
    ]
    unknown_face_dir = next((str(p) for p in unknown_candidates if p.exists()), str(unknown_candidates[0]))

    print(f"\nTraining from '{known_faces_dir}' ...")
    recognizer.train_from_known_faces(known_faces_dir)
    print("Training finished.")

    while True:
        print("\nOptions:")
        print("1. Predict all images in 'unknown_face' folder")
        print("2. Predict from a single image path")
        print("3. Predict from webcam (real-time)")
        print("4. Exit")
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print(f"\nRunning batch prediction from '{unknown_face_dir}' ...")
            results = recognizer.predict_from_unknown_folder(unknown_face_dir)
            for filename, (name, conf) in results.items():
                if name is None:
                    print(f"{filename}: no face detected")
                else:
                    print(f"{filename}: {name} (distance={conf:.2f})")
        elif choice == "2":
            img_path = input("Path to image to test: ").strip()
            name, conf = recognizer.predict_from_image(img_path)
            if name is None:
                print("No face detected.")
            else:
                print(f"Predicted: {name} (distance={conf:.2f})")
        elif choice == "3":
            recognizer.predict_from_webcam()
        else:
            print("Goodbye.")
            break


if __name__ == "__main__":
    main()

