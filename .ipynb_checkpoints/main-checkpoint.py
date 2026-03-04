import face_recognition
import cv2
import numpy as np
import os

# List of (index, encoding) — index: int, encoding: face encoding array or encrypted image bytes
face_storage: list[tuple[int, np.ndarray | bytes]] = []


def add_face(index: int, encoding: np.ndarray | bytes) -> None:
    """Append an index and its face encoding (or encrypted picture data) to the list."""
    face_storage.append((index, encoding))


def get_face(index: int) -> np.ndarray | bytes | None:
    """Return the encoding/encrypted data for the given index, or None if not found."""
    for i, enc in face_storage:
        if i == index:
            return enc
    return None


def face_recognition():
    pass