# Face Studio — Flask + OpenCV + dlib (face_recognition). Use for cloud deploys (GHCR, Render, Fly, etc.).
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py face_count.py face_rec.py ./
COPY templates ./templates
COPY static ./static
COPY data/known_faces ./data/known_faces

ENV PORT=8080
EXPOSE 8080

CMD gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 2 --timeout 120 app:app
