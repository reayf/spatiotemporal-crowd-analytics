FROM python:3.11-slim

# OpenCV runtime dependencies (headless video decoding).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/data \
    YOLO_MODEL=yolov8n.pt \
    DEVICE=cpu
CMD ["python", "-m", "app.main"]
