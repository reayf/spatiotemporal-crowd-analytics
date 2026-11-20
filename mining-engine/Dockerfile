FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
# CPU-only torch wheel first (much smaller than the default CUDA build),
# then the rest of the requirements.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/data \
    MINER=kmeans \
    FORECAST_HORIZON_SECONDS=60.0
CMD ["python", "-m", "app.main"]
