# Spatiotemporal Predictive Tracking & Point Process Analytics

A microservices portfolio project that tracks physical movement in CCTV-style
video, extracts trajectory logs, and applies spatiotemporal data mining
(point processes + clustering) to analyse and **predict** future crowd
dynamics (congestion, dwell times, hotspots).

## Architecture

```
             ┌─────────────────────────────────────────────────────────┐
             │                  JSON File Buffer (data/)                │
  upload     │      simulated message broker between services           │
 ────────►┌──┴───────┐  manifest   ┌───────────┐  trajectories  ┌───────┴──────┐
          │ Go API   │────────────►│ CV Engine │───────────────►│ Mining Engine│
          │ Gateway  │ queue/      │ (Python)  │  + manifest    │  (Python)    │
          │ (Gin)    │ tracking/   │ YOLOv8 +  │  queue/mining/ │ K-Means /    │
          │          │             │ ByteTrack │                │ DeepSTPP     │
          └──┬───────┘             └───────────┘                └───────┬──────┘
             │  GET /api/v1/analytics/:video_id      analytics/         │
             ◄───────────────────────────────────────────────────────────
```

**Data flow**

1. Client `POST /api/v1/videos` (multipart) → gateway saves the video to
   `data/videos/` and a goroutine worker pool asynchronously publishes a job
   manifest to `data/queue/tracking/` (202 Accepted with `job_id`/`video_id`).
2. The **CV engine** claims the manifest (atomic rename = broker ack), runs
   YOLOv8 detection + ByteTrack association frame-by-frame, writes the
   trajectory log `(track_id, x, y, t)` to `data/trajectories/<video_id>.json`,
   and publishes a follow-up manifest to `data/queue/mining/`.
3. The **mining engine** claims it, transforms the raw coordinates into
   time-series / event-sequence features, fits the selected miner (K-Means
   hotspots + occupancy regression, or simplified DeepSTPP intensity model),
   and writes the report to `data/analytics/<video_id>.json`.
4. Client polls `GET /api/v1/analytics/:video_id` — 202 while pending,
   200 with the report once published.

## Services

| Service | Stack | Responsibility |
|---|---|---|
| `gateway-go/` | Go 1.22, Gin, goroutine worker pool | Async ingestion, job status, serving analytics |
| `cv-engine/` | Python 3.10+, Ultralytics YOLOv8, ByteTrack | MOT tracking → trajectory logs |
| `mining-engine/` | Python 3.10+, PyTorch, scikit-learn, pandas | Point-process / clustering analytics & forecasting |

Both Python services follow Clean Architecture: `domain/` (entities) ←
`services/` (use cases) ← `trackers|miners/` (strategy implementations behind
abstract base classes) ← `infrastructure/` (file-buffer adapters) ←
`main.py` (composition root).

Key abstractions:

- `AbstractTracker` (`cv-engine/app/trackers/abstract_tracker.py`) —
  implemented by `YoloByteTracker`.
- `SpatiotemporalMiner` (`mining-engine/app/miners/base_miner.py`) —
  implemented by `KMeansHotspotMiner` and `DeepSTPPMiner` (Strategy pattern;
  swap them in `mining-engine/app/main.py`).

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/videos` | Multipart field `video`; returns 202 + `job_id` |
| `GET` | `/api/v1/jobs/:id` | Ingestion job status |
| `GET` | `/api/v1/analytics/:video_id` | Analytics report (202 while pending) |
| `GET` | `/healthz` | Liveness probe |

## Running (Docker)

```bash
docker compose up --build
# then upload as below against http://localhost:8080
```

## Integration test

`scripts/e2e_smoke.py` boots the real gateway and mining engine against a
throwaway data dir, uploads a video over REST, simulates the CV engine's
file-buffer output, and asserts analytics are served and the job reaches
COMPLETED (needs Go + the mining engine's Python deps, no GPU/YOLO):

```bash
python scripts/e2e_smoke.py
```

## Running (development)

Each service reads `DATA_DIR` (default `../data`, i.e. start each one from
its own directory).

```bash
# 1. Gateway
cd gateway-go
go mod tidy
go run ./cmd/server

# 2. CV engine
cd cv-engine
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m app.main

# 3. Mining engine
cd mining-engine
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m app.main

# 4. Ingest a test video (MOT17/MOT20 clip or Oxford Town Centre sample)
curl -F "video=@towncentre.mp4" http://localhost:8080/api/v1/videos
curl http://localhost:8080/api/v1/analytics/<video_id>
```

## Roadmap

- [x] **Phase 1** — Project structure (Go + 2 Python microservices + file buffer)
- [x] **Phase 2** — Go gateway: Gin routing, async goroutine job queue, file-buffer dispatch
- [x] **Phase 3** — Python OOP skeletons: `AbstractTracker`, `SpatiotemporalMiner`, entities, adapters
- [x] **Phase 4** — Implement `YoloByteTracker` + video processing loop
- [x] **Phase 5** — Implement `KMeansHotspotMiner` and `DeepSTPPMiner` (select via `MINER=kmeans|deepstpp`)
- [x] **Phase 6** — Gateway completion watcher, Dockerfiles + compose, end-to-end integration test (`scripts/e2e_smoke.py`)
- [x] **Phase 7** — Live run verified on a real pedestrian clip (Intel `people-detection.mp4`, YOLOv8n + ByteTrack on CPU): 596 frames tracked in ~12s, 7 trajectories extracted, hotspots + forecasts served over REST, job auto-completed by the watcher
