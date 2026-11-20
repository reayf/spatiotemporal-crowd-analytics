"""End-to-end smoke test of the pipeline plumbing (no YOLO inference).

Boots the real Go gateway and the real mining engine against a throwaway
data directory, uploads a dummy video over REST, then plays the CV engine's
role: claims the tracking manifest from the file buffer, writes a synthetic
trajectory log, and publishes the mining job. Asserts that:

1. POST /api/v1/videos returns 202 with job/video ids,
2. the ingestion worker dispatches a manifest to queue/tracking/,
3. the mining engine consumes the job and publishes analytics,
4. GET /api/v1/analytics/:video_id serves the report (202 -> 200),
5. the completion watcher flips the job to COMPLETED.

Usage (from the repo root)::

    python scripts/e2e_smoke.py

Requires the Go toolchain and the mining engine's Python deps
(numpy, pandas, scikit-learn).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "gateway-go"
MINING_DIR = REPO_ROOT / "mining-engine"
PORT = "8091"
BASE_URL = f"http://127.0.0.1:{PORT}"


def http_get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8") or "{}")


def upload_dummy_video() -> dict:
    boundary = "----e2e-smoke-boundary"
    payload = b"\x00" * 4096  # content is irrelevant: the CV role is simulated
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="video"; filename="sample.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{BASE_URL}/api/v1/videos",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        assert resp.status == 202, f"expected 202, got {resp.status}"
        return json.loads(resp.read().decode("utf-8"))


def wait_for(description: str, predicate, timeout_s: float = 30.0, interval_s: float = 0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval_s)
    raise TimeoutError(f"timed out waiting for: {description}")


def synthetic_trajectory_log(video_id: str) -> dict:
    """Two-hotspot synthetic scene, matching the CV engine's output schema."""
    centres = [(200.0, 200.0), (1000.0, 500.0)]
    fps = 10.0
    trajectories = []
    for track_id in range(6):
        cx, cy = centres[track_id % 2]
        t0 = track_id * 1.5
        points = [
            {
                "x": round(cx + 25.0 * math.cos(k / 6.0 + track_id), 2),
                "y": round(cy + 25.0 * math.sin(k / 4.0 + track_id), 2),
                "t": round(t0 + k / fps, 4),
                "frame": int((t0 + k / fps) * fps),
            }
            for k in range(200)
        ]
        trajectories.append(
            {"track_id": track_id, "class_name": "person", "points": points}
        )
    return {
        "video_id": video_id,
        "metadata": {"fps": fps, "frame_width": 1280, "frame_height": 720},
        "trajectories": trajectories,
    }


def main() -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="stpp-e2e-"))
    exe = data_dir / ("gateway.exe" if os.name == "nt" else "gateway")

    print(f"[e2e] data dir: {data_dir}")
    print("[e2e] building gateway ...")
    subprocess.run(
        ["go", "build", "-o", str(exe), "./cmd/server"],
        cwd=GATEWAY_DIR, check=True,
    )

    gateway_env = {
        **os.environ,
        "GATEWAY_PORT": PORT,
        "DATA_DIR": str(data_dir),
        "WATCH_INTERVAL_SECONDS": "1",
    }
    mining_env = {
        **os.environ,
        "DATA_DIR": str(data_dir),
        "POLL_INTERVAL_SECONDS": "0.5",
        "MINER": "kmeans",
        "FORECAST_HORIZON_SECONDS": "30.0",
    }

    gateway_log = open(data_dir / "gateway.log", "w", encoding="utf-8")
    mining_log = open(data_dir / "mining.log", "w", encoding="utf-8")
    gateway = subprocess.Popen(
        [str(exe)], env=gateway_env, cwd=GATEWAY_DIR,
        stdout=gateway_log, stderr=subprocess.STDOUT,
    )
    mining = subprocess.Popen(
        [sys.executable, "-m", "app.main"], env=mining_env, cwd=MINING_DIR,
        stdout=mining_log, stderr=subprocess.STDOUT,
    )

    try:
        wait_for("gateway /healthz", lambda: _healthy(), timeout_s=15)
        print("[e2e] gateway is up")

        accepted = upload_dummy_video()
        job_id, video_id = accepted["job_id"], accepted["video_id"]
        print(f"[e2e] accepted: job={job_id} video={video_id}")

        # 2) worker dispatches the manifest into the tracking queue
        manifest_path = wait_for(
            "tracking manifest",
            lambda: next(iter((data_dir / "queue" / "tracking").glob("*.json")), None),
            timeout_s=10,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["video_id"] == video_id
        assert Path(manifest["video_path"]).exists(), "uploaded video missing on disk"
        status, job = http_get(f"/api/v1/jobs/{job_id}")
        assert status == 200 and job["status"] == "DISPATCHED", job
        print("[e2e] job dispatched, manifest verified")

        # 3) play the CV engine: claim manifest, publish trajectories + mining job
        manifest_path.unlink()
        log = synthetic_trajectory_log(video_id)
        trajectory_path = data_dir / "trajectories" / f"{video_id}.json"
        trajectory_path.write_text(json.dumps(log), encoding="utf-8")
        (data_dir / "queue" / "mining" / f"{job_id}.json").write_text(
            json.dumps({
                "job_id": job_id,
                "video_id": video_id,
                "trajectory_path": str(trajectory_path),
            }),
            encoding="utf-8",
        )
        print("[e2e] simulated CV engine output published")

        # 4) real mining engine -> analytics served over REST
        report = wait_for(
            "analytics via REST",
            lambda: _analytics_ready(video_id),
            timeout_s=60,
        )
        assert report["miner"] == "kmeans_hotspot"
        assert report["observed_track_count"] == 6
        assert len(report["hotspots"]) > 0 and len(report["forecasts"]) > 0
        print(f"[e2e] analytics served: {len(report['hotspots'])} hotspots, "
              f"{len(report['forecasts'])} forecasts")

        # 5) completion watcher flips the job to COMPLETED
        wait_for(
            "job COMPLETED",
            lambda: http_get(f"/api/v1/jobs/{job_id}")[1].get("status") == "COMPLETED",
            timeout_s=15,
        )
        print("[e2e] job reached COMPLETED — PASS")
        return 0
    except BaseException:
        for name in ("gateway.log", "mining.log"):
            log_file = data_dir / name
            if log_file.exists():
                print(f"----- {name} -----\n{log_file.read_text(encoding='utf-8')}")
        raise
    finally:
        for proc in (gateway, mining):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        gateway_log.close()
        mining_log.close()


def _healthy() -> bool:
    try:
        return http_get("/healthz")[0] == 200
    except (urllib.error.URLError, OSError):
        return False


def _analytics_ready(video_id: str) -> dict | None:
    status, body = http_get(f"/api/v1/analytics/{video_id}")
    return body if status == 200 else None


if __name__ == "__main__":
    sys.exit(main())
