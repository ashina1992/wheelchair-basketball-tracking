"""解析ジョブの保存と実行。

ジョブの状態は storage/jobs/{id}/status.json にファイルで持つ(DB は使わない)。
解析は子プロセスで1件ずつ実行する(CPU 推論で重く、失敗してもサーバーを巻き込まないように)。
解析済みのデモは storage/demo/{name}/ に analysis.json(と任意で video.mp4)を置けば一覧に出る。
"""
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

VIDEO_NAME = "video.mp4"


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _run_job(job_dir: str, profile: str, weights: str, max_seconds: float | None):
    """子プロセスで実行する解析本体。進捗は status.json に書き出す。"""
    from wbt.pipeline import analyze, save_analysis

    d = Path(job_dir)
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    last = [0.0]

    def progress(stage, frac):
        now = time.time()
        if now - last[0] < 1.0 and frac < 1.0:
            return
        last[0] = now
        _write_json(d / "status.json", {**status, "state": "running", "stage": stage, "progress": round(frac, 3)})

    try:
        result = analyze(d / VIDEO_NAME, profile, weights, max_seconds=max_seconds, progress=progress)
        save_analysis(result, d)
        _write_json(d / "status.json", {**status, "state": "done", "stage": "done", "progress": 1.0,
                                        "finished_at": time.time()})
    except Exception as e:  # noqa: BLE001 - 失敗内容を status.json に残してユーザーに見せる
        _write_json(d / "status.json", {**status, "state": "failed", "error": f"{type(e).__name__}: {e}",
                                        "traceback": traceback.format_exc()[-2000:]})


class JobStore:
    def __init__(self, root: Path, weights: str = "yolov8s.pt", max_seconds: float | None = None,
                 extra_demo_dirs: list[Path] = ()):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "demo").mkdir(parents=True, exist_ok=True)
        # storage/demo と、リポジトリに同梱した読み取り専用のデモ(映像なし、解析結果のみ)
        self.demo_dirs = [self.root / "demo"] + [Path(d) for d in extra_demo_dirs if Path(d).is_dir()]
        self.weights = weights
        self.max_seconds = max_seconds
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._recover()

    # --- 参照 ---
    def job_dir(self, job_id: str) -> Path | None:
        for base in (self.jobs_dir, *self.demo_dirs):
            d = base / job_id
            if d.is_dir() and d.resolve().parent == base.resolve():
                return d
        return None

    def status(self, job_id: str) -> dict | None:
        d = self.job_dir(job_id)
        if d is None:
            return None
        if (d / "status.json").exists():
            return json.loads((d / "status.json").read_text(encoding="utf-8"))
        if (d / "analysis.json").exists():   # デモ
            return {"id": job_id, "state": "done", "progress": 1.0, "demo": True, "name": job_id}
        return None

    def list(self) -> list[dict]:
        names = sorted({d.name for base in self.demo_dirs for d in base.iterdir() if d.is_dir()})
        out = [s for n in names if (s := self.status(n))]
        jobs = [s for d in self.jobs_dir.iterdir() if d.is_dir() and (s := self.status(d.name))]
        return out + sorted(jobs, key=lambda s: -s.get("created_at", 0))

    # --- 投入と実行 ---
    def create(self, filename: str, profile: str, write_video) -> dict:
        job_id = uuid.uuid4().hex[:12]
        d = self.jobs_dir / job_id
        d.mkdir()
        write_video(d / VIDEO_NAME)
        status = {"id": job_id, "name": filename, "profile": profile, "state": "queued", "stage": "queued",
                  "progress": 0.0, "created_at": time.time()}
        _write_json(d / "status.json", status)
        self._queue.put(job_id)
        self._ensure_worker()
        return status

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()

    def _loop(self):
        ctx = mp.get_context("spawn")
        while True:
            job_id = self._queue.get()
            d = self.jobs_dir / job_id
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
            p = ctx.Process(target=_run_job, args=(str(d), st["profile"], self.weights, self.max_seconds))
            p.start()
            p.join()
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
            if st["state"] not in ("done", "failed"):   # 子プロセスが異常終了した
                _write_json(d / "status.json", {**st, "state": "failed", "error": f"worker exited ({p.exitcode})"})

    def _recover(self):
        """サーバー再起動時、実行途中だったジョブを失敗扱いにし、待ちのジョブを積み直す。"""
        for d in self.jobs_dir.iterdir():
            s = self.status(d.name) if d.is_dir() else None
            if not s:
                continue
            if s["state"] == "running":
                _write_json(d / "status.json", {**s, "state": "failed", "error": "server restarted"})
            elif s["state"] == "queued":
                self._queue.put(d.name)
                self._ensure_worker()
