"""REST API とフロントエンドの配信(docs/design.md 4章)。

起動:
    uvicorn wbt.api.app:create_app --factory --host 0.0.0.0 --port 8000
環境変数:
    WBT_STORAGE      ジョブ・デモの保存先(既定: ./storage)
    WBT_WEIGHTS      検出モデルの重み(既定: yolov8s.pt)
    WBT_MAX_SECONDS  アップロード動画のうち解析する最大秒数(公開環境で処理時間を抑えるため。既定: 制限なし)
    WBT_MAX_UPLOAD_MB  アップロードできる最大サイズ(既定: 500)
"""
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wbt.profiles import list_profiles

from .jobs import VIDEO_NAME, JobStore

REPO_DIR = Path(__file__).resolve().parents[3]   # パッケージは編集可能モード(pip install -e)で入れる前提
WEB_DIR = REPO_DIR / "web"
BUNDLED_DEMO_DIR = REPO_DIR / "demo"
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def create_app(storage: Path | None = None) -> FastAPI:
    storage = Path(storage or os.environ.get("WBT_STORAGE", "storage"))
    max_sec = os.environ.get("WBT_MAX_SECONDS")
    max_mb = float(os.environ.get("WBT_MAX_UPLOAD_MB", "500"))
    store = JobStore(storage, os.environ.get("WBT_WEIGHTS", "yolov8s.pt"), float(max_sec) if max_sec else None,
                     extra_demo_dirs=[BUNDLED_DEMO_DIR])
    app = FastAPI(title="Wheelchair Basketball Stats & Tracking", version="0.1.0")
    app.state.store = store

    def get_dir(job_id: str) -> Path:
        d = store.job_dir(job_id)
        if d is None:
            raise HTTPException(404, "job not found")
        return d

    @app.get("/api/profiles")
    def profiles():
        return {"profiles": list_profiles()}

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": [{k: v for k, v in s.items() if k != "traceback"} for s in store.list()]}

    @app.post("/api/jobs", status_code=201)
    def create_job(video: UploadFile = File(...), profile: str = Form(...)):
        if profile not in list_profiles():
            raise HTTPException(400, f"unknown profile: {profile}")
        if Path(video.filename or "").suffix.lower() not in ALLOWED_EXT:
            raise HTTPException(400, f"unsupported file type (allowed: {', '.join(sorted(ALLOWED_EXT))})")

        def write(dst: Path):
            with open(dst, "wb") as f:
                shutil.copyfileobj(video.file, f, length=1 << 20)
            if dst.stat().st_size > max_mb * 1024 * 1024:
                dst.unlink()
                shutil.rmtree(dst.parent, ignore_errors=True)
                raise HTTPException(413, f"file too large (max {max_mb:.0f} MB)")

        st = store.create(video.filename or VIDEO_NAME, profile, write)
        return st

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        st = store.status(job_id)
        if st is None:
            raise HTTPException(404, "job not found")
        return {k: v for k, v in st.items() if k != "traceback"}

    @app.get("/api/jobs/{job_id}/analysis")
    def job_analysis(job_id: str):
        p = get_dir(job_id) / "analysis.json"
        if not p.exists():
            raise HTTPException(409, "analysis not ready")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str):
        p = get_dir(job_id) / VIDEO_NAME
        if not p.exists():
            raise HTTPException(404, "video not available")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/{name}.csv")
    def job_csv(job_id: str, name: str):
        if name not in ("stats", "events"):
            raise HTTPException(404)
        p = get_dir(job_id) / f"{name}.csv"
        if not p.exists():
            raise HTTPException(409, "analysis not ready")
        return FileResponse(p, media_type="text/csv", filename=f"{job_id}_{name}.csv")

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
# モジュールの読み込み時にはアプリを作らない(作ると起動時の復旧処理が走り、実行中のジョブを失敗扱いにしてしまう)。
# uvicorn の --factory で create_app を呼ぶ。
