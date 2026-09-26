import json

import pytest
from fastapi.testclient import TestClient

from wbt.api import jobs as jobs_mod
from wbt.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod.JobStore, "_ensure_worker", lambda self: None)   # 解析は実行しない
    demo = tmp_path / "demo" / "sample"
    demo.mkdir(parents=True)
    (demo / "analysis.json").write_text(json.dumps({"version": 1, "events": []}))
    return TestClient(create_app(tmp_path)), tmp_path


def test_importing_app_module_has_no_side_effects():
    """モジュールの読み込みでアプリ(=起動時の復旧処理)が作られないこと。

    以前はモジュール読み込み時に create_app() が走り、テストを実行しただけで
    動作中サーバーの解析ジョブが「server restarted」で失敗扱いになっていた。
    """
    import wbt.api.app as mod
    assert not hasattr(mod, "app")


def test_profiles(client):
    c, _ = client
    assert "ottawa2026" in c.get("/api/profiles").json()["profiles"]


def test_demo_is_listed_and_readable(client):
    c, _ = client
    jobs = {j["id"]: j for j in c.get("/api/jobs").json()["jobs"]}
    assert jobs["sample"]["state"] == "done" and jobs["sample"]["demo"]
    assert "jpn_pol" in jobs   # リポジトリ同梱のデモ(demo/)も一覧に出る
    assert c.get("/api/jobs/sample/analysis").json()["version"] == 1


def test_upload_creates_queued_job(client):
    c, root = client
    r = c.post("/api/jobs", files={"video": ("game.mp4", b"\x00" * 100, "video/mp4")}, data={"profile": "ottawa2026"})
    assert r.status_code == 201
    job = r.json()
    assert job["state"] == "queued" and job["name"] == "game.mp4"
    assert (root / "jobs" / job["id"] / "video.mp4").stat().st_size == 100
    assert c.get(f"/api/jobs/{job['id']}").json()["state"] == "queued"
    assert c.get(f"/api/jobs/{job['id']}/analysis").status_code == 409


def test_upload_validation(client):
    c, _ = client
    bad_ext = c.post("/api/jobs", files={"video": ("x.txt", b"a", "text/plain")}, data={"profile": "ottawa2026"})
    assert bad_ext.status_code == 400
    bad_profile = c.post("/api/jobs", files={"video": ("x.mp4", b"a", "video/mp4")}, data={"profile": "nope"})
    assert bad_profile.status_code == 400


def test_unknown_job_and_path_traversal(client):
    c, _ = client
    assert c.get("/api/jobs/does-not-exist").status_code == 404
    assert c.get("/api/jobs/..%2Fdemo/analysis").status_code == 404


def test_video_range_request(client):
    c, root = client
    (root / "demo" / "sample" / "video.mp4").write_bytes(bytes(range(256)) * 4)
    r = c.get("/api/jobs/sample/video", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == bytes(range(10))
