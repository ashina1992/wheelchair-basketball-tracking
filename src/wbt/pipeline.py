"""解析パイプライン: 動画 → analysis.json。"""
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import __version__
from .camera import CameraTracker
from .court import court_outline
from .detect import Detector
from .events import build_events, compute_stats
from .profiles import Profile, load_profile
from .scoreboard import format_clock, read_scoreboard, score_changes, shot_clock_resets, smooth_scoreboard
from .spacing import possessions
from .teams import assign_teams, floor_color, torso_color
from .tracking import BallTracker, PlayerTracker, interpolate_ball
from .video import detect_content_box

Progress = Callable[[str, float], None]

CUT_HIST_CORR = 0.6      # 直前フレームとの色ヒストグラム相関がこれ未満ならショット切替
COURT_MARGIN = 1.0       # コート外周からこの距離(m)以内の人物だけを選手候補とする(ベンチ・観客の除外)
OCR_INTERVAL = 0.5       # スコアボードを読む間隔(秒)
COLOR_EVERY = 3          # チーム識別用の色をトラックごとに何フレームおきに採るか


def _hist(frame):
    small = cv2.resize(frame, (160, 90))
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _inside(pt, poly) -> bool:
    return cv2.pointPolygonTest(np.float32(poly), (float(pt[0]), float(pt[1])), False) >= 0


def analyze(video: Path, profile: Profile | str, weights: str = "yolov8s.pt", target_fps: float = 10.0,
            max_seconds: float | None = None, progress: Progress | None = None) -> dict:
    progress = progress or (lambda stage, frac: None)
    prof = load_profile(profile) if isinstance(profile, str) else profile
    t_start = time.time()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / target_fps))
    crop = detect_content_box(cap)
    if max_seconds:
        total = min(total, int(max_seconds * fps))
    progress("loading", 0.0)

    detector = Detector(weights)
    cam = CameraTracker(prof.keyframes)
    ptrack, btrack = PlayerTracker(), BallTracker()
    outline = court_outline(COURT_MARGIN)
    rois = prof.scoreboard
    exclude_static = [tuple(rois["overlay"])] if "overlay" in rois else []

    frames, readings = [], []
    track_obs: dict[int, list] = {}   # トラックID → [(胴体の色, 箱の縦横比)]
    prev_hist, prev_wide, segment = None, False, 0
    next_ocr = 0.0

    # フレーム番号でのシークは可変フレームレートの動画(画面録画)で数フレームずれるため、常に先頭から順に読む。
    # 時刻もフレーム番号/平均fps ではなく、動画に記録されたタイムスタンプを使う(長い動画ほど差が開くため)
    for idx in range(0, total, stride):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if max_seconds and t > max_seconds:
            break
        x0, y0, x1, y1 = crop
        frame = frame[y0:y1, x0:x1]
        if frame.shape[1] != prof.frame_size[0]:
            frame = cv2.resize(frame, tuple(prof.frame_size))

        det = detector(frame)
        h = _hist(frame)
        cut = prev_hist is not None and cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL) < CUT_HIST_CORR
        prev_hist = h
        st = cam.update(frame, exclude=[tuple(b) for b in det.players.xyxy] + exclude_static, cut=cut)
        wide = st.valid
        if cut or wide != prev_wide:
            segment += 1
            ptrack.reset()
            btrack.reset()
        prev_wide = wide

        rec = {"t": round(t, 3), "frame": idx, "wide": wide, "segment": segment, "players": [], "ball": None}
        if wide:
            tracks = ptrack.update(det, frame)
            sample_color = len(frames) % COLOR_EVERY == 0
            floor = floor_color(frame, cam.court_to_img(st, court_outline())) if sample_color else None
            if len(tracks):
                feet = np.stack([(tracks[:, 0] + tracks[:, 2]) / 2, tracks[:, 3]], axis=1)
                courts = cam.img_to_court(st, feet)
                for tr, cxy in zip(tracks, courts):
                    if not _inside(cxy, outline):
                        continue
                    tid = int(tr[4])
                    rec["players"].append({"id": tid, "box": [round(float(v), 1) for v in tr[:4]],
                                           "court": [round(float(cxy[0]), 2), round(float(cxy[1]), 2)]})
                    if sample_color:
                        aspect = (tr[3] - tr[1]) / max(tr[2] - tr[0], 1.0)
                        track_obs.setdefault(tid, []).append((torso_color(frame, tr, floor), float(aspect)))
            zoom = cam.zoom(st)
            b = btrack.update(det.balls, zoom)
            if b is not None:
                rec["ball"] = {"xy": [round(float(b[0][0]), 1), round(float(b[0][1]), 1)],
                               "conf": round(b[1], 3), "interpolated": False}
            rims, sides = cam.rims_in_frame(st)
            rec["rims"] = rims.round(1).tolist()
            rec["rim_sides"] = sides
            rec["zoom"] = round(zoom, 3)
            rec["keyframe"] = prof.keyframes[st.keyframe].name
        if t >= next_ocr:
            readings.append({"t": round(t, 3), **read_scoreboard(frame, rois, prof.digits)})
            next_ocr = t + OCR_INTERVAL
        frames.append(rec)
        progress("analyzing", idx / max(total, 1))

        for _ in range(stride - 1):
            cap.grab()
    cap.release()

    progress("events", 0.95)
    series = smooth_scoreboard(readings)
    resets = shot_clock_resets(series)
    _drop_hidden_scoreboard(frames, series)

    team_colors = {t.id: t.color for t in prof.teams if t.color}
    teams = assign_teams(track_obs, team_colors or None)
    for fr in frames:
        for p in fr["players"]:
            p["team"] = teams.get(p["id"], "other")
    interpolate_ball(frames)

    changes = score_changes(series)
    events, stints = build_events(frames, changes, series, resets)
    stats = compute_stats(events)
    poss = possessions(frames, stints)

    result = {
        "version": 1,
        "meta": {"video": Path(video).name, "fps": fps, "stride": stride, "n_frames": total,
                 "duration": round(total / fps, 2), "crop": list(crop), "frame_size": list(prof.frame_size),
                 "model": detector.weights,
                 "detector_classes": sorted(detector.available), "profile": prof.name,
                 "wbt_version": __version__, "processing_sec": round(time.time() - t_start, 1)},
        "teams": [{"id": t.id, "name": t.name, "color": list(t.color) if t.color else None} for t in prof.teams],
        "segments": _segments(frames),
        "frames": frames,
        "scoreboard": [{**r, "clock": format_clock(r["clock"])} for r in series],
        "shot_clock_resets": resets,
        "events": [_public_event(e) for e in events],
        "stats": stats,
        "possessions": poss,
    }
    progress("done", 1.0)
    return result


def _drop_hidden_scoreboard(frames: list[dict], series: list[dict]):
    """スコアボードが消えている区間(リプレイ等)は、広角ショットでも解析対象から外す。"""
    if not series:
        return
    ts = np.array([r["t"] for r in series])
    vis = np.array([r["visible"] for r in series])
    for fr in frames:
        if fr["wide"]:
            k = int(np.argmin(np.abs(ts - fr["t"])))
            lo, hi = max(0, k - 1), min(len(vis), k + 2)
            if not vis[lo:hi].any():
                fr["wide"] = False
                fr["players"], fr["ball"] = [], None


def _segments(frames: list[dict]) -> list[dict]:
    segs = []
    for fr in frames:
        typ = "wide" if fr["wide"] else "other"
        if segs and segs[-1]["type"] == typ:
            segs[-1]["end"] = fr["t"]
        else:
            segs.append({"start": fr["t"], "end": fr["t"], "type": typ})
    return segs


def _public_event(e: dict) -> dict:
    out = {k: v for k, v in e.items() if k not in ("stint", "frame")}
    out["t"] = round(out["t"], 2)
    return out


def save_analysis(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                                           encoding="utf-8")
    names = {t["id"]: t["name"] for t in result["teams"]}
    keys = list(next(iter(result["stats"].values())).keys())
    lines = ["team," + ",".join(keys)]
    lines += [f"{names.get(t, t)}," + ",".join(str(v) for v in s.values()) for t, s in result["stats"].items()]
    (out_dir / "stats.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ev = ["id,t,type,team,points,player,source,confidence"]
    ev += [f'{e["id"]},{e["t"]},{e["type"]},{names.get(e["team"], e["team"])},{e.get("points", "")},'
           f'{e.get("player") or ""},{e["source"]},{e["confidence"]}' for e in result["events"]]
    (out_dir / "events.csv").write_text("\n".join(ev) + "\n", encoding="utf-8")
