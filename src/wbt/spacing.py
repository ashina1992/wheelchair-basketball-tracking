"""攻撃側のスペーシング指標(コート座標の凸包面積・選手間の平均距離)。"""
from itertools import combinations

import cv2
import numpy as np

MERGE_GAP = 5.0          # 同じチームの保持区間をひと続きの攻撃とみなす間隔(秒)
HALF_COURT_RATIO = 0.7   # ハーフコートオフェンスとみなす、全員が同じハーフにいるフレームの割合
MIN_PLAYERS = 3


def frame_spacing(points) -> dict | None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < MIN_PLAYERS:
        return None
    hull = cv2.convexHull(np.float32(points))
    dists = [np.linalg.norm(a - b) for a, b in combinations(points, 2)]
    return {"hull_area_m2": float(cv2.contourArea(hull)), "mean_pair_dist_m": float(np.mean(dists))}


def possessions(frames: list[dict], stints) -> list[dict]:
    """保持区間をチームごとの攻撃区間にまとめ、ハーフコートかどうかとスペーシングを付ける。"""
    groups = []
    for s in stints:
        if groups and groups[-1]["team"] == s.team and groups[-1]["segment"] == s.segment \
                and frames[s.start]["t"] - frames[groups[-1]["end"]]["t"] <= MERGE_GAP:
            groups[-1]["end"] = s.end
        else:
            groups.append({"team": s.team, "start": s.start, "end": s.end, "segment": s.segment})

    out = []
    for g in groups:
        rows, sides = [], []
        for fr in frames[g["start"]: g["end"] + 1]:
            pts = np.array([p["court"] for p in fr["players"] if p.get("team") == g["team"] and p.get("court")])
            if len(pts) < MIN_PLAYERS:
                continue
            sx = np.sign(pts[:, 0])
            sides.append(int(sx[0]) if np.all(sx == sx[0]) and np.all(np.abs(pts[:, 0]) > 0.5) else 0)
            sp = frame_spacing(pts)
            if sp:
                rows.append(sp)
        t0, t1 = frames[g["start"]]["t"], frames[g["end"]]["t"]
        side = max(set(sides), key=sides.count) if sides else 0
        half = bool(sides) and side != 0 and sides.count(side) / len(sides) >= HALF_COURT_RATIO
        out.append({
            "start": round(t0, 2), "end": round(t1, 2), "team": g["team"], "half_court": half,
            "attack_side": side if half else None,
            "spacing": {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]} if rows else None,
        })
    return out
