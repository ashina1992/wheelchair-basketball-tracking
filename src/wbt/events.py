"""ボール保持の推定とイベント判定(ルールベース)。

入力はパイプラインが作るフレーム列(`analysis.json` の frames と同じ形)と、スコアボードの得点変化。
判定ルールは docs/design.md 2.2 を参照。
"""
from dataclasses import dataclass

import numpy as np

from . import court as C

# 判定パラメータ(時間は秒、距離はキーフレーム基準の画素。ズーム率で補正する)
HOLD_MIN_FRAMES = 3        # 保持とみなす最短の連続フレーム数
HOLD_GAP_FRAMES = 3        # 保持中にボールを見失っても保持が続いているとみなすフレーム数
RIM_NEAR_PX = 70           # リングの近くとみなす距離
SHOT_AFTER_HOLD = 2.5      # 保持が解けてからリング付近に達するまでの最大時間
SHOT_DEDUP = 2.0           # 同じシュートとみなす時間幅
SCORE_DELAY = (-1.0, 6.0)  # シュートから得点表示が変わるまでの時間の範囲
REBOUND_WINDOW = 6.0
ASSIST_WINDOW = 3.0
TURNOVER_GAP = 3.0


@dataclass
class Stint:
    """1人の選手がボールを保持していた区間。"""
    player: int
    team: str
    start: int      # フレーム番号(frames のインデックス)
    end: int
    segment: int


def ball_holder(fr: dict) -> tuple[int, str] | None:
    """ボールを持っている選手 (track_id, team) を返す。ボールが選手の上半身〜膝の範囲にあれば保持とみなす。"""
    if not fr["wide"] or fr["ball"] is None:
        return None
    bx, by = fr["ball"]["xy"]
    best, best_d = None, 1e9
    for p in fr["players"]:
        if p.get("team") not in ("a", "b"):
            continue
        x1, y1, x2, y2 = p["box"]
        w, h = x2 - x1, y2 - y1
        if x1 - 0.2 * w <= bx <= x2 + 0.2 * w and y1 - 0.1 * h <= by <= y1 + 0.8 * h:
            d = np.hypot(bx - (x1 + x2) / 2, by - (y1 + 0.4 * h))
            if d < best_d:
                best, best_d = (p["id"], p["team"]), d
    return best


def find_stints(frames: list[dict]) -> list[Stint]:
    stints: list[Stint] = []
    cur: Stint | None = None
    miss = 0
    for i, fr in enumerate(frames):
        h = ball_holder(fr)
        if cur is not None and (fr["segment"] != cur.segment or (h is not None and h[0] != cur.player)
                                or (h is None and miss >= HOLD_GAP_FRAMES)):
            if cur.end - cur.start + 1 >= HOLD_MIN_FRAMES:
                stints.append(cur)
            cur, miss = None, 0
        if h is None:
            miss += 1
            continue
        miss = 0
        if cur is None:
            cur = Stint(h[0], h[1], i, i, fr["segment"])
        else:
            cur.end = i
    if cur is not None and cur.end - cur.start + 1 >= HOLD_MIN_FRAMES:
        stints.append(cur)
    return stints


def _player_court(fr: dict, pid: int):
    for p in fr["players"]:
        if p["id"] == pid:
            return p.get("court")
    return None


def _shot_zone(xy, side: int | None) -> tuple[int | None, float | None]:
    """コート座標から (2 or 3, リングまでの距離) を返す。side はどちらのゴールか(-1/+1)。"""
    if xy is None:
        return None, None
    x, y = xy
    if side is None:
        side = 1 if x > 0 else -1
    d = float(np.hypot(x - C.basket_x(side), y))
    corner = abs(y) > C.THREE_SIDE and abs(x) > abs(C.basket_x(side)) - 1.4
    return (3 if d > C.THREE_R or corner else 2), round(d, 2)


def detect_shots(frames: list[dict], stints: list[Stint]) -> list[dict]:
    """ボールがリング付近に達した時点をシュート試投とする。直前に保持していた選手をシューターとする。"""
    shots = []
    last_t = -1e9
    for i, fr in enumerate(frames):
        if not fr["wide"] or fr["ball"] is None or not fr.get("rims"):
            continue
        bx, by = fr["ball"]["xy"]
        r = RIM_NEAR_PX * fr.get("zoom", 1.0)
        near = [k for k, (rx, ry) in enumerate(fr["rims"]) if np.hypot(bx - rx, by - ry) < r and by < ry + 0.3 * r]
        if not near or fr["t"] - last_t < SHOT_DEDUP:
            continue
        if ball_holder(fr) is not None:   # 持ったままリング付近にいるだけ(ゴール下の保持)は除く
            continue
        prev = [s for s in stints if s.end < i and s.segment == fr["segment"]
                and fr["t"] - frames[s.end]["t"] <= SHOT_AFTER_HOLD]
        shooter = prev[-1] if prev else None
        loc = _player_court(frames[shooter.end], shooter.player) if shooter else None
        rim_side = fr.get("rim_sides", [None] * len(fr["rims"]))[near[0]]
        zone, dist = _shot_zone(loc, rim_side)
        shots.append({"t": fr["t"], "frame": i, "team": shooter.team if shooter else None,
                      "player": shooter.player if shooter else None, "stint": shooter,
                      "court": loc, "zone": zone, "distance": dist, "side": rim_side})
        last_t = fr["t"]
    return shots


def build_events(frames: list[dict], score_changes: list[dict]) -> tuple[list[dict], list[Stint]]:
    stints = find_stints(frames)
    shots = detect_shots(frames, stints)
    events = []
    used_changes = set()

    # シュート試投と得点変化の対応付け
    for s in shots:
        match = None
        for k, ch in enumerate(score_changes):
            if k in used_changes or ch["points"] < 2:
                continue
            if s["team"] not in (None, ch["team"]):
                continue
            if SCORE_DELAY[0] <= ch["t"] - s["t"] <= SCORE_DELAY[1]:
                match = k
                break
        base = {"t": s["t"], "player": s["player"], "court": s["court"], "distance": s["distance"]}
        if match is not None:
            used_changes.add(match)
            ch = score_changes[match]
            events.append({**base, "type": "made_shot", "team": ch["team"], "points": ch["points"],
                           "source": "vision+scoreboard", "confidence": 0.9, "stint": s["stint"]})
        elif s["team"] is not None:
            events.append({**base, "type": "missed_shot", "team": s["team"], "points": s["zone"] or 2,
                           "source": "vision", "confidence": 0.5, "stint": s["stint"], "frame": s["frame"]})

    # 映像側で試投を捉えられなかった得点(フリースロー、見逃したシュート)
    for k, ch in enumerate(score_changes):
        if k in used_changes:
            continue
        if ch["points"] == 1:
            events.append({"t": ch["t"], "type": "free_throw", "team": ch["team"], "points": 1, "player": None,
                           "court": None, "source": "scoreboard", "confidence": 0.9, "stint": None})
            continue
        prev = [s for s in stints if s.team == ch["team"] and 0 <= ch["t"] - frames[s.end]["t"] <= 8.0]
        st = prev[-1] if prev else None
        loc = _player_court(frames[st.end], st.player) if st else None
        events.append({"t": ch["t"] - 1.0, "type": "made_shot", "team": ch["team"], "points": ch["points"],
                       "player": st.player if st else None, "court": loc, "distance": None,
                       "source": "scoreboard", "confidence": 0.8, "stint": st})

    # リバウンド: シュート失敗後、最初に保持したチーム
    for e in [e for e in events if e["type"] == "missed_shot"]:
        after = [s for s in stints if s.start > e["frame"] and frames[s.start]["t"] - e["t"] <= REBOUND_WINDOW]
        if after:
            s = after[0]
            events.append({"t": frames[s.start]["t"], "type": "rebound", "team": s.team, "player": s.player,
                           "offensive": s.team == e["team"], "court": _player_court(frames[s.start], s.player),
                           "source": "vision", "confidence": 0.5, "stint": s})

    # アシスト: 成功シュートの直前に同じチームの別の選手から受けていた
    for e in [e for e in events if e["type"] == "made_shot" and e["stint"] is not None]:
        st = e["stint"]
        prev = [s for s in stints if s.end < st.start]
        if not prev:
            continue
        p = prev[-1]
        gap = frames[st.start]["t"] - frames[p.end]["t"]
        if p.team == st.team and p.player != st.player and p.segment == st.segment and gap <= ASSIST_WINDOW:
            events.append({"t": frames[p.end]["t"], "type": "assist", "team": p.team, "player": p.player,
                           "court": _player_court(frames[p.end], p.player),
                           "source": "vision", "confidence": 0.4, "stint": p})

    # ターンオーバー: シュートを挟まずに保持チームが相手に変わった
    shot_frames = [s["frame"] for s in shots]
    change_ts = [c["t"] for c in score_changes]
    for p, q in zip(stints, stints[1:]):
        if p.team == q.team or p.segment != q.segment:
            continue
        t0, t1 = frames[p.end]["t"], frames[q.start]["t"]
        if t1 - t0 > TURNOVER_GAP:
            continue
        if any(p.end <= f <= q.start for f in shot_frames):
            continue
        if any(t0 - 2 <= t <= t1 + 3 for t in change_ts):
            continue
        events.append({"t": t0, "type": "turnover", "team": p.team, "player": p.player,
                       "court": _player_court(frames[p.end], p.player), "source": "vision",
                       "confidence": 0.4, "stint": p, "recovered_by": {"team": q.team, "player": q.player}})

    events.sort(key=lambda e: e["t"])
    for n, e in enumerate(events, 1):
        e["id"] = n
    return events, stints


def compute_stats(events: list[dict], team_ids=("a", "b")) -> dict:
    keys = ["PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "OREB", "DREB", "AST", "TOV"]
    st = {t: dict.fromkeys(keys, 0) for t in team_ids}
    for e in events:
        s = st.get(e["team"])
        if s is None:
            continue
        typ = e["type"]
        if typ == "made_shot":
            s["PTS"] += e["points"]
            s["FGM"] += 1
            s["FGA"] += 1
            if e["points"] == 3:
                s["3PM"] += 1
                s["3PA"] += 1
        elif typ == "missed_shot":
            s["FGA"] += 1
            if e["points"] == 3:
                s["3PA"] += 1
        elif typ == "free_throw":
            s["PTS"] += 1
            s["FTM"] += 1
        elif typ == "rebound":
            s["OREB" if e["offensive"] else "DREB"] += 1
        elif typ == "assist":
            s["AST"] += 1
        elif typ == "turnover":
            s["TOV"] += 1
    return st
