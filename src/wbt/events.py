"""ボール保持の推定とイベント判定(ルールベース)。

入力はパイプラインが作るフレーム列(`analysis.json` の frames と同じ形)と、スコアボードの得点変化・ショットクロック。
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
TEAM_MIN_HOLD = 1.0       # 保持チームの交代とみなすのに必要な、新しいチームの連続保持時間


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


def _team_around(stints: list[Stint], frames: list[dict], t0: float, t1: float, min_hold: float = 0.5) -> str | None:
    """[t0, t1] の間に最も長くボールを保持していたチーム(合計 min_hold 秒未満なら None)。"""
    hold: dict[str, float] = {}
    for s in stints:
        a, b = max(frames[s.start]["t"], t0), min(frames[s.end]["t"], t1)
        if b > a:
            hold[s.team] = hold.get(s.team, 0.0) + (b - a)
    if not hold:
        return None
    team, dur = max(hold.items(), key=lambda kv: kv[1])
    return team if dur >= min_hold else None


def _shot_clock_seen(scoreboard: list[dict] | None, t0: float, t1: float) -> bool:
    """[t0, t1] にショットクロックの有効な読みがあるか(=ショットクロックで判定できる区間か)。"""
    return bool(scoreboard) and any(r.get("shot_clock") is not None and t0 <= r["t"] <= t1 for r in scoreboard)


def _other(team: str) -> str:
    return "b" if team == "a" else "a"


def build_events(frames: list[dict], score_changes: list[dict], scoreboard: list[dict] | None = None,
                 resets: list[dict] | None = None) -> tuple[list[dict], list[Stint]]:
    """イベントを判定する。

    scoreboard(ショットクロックの読みを含む系列)と resets(ショットクロックのリセット)があれば、
    保持の交代をショットクロックで裏付ける。映像だけでは、攻守の選手が密着した場面で保持者の判定が
    入れ替わりやすく、ターンオーバーを過検出する(Japan v Poland で確認)。
    - ターンオーバー: 24 秒へのリセットがあるときだけ確定する
    - リバウンド: シュート失敗後に 14 秒へ戻ればオフェンス、24 秒へ戻ればディフェンス(FIBA ルール)
    - シュートのチーム: 直前に保持していた選手ではなく、その攻撃中に長く保持していたチーム
    ショットクロックが読めない区間では、映像だけのルールで判定する。
    """
    resets = resets or []
    stints = find_stints(frames)
    shots = detect_shots(frames, stints)
    events = []
    used_changes = set()

    # シュートのチームを、その攻撃中に最も長く保持していたチームで補正する(保持者の判定は守備側に揺れやすい)
    for s in shots:
        if _shot_clock_seen(scoreboard, s["t"] - 8, s["t"] + 1):
            team = _team_around(stints, frames, s["t"] - 8, s["t"])
            if team is not None and team != s["team"]:
                s["team"], s["player"], s["court"], s["stint"] = team, None, None, None

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

    # 14 秒へのリセットで、映像では捉えられなかったシュート失敗+オフェンスリバウンドを補う
    for r in [r for r in resets if r["to"] == 14]:
        if any(r["t"] - 8 <= e["t"] <= r["t"] for e in events if e["type"] in ("missed_shot", "made_shot")):
            continue
        team = _team_around(stints, frames, r["t"] - 8, r["t"])
        if team is not None:
            events.append({"t": r["t"] - 1.0, "type": "missed_shot", "team": team, "points": 2, "player": None,
                           "court": None, "distance": None, "source": "shot_clock", "confidence": 0.5,
                           "stint": None, "frame": _frame_at(frames, r["t"] - 1.0)})

    # リバウンド
    for e in sorted([e for e in events if e["type"] == "missed_shot"], key=lambda e: e["t"]):
        if _shot_clock_seen(scoreboard, e["t"], e["t"] + REBOUND_WINDOW + 2):
            rs = [r for r in resets if e["t"] - 0.5 < r["t"] <= e["t"] + REBOUND_WINDOW + 2]
            if not rs:
                # リングに当たったシュートの後はショットクロックが必ずリセットされる。
                # リセットがないなら、シュートではなかった(リング付近へのパス等)とみなす
                e["discard"] = True
                continue
            r = rs[0]
            offensive = r["to"] == 14
            team = e["team"] if offensive else _other(e["team"])
            st = next((s for s in stints if s.team == team and frames[s.start]["t"] >= e["t"]
                       and frames[s.start]["t"] <= r["t"] + 2), None)
            events.append({"t": r["t"], "type": "rebound", "team": team, "player": st.player if st else None,
                           "offensive": offensive, "court": _player_court(frames[st.start], st.player) if st else None,
                           "source": "vision+shot_clock", "confidence": 0.7, "stint": st})
            continue
        after = [s for s in stints if s.start > e["frame"] and frames[s.start]["t"] - e["t"] <= REBOUND_WINDOW]
        if after:
            s = after[0]
            events.append({"t": frames[s.start]["t"], "type": "rebound", "team": s.team, "player": s.player,
                           "offensive": s.team == e["team"], "court": _player_court(frames[s.start], s.player),
                           "source": "vision", "confidence": 0.5, "stint": s})
    events = [e for e in events if not e.get("discard")]

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

    # ターンオーバー: シュートも得点も挟まずに、保持チームが相手に変わった
    shot_ts = [e["t"] for e in events if e["type"] in ("missed_shot", "made_shot")]
    change_ts = [c["t"] for c in score_changes]
    for k, (p, q) in enumerate(zip(stints, stints[1:])):
        if p.team == q.team or p.segment != q.segment:
            continue
        t0, t1 = frames[p.end]["t"], frames[q.start]["t"]
        if t1 - t0 > TURNOVER_GAP:
            continue
        if any(t0 - 1 <= t <= t1 for t in shot_ts) or any(t0 - 2 <= t <= t1 + 3 for t in change_ts):
            continue
        # 相手チームが1秒以上保持し続けたときだけ、保持の交代とみなす(ボールの奪い合い中の一瞬の接触を除く)
        end = q.end
        for s in stints[k + 2:]:
            if s.team != q.team:
                break
            end = s.end
        if frames[end]["t"] - t1 < TEAM_MIN_HOLD:
            continue
        source, conf = "vision", 0.4
        if _shot_clock_seen(scoreboard, t0 - 3, t1 + 4):
            if not any(r["to"] == 24 and t0 - 1 <= r["t"] <= t1 + 3 for r in resets):
                continue   # ショットクロックが続いている = 保持は変わっていない
            source, conf = "vision+shot_clock", 0.7
        events.append({"t": t0, "type": "turnover", "team": p.team, "player": p.player,
                       "court": _player_court(frames[p.end], p.player), "source": source,
                       "confidence": conf, "stint": p, "recovered_by": {"team": q.team, "player": q.player}})

    events.sort(key=lambda e: e["t"])
    for n, e in enumerate(events, 1):
        e["id"] = n
    return events, stints


def _frame_at(frames: list[dict], t: float) -> int:
    return int(np.argmin([abs(f["t"] - t) for f in frames])) if frames else 0


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
