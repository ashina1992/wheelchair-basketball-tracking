"""公式のプレーバイプレー(FIBA LiveStats の data.json)から、評価用の正解イベントを作る。

公式記録の時刻は「ピリオドと試合時計(残り時間)」なので、解析結果のスコアボードの読み(試合時計)を使って
動画の時刻に対応付ける。試合時計が止まっている間(ファウル、フリースロー等)の出来事は動画上の時刻が
決まらないため、時計が動いている間に起きるイベント(フィールドゴール、リバウンド、アシスト、ターンオーバー)を
評価対象にする。
"""
from .scoreboard import parse_clock

# LiveStats の actionType → 解析結果のイベント種別
FIELD_GOALS = {"2pt": 2, "3pt": 3}
EVAL_TYPES = ("made_shot", "missed_shot", "rebound", "assist", "turnover")


def livestats_clock(s: str) -> float:
    """'07:38:00'(分:秒:1/100秒)→ 残り秒数。"""
    m, sec, cs = (int(v) for v in s.split(":"))
    return m * 60 + sec + cs / 100


def clock_series(scoreboard: list[dict]) -> list[tuple[float, float]]:
    """解析結果のスコアボードの読みから、(動画の時刻, 試合時計の残り秒数) の列を作る。"""
    out = []
    for r in scoreboard:
        c = r.get("clock")
        v = parse_clock(c) if isinstance(c, str) else c
        if v is not None:
            out.append((r["t"], float(v)))
    return out


def video_time(series: list[tuple[float, float]], clock: float, tol: float = 1.0) -> float | None:
    """試合時計が clock になった動画の時刻。動画に含まれない時刻なら None。

    最初に clock 以下になった読みと、その直前の読みの間を、時計が等速で進んだとして補間する。
    """
    if not series or clock > series[0][1] + tol or clock < series[-1][1] - tol:
        return None
    for k, (t, c) in enumerate(series):
        if c <= clock:
            if k == 0:
                return t
            pt, pc = series[k - 1]
            return round(min(t, pt + max(0.0, pc - clock)), 2)
    return None


def team_ids(livestats: dict, analysis_teams: list[dict]) -> dict[str, str]:
    """LiveStats のチーム番号("1"/"2")→ 解析結果のチーム("a"/"b")。チームの略称で対応付ける。"""
    out = {}
    for tno, tm in livestats["tm"].items():
        short = (tm.get("shortName") or tm.get("name", "")).split()[0].upper()
        for t in analysis_teams:
            if t["name"].upper() == short or tm.get("name", "").upper().startswith(t["name"].upper()):
                out[tno] = t["id"]
    return out


def events_from_livestats(livestats: dict, analysis: dict, period: int) -> dict:
    """解析結果の動画に含まれる時間帯の公式イベントを、解析結果と同じ形式で返す。"""
    series = clock_series(analysis["scoreboard"])
    tmap = team_ids(livestats, analysis["teams"])
    events = []
    for a in livestats["pbp"]:
        if a.get("period") != period or a.get("periodType") != "REGULAR":
            continue
        typ, team = a.get("actionType"), tmap.get(str(a.get("tno")))
        if team is None:
            continue
        clock = livestats_clock(a["clock"])
        t = video_time(series, clock)
        if t is None:
            continue
        base = {"t": t, "team": team, "clock": a["clock"], "player": a.get("player", ""),
                "action_number": a.get("actionNumber")}
        if typ in FIELD_GOALS:
            kind = "made_shot" if a.get("success") else "missed_shot"
            events.append({**base, "type": kind, "points": FIELD_GOALS[typ]})
        elif typ == "rebound":
            events.append({**base, "type": "rebound", "offensive": a.get("subType") == "offensive"})
        elif typ == "assist":
            events.append({**base, "type": "assist"})
        elif typ == "turnover":
            events.append({**base, "type": "turnover", "sub_type": a.get("subType")})
    events.sort(key=lambda e: e["t"])
    return {"period": period, "clock_window": [series[0][1], series[-1][1]] if series else None,
            "team_map": tmap, "events": events}
