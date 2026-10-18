from wbt.events import build_events, compute_stats, find_stints
from wbt.spacing import frame_spacing, possessions

RIM = [1300.0, 300.0]
FPS = 10.0


def player(pid, team, x, y=600):
    return {"id": pid, "team": team, "box": [x, y, x + 80, y + 100], "court": [5.0 + pid, 0.0]}


def frame(i, ball, players, segment=1):
    return {"t": i / FPS, "wide": True, "segment": segment, "players": players,
            "ball": {"xy": ball, "interpolated": False} if ball else None,
            "rims": [RIM], "rim_sides": [1], "zoom": 1.0}


def held_by(p):
    x1, y1, x2, _ = p["box"]
    return [(x1 + x2) / 2, y1 + 30]


def build(script):
    """script: [(ball_owner_or_xy, players)] を frames にする。"""
    return [frame(i, held_by(b) if isinstance(b, dict) else b, ps) for i, (b, ps) in enumerate(script)]


def test_pass_shot_made_gives_assist():
    a1, a2, b1 = player(1, "a", 400), player(2, "a", 800), player(11, "b", 1000, 800)
    ps = [a1, a2, b1]
    script = [(a1, ps)] * 5 + [([600, 500], ps)] * 2 + [(a2, ps)] * 5 \
        + [([1000, 450], ps), ([1150, 350], ps), ([1290, 290], ps)] + [(None, ps)] * 10
    frames = build(script)
    changes = [{"t": 1.8, "team": "a", "points": 2, "score": 4}]
    events, stints = build_events(frames, changes)
    types = [(e["type"], e["team"], e.get("player")) for e in events]
    assert ("made_shot", "a", 2) in types
    assert ("assist", "a", 1) in types
    s = compute_stats(events)
    assert s["a"]["PTS"] == 2 and s["a"]["FGM"] == 1 and s["a"]["AST"] == 1
    assert [st.player for st in stints] == [1, 2]


def test_missed_shot_then_defensive_rebound():
    a1, b1 = player(1, "a", 800), player(11, "b", 1200, 700)
    ps = [a1, b1]
    script = [(a1, ps)] * 5 + [([1000, 450], ps), ([1290, 290], ps), ([1250, 500], ps)] + [(b1, ps)] * 5 \
        + [(None, ps)] * 70
    events, _ = build_events(build(script), [])
    types = [(e["type"], e["team"]) for e in events]
    assert ("missed_shot", "a") in types
    reb = [e for e in events if e["type"] == "rebound"]
    assert reb and reb[0]["team"] == "b" and reb[0]["offensive"] is False
    s = compute_stats(events)
    assert s["a"]["FGA"] == 1 and s["a"]["FGM"] == 0 and s["b"]["DREB"] == 1


def test_steal_is_turnover():
    a1, b1 = player(1, "a", 400), player(11, "b", 500)
    ps = [a1, b1]
    script = [(a1, ps)] * 5 + [([480, 640], ps)] + [(b1, ps)] * 12
    events, _ = build_events(build(script), [])
    to = [e for e in events if e["type"] == "turnover"]
    assert len(to) == 1 and to[0]["team"] == "a" and to[0]["recovered_by"]["team"] == "b"


def test_brief_touch_by_defender_is_not_turnover():
    """守備側が0.5秒だけ保持者と判定されても(密着した場面の揺れ)、ターンオーバーにしない。"""
    a1, b1 = player(1, "a", 400), player(11, "b", 500)
    ps = [a1, b1]
    script = [(a1, ps)] * 8 + [(b1, ps)] * 5 + [(a1, ps)] * 8
    events, _ = build_events(build(script), [])
    assert not [e for e in events if e["type"] == "turnover"]


def clock(ts_values):
    return [{"t": t, "shot_clock": v} for t, v in ts_values]


def test_turnover_requires_shot_clock_reset_when_clock_is_visible():
    a1, b1 = player(1, "a", 400), player(11, "b", 500)
    ps = [a1, b1]
    frames = build([(a1, ps)] * 5 + [([480, 640], ps)] + [(b1, ps)] * 15)
    running = clock([(t / 2, 20 - t / 2) for t in range(0, 6)])          # 止まらずに減り続けている
    events, _ = build_events(frames, [], running, [])
    assert not [e for e in events if e["type"] == "turnover"]
    reset = clock([(0.0, 20), (0.5, 19.5), (1.0, 24), (1.5, 23.5), (2.0, 23)])
    events, _ = build_events(frames, [], reset, [{"t": 0.75, "to": 24}])
    to = [e for e in events if e["type"] == "turnover"]
    assert len(to) == 1 and to[0]["source"] == "vision+shot_clock"


def _shot_frames():
    a1, b1 = player(1, "a", 800), player(11, "b", 1200, 700)
    ps = [a1, b1]
    return build([(a1, ps)] * 30 + [([1000, 450], ps), ([1290, 290], ps), ([1250, 500], ps)] + [(None, ps)] * 50)


def test_rebound_from_shot_clock_reset():
    frames = _shot_frames()   # 3.1 秒にシュート
    sb = clock([(t / 2, 10) for t in range(0, 14)])
    oreb, _ = build_events(frames, [], sb, [{"t": 4.0, "to": 14}])
    r = [e for e in oreb if e["type"] == "rebound"]
    assert len(r) == 1 and r[0]["team"] == "a" and r[0]["offensive"]
    dreb, _ = build_events(frames, [], sb, [{"t": 4.0, "to": 24}])
    r = [e for e in dreb if e["type"] == "rebound"]
    assert len(r) == 1 and r[0]["team"] == "b" and not r[0]["offensive"]


def test_shot_without_reset_is_discarded_when_clock_is_visible():
    """リングに当たればショットクロックは必ずリセットされる。リセットがなければシュートではなかったとみなす。"""
    frames = _shot_frames()
    sb = clock([(t / 2, 20 - t / 2) for t in range(0, 20)])
    events, _ = build_events(frames, [], sb, [])
    assert not [e for e in events if e["type"] in ("missed_shot", "rebound")]


def test_oreb_reset_adds_missed_shot_not_seen_on_video():
    a1 = player(1, "a", 400)
    frames = build([(a1, [a1])] * 60)
    sb = clock([(t / 2, 10) for t in range(0, 14)])
    events, _ = build_events(frames, [], sb, [{"t": 5.5, "to": 14}])
    kinds = [(e["type"], e["team"], e["source"]) for e in events]
    assert ("missed_shot", "a", "shot_clock") in kinds and ("rebound", "a", "vision+shot_clock") in kinds


def test_possession_change_after_score_is_not_turnover():
    a1, b1 = player(1, "a", 400), player(11, "b", 500)
    ps = [a1, b1]
    script = [(a1, ps)] * 5 + [([480, 640], ps)] + [(b1, ps)] * 5
    events, _ = build_events(build(script), [{"t": 0.6, "team": "a", "points": 1, "score": 3}])
    assert not [e for e in events if e["type"] == "turnover"]
    assert [e["type"] for e in events] == ["free_throw"]


def test_short_touch_is_not_possession():
    a1 = player(1, "a", 400)
    frames = build([(a1, [a1])] * 2 + [(None, [a1])] * 5)
    assert find_stints(frames) == []


def test_spacing_metrics():
    sp = frame_spacing([[0, 0], [4, 0], [4, 3], [0, 3]])
    assert abs(sp["hull_area_m2"] - 12.0) < 1e-6
    assert frame_spacing([[0, 0], [1, 1]]) is None


def test_half_court_possession():
    team = [player(i, "a", 300 + 100 * i) for i in range(1, 5)]
    for p, xy in zip(team, [[6.0, -5.0], [11.0, -3.0], [12.0, 4.0], [7.5, 5.0]]):
        p["court"] = xy
    frames = build([(team[0], team)] * 40)
    _, stints = build_events(frames, [])
    poss = possessions(frames, stints)
    assert len(poss) == 1 and poss[0]["half_court"] and poss[0]["attack_side"] == 1
    assert poss[0]["spacing"]["hull_area_m2"] > 0
