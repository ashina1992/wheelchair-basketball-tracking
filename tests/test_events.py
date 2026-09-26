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
    script = [(a1, ps)] * 5 + [([480, 640], ps)] + [(b1, ps)] * 5
    events, _ = build_events(build(script), [])
    to = [e for e in events if e["type"] == "turnover"]
    assert len(to) == 1 and to[0]["team"] == "a" and to[0]["recovered_by"]["team"] == "b"


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
