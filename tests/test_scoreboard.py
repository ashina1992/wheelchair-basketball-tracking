from wbt.scoreboard import (_stable_clock, _stable_scores, clock_candidates, format_clock, parse_clock,
                            score_changes, smooth_scoreboard)


def test_parse_clock():
    assert parse_clock("7:20") == 440
    assert parse_clock("59.9") == 59.9
    assert parse_clock("71:14") is None
    assert parse_clock("7:75") is None


def test_clock_candidates_swaps_1_and_7():
    assert clock_candidates("1:38") == [(98, 1.0), (458, 0.5)]
    assert clock_candidates("") == []


def test_stable_scores_ignores_single_misread_and_decrease():
    # 4 → 6 の途中で 2 を1回だけ誤読しても、確定値は下がらない
    raw = [4, 4, 4, 2, 6, 6, 6, None, 6]
    assert _stable_scores(raw) == [4, 4, 4, 4, 6, 6, 6, 6, 6]


def test_stable_scores_backfills_first_reading():
    raw = [2, 2, 2, 4, 4, 4]
    assert _stable_scores(raw) == [2, 2, 2, 4, 4, 4]


def test_stable_scores_rejects_implausible_jump():
    raw = [2, 2, 9, 9, 2]
    assert _stable_scores(raw) == [2, 2, 2, 2, 2]


def test_stable_clock_prefers_candidate_consistent_with_time():
    ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    reads = ["7:50", "7:50", "1:49", "7:49", "1:48", "7:48"]
    out = _stable_clock(ts, [clock_candidates(s) for s in reads])
    assert out == [470, 470, 469, 469, 468, 468]


def test_score_changes():
    readings = [{"t": t, "score_a": a, "score_b": b, "clock_candidates": []}
                for t, a, b in [(0, 2, 4), (1, 2, 4), (2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 7), (6, 4, 7)]]
    ch = score_changes(smooth_scoreboard(readings))
    assert [(c["t"], c["team"], c["points"]) for c in ch] == [(2, "a", 2), (5, "b", 3)]


def test_format_clock():
    assert format_clock(440) == "7:20"
    assert format_clock(59.9) == "59.9"
    assert format_clock(None) is None
