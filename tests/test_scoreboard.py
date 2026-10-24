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


def test_stable_shot_clock_rejects_misreads_and_keeps_resets():
    # Japan v Poland の実際の読みに近い列: 29(20 の誤読)、19(15 の誤読)、14 へのリセット直前の 24(誤読)
    ts = [9.5, 10.1, 10.6, 11.1, 14.9, 15.4, 15.9, 16.5, 30.3, 30.8, 31.3, 31.9, 32.4]
    raw = [20, None, None, 19, 15, 15, 19, 14, 5, 5, 24, 14, 13]
    from wbt.scoreboard import _stable_shot_clock, shot_clock_resets
    out = _stable_shot_clock(ts, raw)
    assert out == [20, None, None, 19, 15, 15, None, 14, 5, 5, None, 14, 13]
    resets = shot_clock_resets([{"t": t, "shot_clock": v} for t, v in zip(ts, out)])
    assert resets == [{"t": 31.35, "to": 14}]


def test_shot_clock_reset_to_24_detected_from_23():
    from wbt.scoreboard import shot_clock_resets
    series = [{"t": 0.0, "shot_clock": 6}, {"t": 0.5, "shot_clock": None}, {"t": 1.0, "shot_clock": 23}]
    assert shot_clock_resets(series) == [{"t": 0.5, "to": 24}]


def test_transitional_reset_14_then_24_is_merged():
    """リングに当たった直後に一瞬 14 が表示されてから 24 になる場合は、24 へのリセット1回とみなす。"""
    from wbt.scoreboard import shot_clock_resets
    series = [{"t": 0.0, "shot_clock": 6}, {"t": 0.5, "shot_clock": 14}, {"t": 1.0, "shot_clock": 14},
              {"t": 1.5, "shot_clock": 24}, {"t": 2.0, "shot_clock": 24}, {"t": 6.0, "shot_clock": 20},
              {"t": 6.5, "shot_clock": 14}]
    assert shot_clock_resets(series) == [{"t": 0.25, "to": 24}]


def test_format_clock():
    assert format_clock(440) == "7:20"
    assert format_clock(59.9) == "59.9"
    assert format_clock(None) is None
