from wbt.gt import clock_series, events_from_livestats, livestats_clock, team_ids, video_time


def test_livestats_clock():
    assert livestats_clock("07:38:00") == 458
    assert livestats_clock("00:05:30") == 5.3


def test_video_time_interpolates_and_handles_stoppage():
    # 動画 10〜15 秒は時計が 7:30 で止まっていて、その後また進む
    series = [(0.0, 460.0), (5.0, 455.0), (10.0, 450.0), (15.0, 450.0), (20.0, 445.0)]
    assert video_time(series, 457.0) == 3.0
    assert video_time(series, 450.0) == 10.0          # 止まった最初の時刻
    assert video_time(series, 448.0) == 17.0          # 再開後、止まっていた分だけ遅れる
    assert video_time(series, 470.0) is None          # 動画が始まる前
    assert video_time(series, 400.0) is None          # 動画が終わった後


def _livestats():
    return {
        "tm": {"1": {"name": "Japan", "shortName": "JPN M"}, "2": {"name": "USA", "shortName": "USA M"}},
        "pbp": [
            {"period": 2, "periodType": "REGULAR", "tno": 1, "actionType": "2pt", "success": 1, "clock": "07:30:00"},
            {"period": 2, "periodType": "REGULAR", "tno": 2, "actionType": "rebound", "subType": "offensive",
             "success": 1, "clock": "07:20:00"},
            {"period": 2, "periodType": "REGULAR", "tno": 2, "actionType": "3pt", "success": 0, "clock": "07:22:00"},
            {"period": 2, "periodType": "REGULAR", "tno": 1, "actionType": "foul", "success": 1, "clock": "07:10:00"},
            {"period": 1, "periodType": "REGULAR", "tno": 1, "actionType": "2pt", "success": 1, "clock": "07:30:00"},
            {"period": 2, "periodType": "REGULAR", "tno": 1, "actionType": "turnover", "subType": "badpass",
             "success": 1, "clock": "05:00:00"},
        ],
    }


def test_events_from_livestats():
    analysis = {"teams": [{"id": "a", "name": "JPN"}, {"id": "b", "name": "USA"}],
                "scoreboard": [{"t": float(i), "clock": f"7:{40 - i:02d}"} for i in range(0, 31)]}
    assert team_ids(_livestats(), analysis["teams"]) == {"1": "a", "2": "b"}
    gt = events_from_livestats(_livestats(), analysis, period=2)
    got = [(e["t"], e["type"], e["team"], e.get("points"), e.get("offensive")) for e in gt["events"]]
    # ファウル(評価対象外)、第1ピリオド、動画に含まれない 5:00 のターンオーバーは除かれる
    assert got == [(10.0, "made_shot", "a", 2, None), (18.0, "missed_shot", "b", 3, None),
                   (20.0, "rebound", "b", None, True)]
    assert clock_series(analysis["scoreboard"])[0] == (0.0, 460.0)
