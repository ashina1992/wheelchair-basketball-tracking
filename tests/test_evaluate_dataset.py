import zipfile

import numpy as np

from wbt.detect import suppress_contained
from wbt.dataset import export_cvat_yolo, import_cvat_yolo, read_yolo_labels, split_by_game, to_yolo_lines
from wbt.evaluate import average_precision, iou, match_events
from wbt.shots import is_wide_shot


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10])[0, 0] == 1.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10])[0, 0] - 1 / 3) < 1e-9
    assert iou([0, 0, 1, 1], [5, 5, 6, 6])[0, 0] == 0.0


def test_average_precision_perfect_and_with_false_positive():
    gts = {0: np.array([[0, 0, 10, 10]]), 1: np.array([[20, 20, 30, 30]])}
    perfect = [(0, 0.9, np.array([0, 0, 10, 10])), (1, 0.8, np.array([20, 20, 30, 30]))]
    assert average_precision(perfect, gts)["ap"] == 1.0
    # 最も信頼度の高い予測が誤検出だと、AP は下がる
    with_fp = [(0, 0.95, np.array([50, 50, 60, 60]))] + perfect
    r = average_precision(with_fp, gts)
    assert 0.5 < r["ap"] < 1.0 and r["recall"] == 1.0


def test_duplicate_prediction_counts_once():
    gts = {0: np.array([[0, 0, 10, 10]])}
    preds = [(0, 0.9, np.array([0, 0, 10, 10])), (0, 0.8, np.array([0, 0, 10, 10]))]
    r = average_precision(preds, gts)
    assert r["recall"] == 1.0 and r["precision"] == 0.5


def test_match_events():
    gt = [{"t": 10, "type": "made_shot", "team": "a"}, {"t": 30, "type": "made_shot", "team": "b"}]
    pred = [{"t": 11, "type": "made_shot", "team": "a"}, {"t": 30.5, "type": "made_shot", "team": "a"},
            {"t": 50, "type": "turnover", "team": "b"}]
    rep = match_events(pred, gt)
    assert rep["made_shot"]["tp"] == 1 and rep["made_shot"]["precision"] == 0.5 and rep["made_shot"]["recall"] == 0.5
    assert rep["turnover"]["precision"] == 0.0 and rep["turnover"]["recall"] is None


def test_suppress_contained_removes_upper_body_duplicate():
    full = [100, 100, 200, 260]        # 車いすまで含む箱
    upper = [110, 100, 190, 180]       # 同じ選手の上半身だけの箱
    other = [180, 120, 280, 280]       # 隣の選手(一部重なるだけ)
    keep = suppress_contained(np.array([full, upper, other], float), np.array([0.8, 0.6, 0.7]))
    assert keep.tolist() == [True, False, True]


def test_suppress_contained_keeps_box_when_container_is_weak():
    keep = suppress_contained(np.array([[0, 0, 100, 100], [10, 10, 50, 50]], float), np.array([0.1, 0.9]))
    assert keep.tolist() == [True, True]


def test_estimate_offset_recovers_constant_delay():
    from wbt.evaluate import estimate_offset
    pred = [{"t": t, "type": "missed_shot", "team": "a"} for t in (10.0, 30.0, 50.0, 70.0)]
    gt = [{"t": t + 1.6, "type": "missed_shot", "team": "a"} for t in (10.0, 30.0, 50.0, 70.0)]
    off = estimate_offset(pred, gt, tol=0.5)
    assert abs(off + 1.6) <= 0.5


def test_yolo_roundtrip():
    lines = to_yolo_lines([("ball", np.array([100, 50, 120, 70]))], 1920, 1080)
    assert lines[0].startswith("1 ")


def test_is_wide_shot():
    small = np.array([[0, 0, 40, 100]] * 8)
    assert is_wide_shot(small, 1080)
    assert not is_wide_shot(small[:3], 1080)
    # 観客が小さく多数写っていても、手前に大きな人物がいればサイドラインカメラ等とみなす
    sideline = np.vstack([small, [[0, 0, 300, 600]]])
    assert not is_wide_shot(sideline, 1080)


def test_split_groups_multi_file_recordings(tmp_path):
    """前半・後半を別ファイルで録画した試合(_1, _2)は、試合名の指定でまとめて同じ分割に入る。"""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    for stem in ("usa_jpn_1_000010", "usa_jpn_2_000010", "usa_jpn_10_000010", "usa_jpnx_1_000010"):
        (src / "images" / f"{stem}.jpg").write_bytes(b"jpg")
    counts = split_by_game(src, tmp_path / "out", val_games=[], test_games=["usa_jpn"])
    assert counts == {"train": 1, "val": 0, "test": 3}
    assert (tmp_path / "out" / "images" / "train" / "usa_jpnx_1_000010.jpg").exists()


def test_cvat_export_import_and_split(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    for game in ("g1", "g2", "g3"):
        (raw / "images" / f"{game}_000010.jpg").write_bytes(b"jpg")
        (raw / "labels" / f"{game}_000010.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    zp = tmp_path / "batch.zip"
    assert export_cvat_yolo(raw, zp) == 3
    with zipfile.ZipFile(zp) as z:
        assert z.read("obj.names").decode().split() == ["player", "ball", "rim", "referee"]
    out = tmp_path / "annotated"
    assert import_cvat_yolo(zp, raw / "images", out) == 3
    counts = split_by_game(out, tmp_path / "yolo", val_games=["g2"], test_games=["g3"])
    assert counts == {"train": 1, "val": 1, "test": 1}
    assert (tmp_path / "yolo" / "labels" / "test" / "g3_000010.txt").exists()
    labels = read_yolo_labels(tmp_path / "yolo" / "labels" / "train" / "g1_000010.txt", 100, 100)
    np.testing.assert_allclose(labels[0][1], [45, 40, 55, 60])
