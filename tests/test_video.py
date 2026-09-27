import cv2
import numpy as np

from wbt.video import detect_content_box

W, H = 160, 96


def _write(path, top_band):
    """上端 16 行に top_band(フレーム番号 → 画像)を描き、その下は毎フレーム変わる映像にした動画を作る。"""
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for i in range(30):
        f = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        f[:16] = top_band(i)
        vw.write(f)
    vw.release()
    return cv2.VideoCapture(str(path))


def test_black_bar_is_cropped(tmp_path):
    cap = _write(tmp_path / "bar.mp4", lambda i: 0)
    x0, y0, x1, y1 = detect_content_box(cap)
    assert (x0, x1, y1) == (0, W, H) and 14 <= y0 <= 18


def test_static_textured_area_is_not_cropped(tmp_path):
    """観客席のように時間方向にはほぼ変化しないが模様のある領域は、帯として切り取らない。"""
    texture = np.random.default_rng(1).integers(0, 255, (16, W, 3), dtype=np.uint8)
    cap = _write(tmp_path / "static.mp4", lambda i: texture)
    assert detect_content_box(cap)[1] == 0
