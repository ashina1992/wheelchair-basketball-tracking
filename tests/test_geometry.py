import cv2
import numpy as np
import pytest

from wbt.calibration import fit_homography
from wbt.camera import CameraState, CameraTracker, Features, Keyframe
from wbt.court import court_polylines

# 実際のキーフレーム校正に近い、コート→画像のホモグラフィ
H_TRUE = np.array([[119.4, -22.2, 790.5], [0.12, 25.0, 612.7], [0.0011, -0.025, 1.0]])


def proj(H, pts):
    return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def test_fit_homography_from_points():
    court = [[-2.3, 0], [2.3, 0], [0, -2.3], [0, 2.3], [8.2, -2.45], [8.2, 2.45]]
    img = proj(H_TRUE, court)
    pts = [[*i, *c] for i, c in zip(img.tolist(), court)]
    H, s = fit_homography(pts)
    assert s["point_err_px_max"] < 0.01
    np.testing.assert_allclose(proj(H, [[0, -7.5]]), proj(H_TRUE, [[0, -7.5]]), atol=0.01)


def test_line_constraints_fix_extrapolation():
    """4点が中央に偏っていても、サイドライン上の拘束があれば遠い場所の誤差が小さくなる。"""
    rng = np.random.default_rng(0)
    court = [[-2.3, 0], [2.3, 0], [0, -2.3], [0, 2.3]]
    img = proj(H_TRUE, court) + rng.normal(0, 3, (4, 2))
    pts = [[*i, *c] for i, c in zip(img.tolist(), court)]
    far = proj(H_TRUE, [[-10, -7.5], [0, -7.5], [10, -7.5], [-10, 7.5], [0, 7.5], [10, 7.5]])
    lines = [[x, y, None, -7.5] for x, y in far[:3]] + [[x, y, None, 7.5] for x, y in far[3:]]
    target = proj(H_TRUE, [[10, -7.5]])
    err_pts = np.linalg.norm(proj(fit_homography(pts)[0], [[10, -7.5]]) - target)
    err_lines = np.linalg.norm(proj(fit_homography(pts + lines)[0], [[10, -7.5]]) - target)
    assert err_lines < err_pts


def test_fit_homography_requires_four_points():
    with pytest.raises(ValueError):
        fit_homography([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, None, 7.5]])


def _tracker():
    empty = Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    kf_a = Keyframe("a", H_TRUE, np.array([[1300.0, 340.0]]), empty, side=1)
    # キーフレーム b はカメラを右にパンした画像: b の画像 = S · a の画像
    S = np.array([[1, 0, -400], [0, 1, 20], [0, 0, 1.0]])
    kf_b = Keyframe("b", S @ H_TRUE, np.array([[500.0, 360.0]]), empty, side=1)
    return CameraTracker([kf_a, kf_b], extractor=object()), S


def test_court_img_roundtrip():
    cam, _ = _tracker()
    G = np.array([[1.1, 0.01, -30], [0.0, 1.1, 10], [0, 0, 1.0]])   # フレーム → キーフレーム a
    st = CameraState(0, G, 100)
    pts = np.array([[0.0, 0.0], [8.2, 2.45], [-12.0, -6.0]])
    img = cam.court_to_img(st, pts)
    np.testing.assert_allclose(cam.img_to_court(st, img), pts, atol=1e-6)


def test_rims_transfer_between_keyframes():
    """キーフレーム b で指定したリングが、キーフレーム a に照合中のフレームにも正しく写る。"""
    cam, S = _tracker()
    st = CameraState(0, np.eye(3), 100)
    rims, sides = cam.rims_in_frame(st)
    np.testing.assert_allclose(rims[0], [1300, 340], atol=1e-6)
    expected_b_in_a = proj(np.linalg.inv(S), [[500, 360]])[0]
    np.testing.assert_allclose(rims[1], expected_b_in_a, atol=1e-6)
    assert sides == [1, 1]


def test_court_polylines_are_within_court():
    for pl in court_polylines():
        assert np.all(np.abs(pl[:, 0]) <= 14 + 1e-9) and np.all(np.abs(pl[:, 1]) <= 7.5 + 1e-9)
