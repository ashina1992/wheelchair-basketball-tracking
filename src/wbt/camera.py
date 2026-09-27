"""カメラ運動推定と、画像座標↔コート座標の変換(動的ホモグラフィ)。

放送用の広角カメラは位置を固定したままパン・チルト・ズームするため、
2フレーム間の画像の対応は奥行きに関係なく1つのホモグラフィで表せる(回転のみのカメラモデル)。
会場ごとに校正したキーフレーム(コート→画像のホモグラフィとリング位置が既知)に対して、
各フレームのホモグラフィ G(フレーム→キーフレーム)を推定し、校正結果を各フレームへ伝播する。
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FEATURE_SCALE = 0.5      # 特徴点抽出時の縮小率(速度のため)
MIN_INLIERS_CHAIN = 40   # 隣接フレーム間の照合に必要なインライア数
MIN_INLIERS_ANCHOR = 50  # キーフレームへの照合に必要なインライア数
# 30 では、同じ大会の看板が映る別会場の映像が誤って照合された(33点)。正しい照合は 56 点以上だった
MIN_ANCHOR_SPREAD = 0.10  # キーフレームへの照合で、インライアの凸包が画像に占める面積の下限
# 別会場でも、センターサークルの大会ロゴ(床に描かれた平面)だけで 50 点照合されることがあった。
# ロゴは平面なのでホモグラフィとしては整合してしまうため、インライアが画像全体に散らばっているかで判定する
REANCHOR_EVERY = 15      # この回数ごとにキーフレームへ直接照合し直してドリフトを抑える


@dataclass
class Features:
    pts: np.ndarray    # (N, 2) 元解像度の画像座標
    desc: np.ndarray   # (N, 128) SIFT 記述子


@dataclass
class Keyframe:
    """校正済みのキーフレーム。H_court は コート座標(m)→キーフレーム画像座標。"""
    name: str
    H_court: np.ndarray
    rims: np.ndarray            # (R, 2) キーフレーム画像上のリング中心
    features: Features
    side: int = 0               # このキーフレームが写すゴールの側(-1: 左, +1: 右)


@dataclass
class CameraState:
    keyframe: int | None = None   # 照合中のキーフレーム番号(None は未照合)
    G: np.ndarray | None = None   # 現フレーム→キーフレームのホモグラフィ
    inliers: int = 0
    anchored: bool = False        # このフレームでキーフレームに直接照合したか

    @property
    def valid(self) -> bool:
        return self.G is not None


class FeatureExtractor:
    def __init__(self, n_features: int = 3000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, frame: np.ndarray, exclude: list[tuple[float, float, float, float]] = ()) -> Features:
        """exclude(選手の箱、スコアボード等)を除いた領域から特徴点を抽出する。"""
        small = cv2.resize(frame, None, fx=FEATURE_SCALE, fy=FEATURE_SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in exclude:
            s = FEATURE_SCALE
            cv2.rectangle(mask, (int(x1 * s) - 4, int(y1 * s) - 4), (int(x2 * s) + 4, int(y2 * s) + 4), 0, -1)
        kps, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None:
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        pts = np.float32([k.pt for k in kps]) / FEATURE_SCALE
        return Features(pts, desc)

    def match(self, a: Features, b: Features, return_inliers: bool = False):
        """a の画像座標 → b の画像座標のホモグラフィとインライア数を返す。

        return_inliers=True のときは、a 側のインライアの座標 (K, 2) も返す。
        """
        fail = (None, 0, np.zeros((0, 2), np.float32)) if return_inliers else (None, 0)
        if len(a.pts) < 8 or len(b.pts) < 8:
            return fail
        pairs = self.matcher.knnMatch(a.desc, b.desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return fail
        src = a.pts[[m.queryIdx for m in good]]
        dst = b.pts[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            return fail
        if return_inliers:
            return H, int(inl.sum()), src[inl.ravel().astype(bool)]
        return H, int(inl.sum())


def inlier_spread(pts: np.ndarray, frame_area: float) -> float:
    """インライアの凸包の面積が画像に占める割合。"""
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.float32(pts)))) / frame_area


class CameraTracker:
    """フレームごとに、キーフレームに対するホモグラフィ G を推定する。

    通常は隣接フレーム間の推定を連鎖させ、ショット切替時と一定間隔でキーフレームへ直接照合し直す。
    """

    def __init__(self, keyframes: list[Keyframe], extractor: FeatureExtractor | None = None):
        self.keyframes = keyframes
        self.fx = extractor or FeatureExtractor()
        self.prev: Features | None = None
        self.state = CameraState()
        self.since_anchor = 0

    def reset(self):
        self.prev, self.state, self.since_anchor = None, CameraState(), 0

    def _anchor(self, cur: Features, frame_area: float) -> CameraState:
        best = CameraState()
        for i, kf in enumerate(self.keyframes):
            H, n, pts = self.fx.match(cur, kf.features, return_inliers=True)
            if H is None or n < MIN_INLIERS_ANCHOR or n <= best.inliers:
                continue
            if inlier_spread(pts, frame_area) < MIN_ANCHOR_SPREAD:
                continue
            best = CameraState(i, H, n, True)
        return best

    def update(self, frame: np.ndarray, exclude=(), cut: bool = False) -> CameraState:
        cur = self.fx.extract(frame, exclude)
        st = CameraState()
        need_anchor = cut or not self.state.valid or self.since_anchor >= REANCHOR_EVERY
        if need_anchor:
            st = self._anchor(cur, frame.shape[0] * frame.shape[1])
        if not st.valid and self.state.valid and not cut and self.prev is not None:
            H, n = self.fx.match(cur, self.prev)   # 現フレーム → 前フレーム
            if H is not None and n >= MIN_INLIERS_CHAIN:
                st = CameraState(self.state.keyframe, self.state.G @ H, n, False)
        self.since_anchor = 0 if st.anchored else self.since_anchor + 1
        self.prev, self.state = cur, st
        return st

    # --- 座標変換 ---
    def img_to_court(self, st: CameraState, pts: np.ndarray) -> np.ndarray | None:
        if not st.valid or len(pts) == 0:
            return None
        kf = self.keyframes[st.keyframe]
        H = np.linalg.inv(kf.H_court) @ st.G
        return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)

    def court_to_img(self, st: CameraState, pts: np.ndarray) -> np.ndarray | None:
        if not st.valid:
            return None
        kf = self.keyframes[st.keyframe]
        H = np.linalg.inv(st.G) @ kf.H_court
        return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)

    def rims_in_frame(self, st: CameraState) -> tuple[np.ndarray, list[int]] | None:
        """全キーフレームで指定したリング位置を現フレームの画像座標に写し、(座標, ゴールの側 -1/+1) を返す。

        回転のみのカメラモデルでは画像間のホモグラフィは平面の選び方によらないため、
        キーフレーム j の画像座標は H_court_k · H_court_j^-1 でキーフレーム k の画像座標に移せる。
        """
        if not st.valid:
            return None
        k = self.keyframes[st.keyframe]
        pts, sides = [], []
        for kf in self.keyframes:
            to_k = k.H_court @ np.linalg.inv(kf.H_court)
            for rim in kf.rims:
                pts.append(cv2.perspectiveTransform(np.float64(rim).reshape(1, 1, 2), np.linalg.inv(st.G) @ to_k)[0, 0])
                sides.append(kf.side)
        return np.array(pts), sides

    def zoom(self, st: CameraState) -> float:
        """キーフレームに対する拡大率(>1 で寄り)。リング付近の判定半径の調整に使う。"""
        if not st.valid:
            return 1.0
        J = np.linalg.inv(st.G)[:2, :2]
        return float(np.sqrt(abs(np.linalg.det(J))))


def save_keyframe_features(path: Path, f: Features):
    np.savez_compressed(path, pts=f.pts.astype(np.float32), desc=np.clip(f.desc, 0, 255).astype(np.uint8))


def load_keyframe_features(path: Path) -> Features:
    d = np.load(path)
    return Features(d["pts"], d["desc"].astype(np.float32))
