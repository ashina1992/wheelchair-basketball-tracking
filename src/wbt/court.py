"""コートの幾何(FIBA規格。車いすバスケも同じコートを使う)。

コート座標系: 原点をコート中心、X 軸を長辺方向(左ゴール側が負)、
Y 軸を短辺方向(カメラから見て奥のサイドラインが負)、単位はメートル。
"""
import numpy as np

LENGTH, WIDTH = 28.0, 15.0
HALF_L, HALF_W = LENGTH / 2, WIDTH / 2
KEY_WIDTH = 4.9                 # 制限区域の幅
FT_DIST = 5.8                   # ベースライン内側からフリースローラインまで
BASKET_DIST = 1.575             # ベースラインからリング中心まで
THREE_R = 6.75                  # 3ポイントラインの半径(リング中心から)
THREE_SIDE = 6.6                # 3ポイントラインの直線部分のY(中心線から)
CIRCLE_R = 1.8                  # センターサークル・フリースローサークルの半径
NO_CHARGE_R = 1.25              # ノーチャージ半円の半径
RIM_HEIGHT = 3.05


def basket_x(side: int) -> float:
    """side=-1 が左ゴール、+1 が右ゴールのリング中心の X 座標。"""
    return side * (HALF_L - BASKET_DIST)


def _arc(cx, cy, r, a0, a1, n=48):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def court_polylines() -> list[np.ndarray]:
    """コートラインを折れ線(各 (N, 2) 配列)のリストで返す。描画・校正の照合に使う。"""
    lines = [
        np.array([[-HALF_L, -HALF_W], [HALF_L, -HALF_W], [HALF_L, HALF_W], [-HALF_L, HALF_W], [-HALF_L, -HALF_W]]),
        np.array([[0, -HALF_W], [0, HALF_W]]),
        _arc(0, 0, CIRCLE_R, 0, 2 * np.pi, 96),
    ]
    for s in (-1, 1):
        bx = s * HALF_L
        ftx = s * (HALF_L - FT_DIST)
        k = KEY_WIDTH / 2
        lines.append(np.array([[bx, -k], [ftx, -k], [ftx, k], [bx, k]]))
        # フリースローサークル(コート中央側の半分)
        a0 = np.pi / 2 if s > 0 else -np.pi / 2
        lines.append(_arc(ftx, 0, CIRCLE_R, a0, a0 + np.pi))
        # 3ポイントライン: 直線部分+円弧
        cx = basket_x(s)
        th = np.arcsin(THREE_SIDE / THREE_R)
        arc = _arc(cx, 0, THREE_R, np.pi - th, np.pi + th) if s > 0 else _arc(cx, 0, THREE_R, -th, th)
        x_join = cx - s * THREE_R * np.cos(th)
        lines.append(np.vstack([[bx, -THREE_SIDE], [x_join, -THREE_SIDE]]))
        lines.append(np.vstack([[bx, THREE_SIDE], [x_join, THREE_SIDE]]))
        lines.append(arc)
    return lines


def court_outline(margin: float = 0.0) -> np.ndarray:
    """コート外周(margin メートル外側に広げた四角形)の4頂点。"""
    x, y = HALF_L + margin, HALF_W + margin
    return np.array([[-x, -y], [x, -y], [x, y], [-x, y]], dtype=np.float64)
