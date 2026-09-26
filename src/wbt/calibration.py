"""キーフレームの校正: 点の対応とライン上の拘束から、コート→画像のホモグラフィを推定する。

校正点は [img_x, img_y, X, Y] の形で与える。X か Y の一方を None にすると
「その画像点はコート上の直線 Y=一定(サイドライン等)または X=一定(ベースライン等)の上にある」という拘束になる。
コート中央のようにラインの交点が見えにくい場面でも、サイドラインの縁などを使って校正できる。
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

LINE_WEIGHT = 60.0   # ライン拘束の残差(m)を画素相当に換算する係数


def _proj(H, P):
    return cv2.perspectiveTransform(np.float64(P).reshape(-1, 1, 2), H).reshape(-1, 2)


def fit_homography(points: list[list[float | None]]) -> tuple[np.ndarray, dict]:
    """コート座標 → 画像座標のホモグラフィと、残差の要約を返す。"""
    full = [p for p in points if p[2] is not None and p[3] is not None]
    on_x = [p for p in points if p[2] is not None and p[3] is None]   # X=一定の直線上
    on_y = [p for p in points if p[2] is None and p[3] is not None]   # Y=一定の直線上
    if len(full) < 4:
        raise ValueError("完全な対応点(X, Y とも既知)が4点以上必要です")
    P_img = np.float64([p[:2] for p in full])
    P_court = np.float64([p[2:] for p in full])
    H0, _ = cv2.findHomography(P_court, P_img, 0)
    H0 = H0 / H0[2, 2]
    if not on_x and not on_y:
        return H0, _summary(H0, full, on_x, on_y)

    def residuals(h):
        H = np.append(h, 1).reshape(3, 3)
        r = [(_proj(H, P_court) - P_img).ravel()]
        Hi = np.linalg.inv(H)
        if on_x:
            r.append((_proj(Hi, [p[:2] for p in on_x])[:, 0] - [p[2] for p in on_x]) * LINE_WEIGHT)
        if on_y:
            r.append((_proj(Hi, [p[:2] for p in on_y])[:, 1] - [p[3] for p in on_y]) * LINE_WEIGHT)
        return np.concatenate(r)

    sol = least_squares(residuals, H0.ravel()[:8])
    H = np.append(sol.x, 1).reshape(3, 3)
    return H, _summary(H, full, on_x, on_y)


def _summary(H, full, on_x, on_y) -> dict:
    e = np.linalg.norm(_proj(H, [p[2:] for p in full]) - np.float64([p[:2] for p in full]), axis=1)
    Hi = np.linalg.inv(H)
    line_err = []
    if on_x:
        line_err += np.abs(_proj(Hi, [p[:2] for p in on_x])[:, 0] - [p[2] for p in on_x]).tolist()
    if on_y:
        line_err += np.abs(_proj(Hi, [p[:2] for p in on_y])[:, 1] - [p[3] for p in on_y]).tolist()
    return {"n_points": len(full), "n_line": len(on_x) + len(on_y),
            "point_err_px_mean": round(float(e.mean()), 2), "point_err_px_max": round(float(e.max()), 2),
            "line_err_m_max": round(float(max(line_err)), 3) if line_err else None}
