"""ユニフォーム色による選手のチーム識別。

車いすの選手は座っているため箱の縦横比が小さく、立っている審判・スタッフは縦長になる。
縦横比で審判等を先に除き、残った(座っている)人物の胴体の色を2クラスタに分けてチームとする。
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

STANDING_ASPECT = 2.05   # 箱の縦/横がこれを超えるトラックは立っている人物(審判等)とみなす
FLOOR_DIST = 22.0        # 床の色からこの距離(Lab)以内の画素は胴体の色の計算から除く


def floor_color(frame: np.ndarray, court_poly_img: np.ndarray | None) -> np.ndarray | None:
    """画像上のコート領域の代表色(Lab)。木目の床が大半を占めるので中央値で求める。"""
    if court_poly_img is None:
        return None
    small = cv2.resize(frame, None, fx=0.25, fy=0.25)
    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(court_poly_img * 0.25)], 255)
    if mask.sum() == 0:
        return None
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[mask > 0].astype(np.float32)
    return np.median(lab, axis=0)


def torso_color(frame: np.ndarray, box, floor_lab: np.ndarray | None = None) -> np.ndarray | None:
    """箱の上部中央(着座した選手の胴体付近)の代表色を Lab で返す。床の画素は除く。"""
    x1, y1, x2, y2 = map(int, box[:4])
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    crop = frame[max(0, y1 + int(0.15 * h)): y1 + int(0.45 * h), max(0, x1 + int(0.25 * w)): x2 - int(0.25 * w)]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if floor_lab is not None:
        keep = np.linalg.norm(lab - floor_lab, axis=1) > FLOOR_DIST
        if keep.mean() < 0.2:
            return None
        lab = lab[keep]
    return np.median(lab, axis=0)


def rgb_to_lab(rgb) -> np.ndarray:
    px = np.uint8([[list(reversed(rgb))]])  # RGB → BGR
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def assign_teams(track_obs: dict[int, list[tuple[np.ndarray | None, float]]],
                 team_colors: dict[str, tuple] | None = None, min_obs: int = 3) -> dict[int, str]:
    """トラックごとの観測 (胴体の色, 箱の縦横比) から、各トラックを "a" / "b" / "other" に割り当てる。

    縦横比の中央値が大きいトラックは立っている人物として "other" にする。
    残りの色の観測を k=2 でクラスタリングし、トラックごとに多数決する。
    team_colors(RGB)が与えられれば各クラスタを近い方のチームに対応付け、なければ観測数の多い順に a / b とする。
    """
    result, tids, obs = {}, [], []
    for tid, rows in track_obs.items():
        if len(rows) < min_obs:
            continue
        if np.median([a for _, a in rows]) > STANDING_ASPECT:
            result[tid] = "other"
            continue
        for c, _ in rows:
            if c is not None:
                tids.append(tid)
                obs.append(c)
    if len(obs) < 10:
        return result
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.array(obs))
    centers, labels = km.cluster_centers_, km.labels_

    if team_colors and len(team_colors) == 2:
        (ta, ca), (tb, cb) = team_colors.items()
        la, lb = rgb_to_lab(ca), rgb_to_lab(cb)
        # 2通りの対応のうち、色の距離の合計が小さい方を採る
        d0 = np.linalg.norm(centers[0] - la) + np.linalg.norm(centers[1] - lb)
        d1 = np.linalg.norm(centers[1] - la) + np.linalg.norm(centers[0] - lb)
        cluster_team = {0: ta, 1: tb} if d0 <= d1 else {0: tb, 1: ta}
    else:
        order = np.argsort(-np.bincount(labels, minlength=2))
        cluster_team = {int(order[0]): "a", int(order[1]): "b"}

    votes: dict[int, list[str]] = {}
    for tid, lab in zip(tids, labels):
        votes.setdefault(tid, []).append(cluster_team[int(lab)])
    for tid, v in votes.items():
        vals, counts = np.unique(v, return_counts=True)
        result[tid] = str(vals[np.argmax(counts)])
    return result
