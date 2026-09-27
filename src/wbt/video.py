"""動画の読み込みとフレーム抽出。"""
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


BAR_SPATIAL_STD = 15.0   # 帯とみなす行・列の、行(列)内の画素値の標準偏差の上限


def detect_content_box(cap: cv2.VideoCapture, n_samples: int = 12, ratio: float = 0.3) -> Box:
    """画面録画の帯を除いた映像領域 (x0, y0, x1, y1) を返す。

    帯は「時間方向の変化が小さい」かつ「行(列)の中でほぼ一様」な行・列とする。
    - 帯は純黒とは限らず(グラデーションのグレー、フェード等)明るさも変わるため、時間方向の変化は
      サンプルフレーム間の標準偏差が全体の中央値の ratio 倍未満かどうかで判定する
    - 観客席のように時間方向にはほとんど変化しないが、行の中では模様がある領域を帯と誤判定しないよう、
      行(列)内の標準偏差も見る
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grays = []
    for idx in np.linspace(0, total - 1, n_samples + 2, dtype=int)[1:-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(grays) < 2:
        raise RuntimeError("フレームを読み込めませんでした")
    stack = np.stack(grays)
    tstd = stack.std(axis=0)
    row_v, col_v = tstd.mean(axis=1), tstd.mean(axis=0)
    # 行(列)内の一様さはフレームごとに測る(平均画像ではカメラの動きで模様がぼけて一様に見えるため)
    row_s = np.median(stack.std(axis=2), axis=0)
    col_s = np.median(stack.std(axis=1), axis=0)
    row_content = (row_v > ratio * np.median(row_v)) | (row_s > BAR_SPATIAL_STD)
    col_content = (col_v > ratio * np.median(col_v)) | (col_s > BAR_SPATIAL_STD)
    rows, cols = np.where(row_content)[0], np.where(col_content)[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def extract_frames(video: Path, out: Path, every_sec: float = 1.0, crop: bool = True) -> dict:
    """every_sec 秒ごとにフレームを JPEG で out に保存し、抽出条件を返す。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * every_sec))
    box = detect_content_box(cap) if crop else None

    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        if box:
            x0, y0, x1, y1 = box
            frame = frame[y0:y1, x0:x1]
        cv2.imwrite(str(out / f"frame_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return {"fps": fps, "total_frames": total, "step": step, "crop": box, "n_saved": n}
