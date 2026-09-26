"""動画の読み込みとフレーム抽出。"""
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def detect_content_box(cap: cv2.VideoCapture, n_samples: int = 12, ratio: float = 0.3) -> Box:
    """時間方向の変化が小さい行・列を帯とみなし、映像領域 (x0, y0, x1, y1) を返す。

    画面録画の帯は純黒とは限らず(グラデーションのグレー、フェード等)明るさも変わるため、
    サンプルフレーム間の画素値の標準偏差が全体の中央値の ratio 倍未満の行・列を帯とする。
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
    tstd = np.stack(grays).std(axis=0)
    row_v, col_v = tstd.mean(axis=1), tstd.mean(axis=0)
    rows = np.where(row_v > ratio * np.median(row_v))[0]
    cols = np.where(col_v > ratio * np.median(col_v))[0]
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
