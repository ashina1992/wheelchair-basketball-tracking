"""スコアボードの数字の読み取り(フォントの見本画像との照合)。

中継のスコアボードは決まったフォントで描かれるため、数字の見本(1文字ずつの切り出し画像)と照合すれば
汎用 OCR より確実に読める。Tesseract はこのフォントの「5」を読めず、「14」を「24」と誤読することがあった。

見本は手作業で用意せず、時間方向の整合チェックで値が確定した試合時計の読み取りから自動で集める
(scripts/build_digit_templates.py)。試合時計は1秒ずつ変わるため、数十秒の映像で 0〜9 がそろう。
"""
from pathlib import Path

import cv2
import numpy as np

GLYPH = 20            # 1文字を正規化する大きさ(GLYPH x GLYPH、縦横比は保つ)
MIN_HEIGHT_RATIO = 0.6    # 最も背の高い塊に対する、数字とみなす塊の最小の高さ(コロンや汚れを除く)
MERGED_WIDTH_RATIO = 0.95  # 幅が高さのこの倍を超える塊は、2文字がつながったものとみなして分ける


def binarize(roi: np.ndarray) -> np.ndarray:
    """文字が白(255)、背景が黒(0)の2値画像にする。"""
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:   # 背景の方が明るい場合は反転して、文字を白にそろえる
        b = 255 - b
    return b


def glyphs(roi: np.ndarray) -> list[np.ndarray]:
    """領域から数字の塊を左から順に切り出し、GLYPH x GLYPH に正規化して返す。"""
    b = binarize(roi)
    # 隣り合う数字が拡大・ぼけでつながることがあるため、少し細らせてから塊に分ける
    thin = cv2.erode(b, np.ones((3, 3), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=4)
    if n <= 1:
        return []
    h_max = stats[1:, cv2.CC_STAT_HEIGHT].max()
    boxes = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= MIN_HEIGHT_RATIO * h_max
             and stats[i, cv2.CC_STAT_HEIGHT] < 0.98 * b.shape[0]]
    # つながったままの2文字(1文字の想定幅よりかなり広い塊)は、最も細い列で左右に分ける
    split = []
    for x, y, w, h, a in boxes:
        if w > MERGED_WIDTH_RATIO * h:
            cols = thin[y:y + h, x:x + w].sum(axis=0)
            lo, hi = int(w * 0.3), int(w * 0.7)
            cut = lo + int(np.argmin(cols[lo:hi]))
            split += [(x, y, cut, h, a), (x + cut, y, w - cut, h, a)]
        else:
            split.append((x, y, w, h, a))
    out = []
    for x, y, w, h, _ in sorted(split, key=lambda s: s[0]):
        x, y, w, h = max(0, x - 2), max(0, y - 2), w + 4, h + 4   # 細らせた分を戻す
        crop = b[y:y + h, x:x + w]
        s = (GLYPH - 2) / max(w, h)
        crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((GLYPH, GLYPH), np.float32)
        oy, ox = (GLYPH - crop.shape[0]) // 2, (GLYPH - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop / 255.0
        out.append(canvas)
    return out


class DigitReader:
    """見本との最近傍照合で数字列を読む。"""

    def __init__(self, samples: np.ndarray, labels: np.ndarray, max_dist: float = 0.25):
        self.samples = samples.reshape(len(samples), -1).astype(np.float32)
        self.labels = labels
        self.max_dist = max_dist

    def read_glyph(self, g: np.ndarray) -> int | None:
        d = np.sqrt(((self.samples - g.reshape(1, -1)) ** 2).mean(axis=1))
        k = int(np.argmin(d))
        return None if d[k] > self.max_dist else int(self.labels[k])

    def read(self, roi: np.ndarray) -> str:
        """数字列を返す。照合できない塊があれば空文字を返す(誤読より読めない方を選ぶ)。"""
        chars = []
        for g in glyphs(roi):
            c = self.read_glyph(g)
            if c is None:
                return ""
            chars.append(str(c))
        return "".join(chars)

    def save(self, path: Path):
        np.savez_compressed(path, samples=self.samples.reshape(-1, GLYPH, GLYPH), labels=self.labels)

    @classmethod
    def load(cls, path: Path) -> "DigitReader":
        d = np.load(path)
        return cls(d["samples"], d["labels"])


def harvest(roi: np.ndarray, text: str) -> list[tuple[np.ndarray, int]]:
    """値が分かっている表示(例: "7:25")から、数字1文字ずつの見本を切り出す。塊の数が合わなければ空。"""
    digits = [int(c) for c in text if c.isdigit()]
    gs = glyphs(roi)
    return list(zip(gs, digits)) if len(gs) == len(digits) else []
