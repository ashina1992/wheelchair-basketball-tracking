"""スコアボードの数字の見本を、試合時計の読み取りから自動で集めて会場プロファイルに登録する。

1. 試合時計を Tesseract で読み、時間方向の整合チェック(wbt.scoreboard)で値を確定する
2. 値が確定した表示(例: "7:25")から数字を1文字ずつ切り出して見本にする
3. 見本の半分で残り半分を読ませて精度を確認し、全見本を profiles/<name>_digits.npz に保存する

使い方:
    python scripts/build_digit_templates.py <video> --profile ottawa2026
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from wbt.digits import DigitReader, harvest
from wbt.profiles import load_profile
from wbt.scoreboard import _ocr, _stable_clock, clock_candidates, format_clock
from wbt.video import detect_content_box

PER_DIGIT = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--every-sec", type=float, default=0.5)
    args = ap.parse_args()

    prof = load_profile(args.profile)
    x0c, y0c, x1c, y1c = prof.scoreboard["clock"]
    cap = cv2.VideoCapture(str(args.video))
    crop = detect_content_box(cap)
    ts, cands, rois = [], [], []
    next_t = 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < next_t:
            continue
        next_t = t + args.every_sec
        x0, y0, x1, y1 = crop
        f = cv2.resize(f[y0:y1, x0:x1], tuple(prof.frame_size))
        roi = f[y0c:y1c, x0c:x1c].copy()
        ts.append(t)
        cands.append(clock_candidates(_ocr(f, prof.scoreboard["clock"], "0123456789:.")))
        rois.append(roi)
    clock = _stable_clock(ts, cands)

    samples = []
    for roi, v in zip(rois, clock):
        if v is not None and v >= 60:   # "M:SS" 表示の区間だけ使う
            samples += harvest(roi, format_clock(v))
    counts = Counter(d for _, d in samples)
    print(f"確定した時計の読み取り {sum(v is not None for v in clock)}/{len(clock)}、見本 {len(samples)} 個、数字ごと: {dict(sorted(counts.items()))}")
    missing = [d for d in range(10) if counts[d] == 0]
    if missing:
        raise SystemExit(f"見本が集まらなかった数字があります: {missing}(もっと長い映像で実行してください)")

    random.seed(0)
    by_digit: dict[int, list] = {}
    for g, d in samples:
        by_digit.setdefault(d, []).append(g)
    keep = [(g, d) for d, gs in by_digit.items() for g in random.sample(gs, min(PER_DIGIT, len(gs)))]
    random.shuffle(keep)
    half = len(keep) // 2
    tr, te = keep[:half], keep[half:]
    reader = DigitReader(np.array([g for g, _ in tr]), np.array([d for _, d in tr]))
    ok = sum(reader.read_glyph(g) == d for g, d in te)
    print(f"見本の半分で残り半分を読んだ正解率: {ok}/{len(te)}")

    full = DigitReader(np.array([g for g, _ in keep]), np.array([d for _, d in keep]))
    out = prof.path.parent / f"{prof.name}_digits.npz"
    full.save(out)
    d = json.loads(prof.path.read_text(encoding="utf-8"))
    d["digits"] = out.name
    prof.path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
