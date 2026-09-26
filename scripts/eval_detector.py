"""test セットで検出モデルを評価する(既製 COCO モデルと自前モデルを同じ方法で比べる)。

使い方:
    python scripts/eval_detector.py data/dataset/yolo --split test --weights yolov8s.pt runs/train/best.pt \
        --out docs/results/detector_eval.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from wbt.dataset import read_yolo_labels
from wbt.detect import CLASSES, Detector
from wbt.evaluate import detection_report


def evaluate(weights: str, dataset: Path, split: str, imgsz: int) -> dict:
    det = Detector(weights, imgsz=imgsz, conf={n: 0.01 for n in CLASSES})
    preds: dict[int, list] = {}
    gts: dict[int, dict] = {}
    images = sorted((dataset / "images" / split).glob("*.jpg"))
    for k, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for c, box in read_yolo_labels(dataset / "labels" / split / f"{img_path.stem}.txt", w, h):
            gts.setdefault(c, {}).setdefault(k, []).append(box)
        d = det(img)
        rows = {0: np.hstack([d.players.xyxy, d.players.conf[:, None]]) if len(d.players) else np.zeros((0, 5)),
                1: d.balls, 2: d.rims, 3: d.referees}
        for c, arr in rows.items():
            for r in arr:
                preds.setdefault(c, []).append((k, float(r[4]), r[:4]))
    gts = {c: {k: np.array(v) for k, v in g.items()} for c, g in gts.items()}
    names = {i: n for i, n in enumerate(CLASSES) if n in det.available}
    return {"weights": weights, "n_images": len(images), "classes": detection_report(preds, gts, names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    results = [evaluate(w, args.dataset, args.split, args.imgsz) for w in args.weights]
    for r in results:
        print(r["weights"])
        for name, m in r["classes"].items():
            ap50 = "-" if m["ap"] is None else f"{m['ap']:.3f}"
            ap5095 = "-" if m["ap50_95"] is None else f"{m['ap50_95']:.3f}"
            print(f"  {name:8s} AP50={ap50} AP50-95={ap5095} n_gt={m['n_gt']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
