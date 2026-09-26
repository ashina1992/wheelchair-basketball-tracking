"""既製のYOLOv8(COCO学習済み)でフレーム群の person / sports ball を検出し、ベースラインを記録する。

出力:
    <out>/<model>/vis/*.jpg        検出結果の可視化
    <out>/<model>/detections.csv   frame, cls, x1, y1, x2, y2, conf
    <out>/<model>/summary.json     クラス別の検出数・信頼度、推論時間

使い方:
    python scripts/detect_baseline.py data/frames/jpn_pol --models yolov8n.pt yolov8s.pt yolov8m.pt
"""
import argparse
import json
from pathlib import Path

from wbt.baseline import run_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    ap.add_argument("--out", type=Path, default=Path("runs/baseline"))
    ap.add_argument("--imgsz", type=int, default=1280, help="選手・ボールが小さく映るため既定値640より大きめにする")
    args = ap.parse_args()

    frames = sorted(args.frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"フレームがありません: {args.frames_dir}")
    for m in args.models:
        print(json.dumps(run_baseline(m, frames, args.out, imgsz=args.imgsz), ensure_ascii=False))


if __name__ == "__main__":
    main()
