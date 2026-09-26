"""試合動画から一定間隔でフレームを抽出する(画面録画の帯は自動クロップ)。

使い方:
    python scripts/extract_frames.py <video> --every-sec 1.0 --out data/frames/jpn_pol
"""
import argparse
from pathlib import Path

from wbt.video import extract_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every-sec", type=float, default=1.0, help="抽出間隔(秒)")
    ap.add_argument("--no-crop", action="store_true", help="黒帯クロップを無効化")
    args = ap.parse_args()

    info = extract_frames(args.video, args.out, args.every_sec, crop=not args.no_crop)
    print(info)


if __name__ == "__main__":
    main()
