"""データセット作成の各工程をまとめたコマンド。

1. 広角ショットのフレーム抽出と下書きラベル(既製モデル)
    python scripts/prepare_dataset.py extract <video> --game jpn_pol --out data/dataset/raw
2. CVAT に読み込む zip の作成(ランダムに N 枚選ぶ)
    python scripts/prepare_dataset.py cvat-export data/dataset/raw --n 300 --zip data/dataset/cvat_batch1.zip
   → CVAT でタスクを作成して画像をアップロードし、「Upload annotations」で YOLO 1.1 形式の zip を読み込んで修正する
3. CVAT で修正したラベルの取り込み(「Export task dataset」→ YOLO 1.1)
    python scripts/prepare_dataset.py cvat-import data/dataset/cvat_batch1_done.zip --images data/dataset/raw/images \
        --out data/dataset/annotated
4. 試合単位の分割と data.yaml の作成
    python scripts/prepare_dataset.py split data/dataset/annotated --val jpn_gbr --test jpn_pol --out data/dataset/yolo
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from wbt.dataset import export_cvat_yolo, extract_candidates, import_cvat_yolo, sample_for_annotation, split_by_game


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("video", type=Path)
    e.add_argument("--game", required=True, help="試合の識別名(ファイル名の先頭になり、分割に使う)")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--every-sec", type=float, default=2.0)
    e.add_argument("--weights", default="yolov8s.pt")
    c = sub.add_parser("cvat-export")
    c.add_argument("src", type=Path)
    c.add_argument("--n", type=int, help="選ぶ枚数(省略時は全部)")
    c.add_argument("--zip", type=Path, required=True)
    i = sub.add_parser("cvat-import")
    i.add_argument("zip", type=Path)
    i.add_argument("--images", type=Path, required=True)
    i.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("split")
    s.add_argument("src", type=Path)
    s.add_argument("--val", nargs="*", default=[])
    s.add_argument("--test", nargs="*", default=[])
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "extract":
        n = extract_candidates(args.video, args.out, args.game, args.every_sec, weights=args.weights)
        print(f"{n} frames -> {args.out}")
    elif args.cmd == "cvat-export":
        src = args.src
        if args.n:
            chosen = sample_for_annotation(src, args.n)
            tmp = Path(tempfile.mkdtemp())
            for sub_dir, ext in (("images", ".jpg"), ("labels", ".txt")):
                (tmp / sub_dir).mkdir()
                for img in chosen:
                    p = src / sub_dir / f"{img.stem}{ext}"
                    if p.exists():
                        shutil.copy(p, tmp / sub_dir / p.name)
            src = tmp
        n = export_cvat_yolo(src, args.zip)
        img_dir = args.zip.with_suffix("")
        img_dir.mkdir(parents=True, exist_ok=True)
        for p in (src / "images").glob("*.jpg"):
            shutil.copy(p, img_dir / p.name)
        print(f"{n} images: annotations -> {args.zip}, images to upload -> {img_dir}")
    elif args.cmd == "cvat-import":
        print(f"{import_cvat_yolo(args.zip, args.images, args.out)} labeled images -> {args.out}")
    elif args.cmd == "split":
        print(json.dumps(split_by_game(args.src, args.out, args.val, args.test)))


if __name__ == "__main__":
    main()
