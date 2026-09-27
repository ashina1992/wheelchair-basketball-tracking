"""データセット作成の各工程をまとめたコマンド。

1. 広角ショットのフレーム抽出と下書きラベル(既製モデル)
    python scripts/prepare_dataset.py extract <video> --game Japan_v_Poland_men_1 --out data/dataset/raw
2. CVAT に読み込む zip の作成(試合ごとに均等に N 枚選び、下書きラベルを作り直す)
    python scripts/prepare_dataset.py cvat-export data/dataset/raw --n 100 --zip data/dataset/cvat_batch1.zip \
        --weights yolov8x.pt --profile ottawa2026
   2回目以降は、学習したモデルで下書きを作り、前の回の画像を除く:
    python scripts/prepare_dataset.py cvat-export data/dataset/raw --n 200 --zip data/dataset/cvat_batch2.zip \
        --weights models/batch1.pt --profile ottawa2026 --exclude data/dataset/cvat_batch1
   → CVAT でタスクを作成して画像をアップロードし、「Upload annotations」で YOLO 1.1 形式の zip を読み込んで修正する
3. CVAT で修正したラベルの取り込み(「Export task dataset」→ YOLO 1.1)
    python scripts/prepare_dataset.py cvat-import data/dataset/cvat_batch1_done.zip --images data/dataset/raw/images \
        --out data/dataset/annotated
4. 試合単位の分割と data.yaml の作成(前半・後半に分けた録画は試合名の先頭部分で指定する)
    python scripts/prepare_dataset.py split data/dataset/annotated --val Germany_v_Japan_women --test USA_v_Japan_men \
        --out data/dataset/yolo
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from wbt.dataset import (export_cvat_yolo, extract_candidates, import_cvat_yolo, relabel, sample_for_annotation,
                         split_by_game)
from wbt.profiles import load_profile


def cvat_export(args):
    exclude = {p.stem for d in args.exclude for p in Path(d).glob("*.jpg")}
    chosen = sample_for_annotation(args.src, args.n, exclude=exclude) if args.n else \
        [p for p in sorted((args.src / "images").glob("*.jpg")) if p.stem not in exclude]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "images").mkdir()
    for img in chosen:
        shutil.copy(img, tmp / "images" / img.name)
    if args.weights:
        profile = load_profile(args.profile) if args.profile else None
        print(f"下書きラベルを作成中({args.weights}, {len(chosen)} 枚)…")
        relabel(chosen, tmp / "labels", args.weights, profile,
                progress=lambda k, n: print(f"  {k}/{n}", flush=True) if k % 10 == 0 or k == n else None)
    else:
        (tmp / "labels").mkdir()
        for img in chosen:
            lab = args.src / "labels" / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy(lab, tmp / "labels" / lab.name)
    n = export_cvat_yolo(tmp, args.zip)
    img_dir = args.zip.with_suffix("")
    img_dir.mkdir(parents=True, exist_ok=True)
    for p in (tmp / "images").glob("*.jpg"):
        shutil.copy(p, img_dir / p.name)
    print(f"{n} images: annotations -> {args.zip}, images to upload -> {img_dir}")


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
    c.add_argument("--weights", help="下書きラベルをこのモデルで作り直す(省略時は extract 時の下書きを使う)")
    c.add_argument("--profile", help="会場プロファイル。照合できたフレームはコート外の人物を除き、リングを自動で入れる")
    c.add_argument("--exclude", nargs="*", default=[], help="前の回に書き出した画像フォルダ(そこにある画像は選ばない)")
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
        cvat_export(args)
    elif args.cmd == "cvat-import":
        print(f"{import_cvat_yolo(args.zip, args.images, args.out)} labeled images -> {args.out}")
    elif args.cmd == "split":
        print(json.dumps(split_by_game(args.src, args.out, args.val, args.test)))


if __name__ == "__main__":
    main()
