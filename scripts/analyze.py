"""試合動画を解析し、analysis.json・stats.csv・events.csv を出力する。

使い方:
    python scripts/analyze.py <video> --profile ottawa2026 --out runs/analysis/jpn_pol
"""
import argparse
import json
from pathlib import Path

from wbt.pipeline import analyze, save_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--profile", required=True, help="profiles/ 以下のプロファイル名、または JSON のパス")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolov8s.pt", help="検出モデル(COCO 学習済み、または自前学習の重み)")
    ap.add_argument("--fps", type=float, default=10.0, help="解析するフレームレート")
    ap.add_argument("--max-seconds", type=float, help="先頭から指定秒数だけ解析する(動作確認用)")
    args = ap.parse_args()

    last = {"pct": -1}

    def progress(stage, frac):
        pct = int(frac * 100)
        if pct != last["pct"] and pct % 5 == 0:
            print(f"[{stage}] {pct}%", flush=True)
            last["pct"] = pct

    result = analyze(args.video, args.profile, args.weights, args.fps, args.max_seconds, progress)
    save_analysis(result, args.out)
    print(json.dumps(result["stats"], ensure_ascii=False))
    print(f"events: {len(result['events'])}, processing: {result['meta']['processing_sec']}s -> {args.out}")


if __name__ == "__main__":
    main()
