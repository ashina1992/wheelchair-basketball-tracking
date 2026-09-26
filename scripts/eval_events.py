"""解析結果のイベントを正解と比べ、種別ごとの適合率・再現率を出す。

正解ファイル(JSON)の形式:
    {"video": "...", "events": [{"t": 23.0, "type": "made_shot", "team": "a"}, ...]}
    t は動画の先頭からの秒。type は analysis.json と同じ(made_shot / missed_shot / free_throw / rebound /
    assist / turnover)。公式のプレーバイプレーから作る場合は、試合時計を動画の時刻に対応付けて変換する。

使い方:
    python scripts/eval_events.py runs/analysis/jpn_pol/analysis.json data/gt/jpn_pol_events.json
"""
import argparse
import json
from pathlib import Path

from wbt.evaluate import match_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--tol", type=float, default=2.0, help="同じイベントとみなす時刻差(秒)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    pred = json.loads(args.analysis.read_text(encoding="utf-8"))["events"]
    gt = json.loads(args.gt.read_text(encoding="utf-8"))["events"]
    rep = match_events(pred, gt, args.tol)
    for typ, m in rep.items():
        p = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        r = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        print(f"{typ:12s} P={p} R={r} (tp={m['tp']}, pred={m['n_pred']}, gt={m['n_gt']})")
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
