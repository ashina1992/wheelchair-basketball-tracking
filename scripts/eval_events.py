"""解析結果のイベントを正解と比べ、種別ごとの適合率・再現率を出す。

正解ファイル(JSON)の形式:
    {"video": "...", "events": [{"t": 23.0, "type": "made_shot", "team": "a"}, ...]}
    t は動画の先頭からの秒。type は analysis.json と同じ(made_shot / missed_shot / rebound / assist / turnover)。
    公式のプレーバイプレーからは scripts/pbp_to_gt.py で作る。

--auto-offset を付けると、正解の時刻の一定のずれ(中継の時計表示の遅れ)を推定して補正してから比べる。

使い方:
    python scripts/eval_events.py runs/analysis/usa_jpn/analysis.json data/gt/usa_jpn_events.json --tol 2 --auto-offset
"""
import argparse
import json
from pathlib import Path

from wbt.evaluate import estimate_offset, match_events
from wbt.gt import EVAL_TYPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--tol", type=float, default=2.0, help="同じイベントとみなす時刻差(秒)")
    ap.add_argument("--offset", type=float, default=0.0, help="正解の時刻に足すずれ(秒)")
    ap.add_argument("--auto-offset", action="store_true", help="ずれを推定して補正する")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    pred = json.loads(args.analysis.read_text(encoding="utf-8"))["events"]
    gt = json.loads(args.gt.read_text(encoding="utf-8"))["events"]
    offset = estimate_offset(pred, gt, args.tol, EVAL_TYPES) if args.auto_offset else args.offset
    gt = [{**e, "t": e["t"] + offset} for e in gt]
    rep = match_events(pred, gt, args.tol, EVAL_TYPES)
    print(f"正解の時刻の補正: {offset:+.1f} 秒、時刻差の許容: ±{args.tol} 秒")
    for typ, m in rep.items():
        p = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        r = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        print(f"{typ:12s} P={p} R={r} (tp={m['tp']}, pred={m['n_pred']}, gt={m['n_gt']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"offset": offset, "tol": args.tol, "report": rep}, indent=2,
                                       ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
