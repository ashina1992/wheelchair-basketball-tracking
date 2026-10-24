"""公式のプレーバイプレー(FIBA LiveStats の data.json)を、解析結果と比べられる正解イベントに変換する。

data.json の取得(IWBF の公式記録は Genius Sports / FIBA LiveStats で公開されている):
    https://fibalivestats.dcd.shared.geniussports.com/data/<試合ID>/data.json
    試合ID は https://hosted.dcd.shared.geniussports.com/IWBF/en/competition/<大会ID>/match/<試合ID>/summary の番号

使い方:
    python scripts/pbp_to_gt.py data/gt/raw/2901510_data.json runs/analysis/usa_jpn/analysis.json --period 2 \
        --out data/gt/usa_jpn_events.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from wbt.gt import events_from_livestats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("livestats", type=Path)
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--period", type=int, required=True, help="動画が映しているピリオド")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ls = json.loads(args.livestats.read_text(encoding="utf-8"))
    an = json.loads(args.analysis.read_text(encoding="utf-8"))
    gt = events_from_livestats(ls, an, args.period)
    gt["video"] = an["meta"]["video"]
    gt["source"] = f"FIBA LiveStats {args.livestats.name}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"チームの対応: {gt['team_map']}、試合時計の範囲: {gt['clock_window']}")
    print(f"{len(gt['events'])} 件 -> {args.out}")
    print(dict(Counter((e['type'], e['team']) for e in gt["events"])))


if __name__ == "__main__":
    main()
