"""解析済みの結果に対して、スコアボードの読み取りとイベント判定だけをやり直す(検出・追跡はやり直さない)。

判定ルールやスコアボードの読み取り方法を変えたとき、1時間近くかかる検出をやり直さずに結果を更新できる。

使い方:
    python scripts/rebuild_events.py runs/analysis/usa_jpn/analysis.json data/videos/USA_v_Japan_men_1.mp4
"""
import argparse
import json
from pathlib import Path

from wbt.pipeline import finish_events, read_scoreboard_series, save_analysis
from wbt.profiles import load_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, help="出力先のフォルダ(省略時は analysis.json と同じ場所に上書き)")
    args = ap.parse_args()

    result = json.loads(args.analysis.read_text(encoding="utf-8"))
    prof = load_profile(result["meta"]["profile"])
    readings = read_scoreboard_series(args.video, prof, result["meta"]["crop"])
    finish_events(result, readings)
    save_analysis(result, args.out or args.analysis.parent)
    print(json.dumps(result["stats"], ensure_ascii=False))
    print(f"events: {len(result['events'])}, shot clock resets: {len(result['shot_clock_resets'])}")


if __name__ == "__main__":
    main()
