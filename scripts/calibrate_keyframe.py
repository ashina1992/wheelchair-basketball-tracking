"""会場プロファイルにキーフレーム(コート校正済みの基準フレーム)を追加する。

校正点は [img_x, img_y, court_X, court_Y] のリストで与える。X か Y の一方を null にすると、
その点がコート上の直線(例: [100, 358, null, -7.5] は奥のサイドライン上)にあるという拘束になる。
完全な対応点(X, Y とも既知)が4点以上必要。
  --points で JSON を渡すか、--interactive でウィンドウ上をクリックして入力する
リング中心の画像座標は --rims で渡す(写っていなければ省略)。

使い方:
    python scripts/calibrate_keyframe.py profiles/ottawa2026.json data/frames/jpn_pol/frame_000870.jpg \
        --name left --points "[[1190,611,-8.2,-2.45],[1024,830,-8.2,2.45],[451,548,-14,-2.45],[211,759,-14,2.45]]" \
        --rims "[[544,255]]" --preview runs/calib/left.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from wbt.calibration import fit_homography
from wbt.camera import FeatureExtractor, save_keyframe_features
from wbt.court import court_polylines
from wbt.detect import Detector


def click_points(img):
    pts = []
    view = img.copy()

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            raw = input(f"({x}, {y}) のコート座標 X Y [m](直線上の点なら片方を - ): ").split()
            X, Y = (None if v == "-" else float(v) for v in raw)
            pts.append([x, y, X, Y])
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calib", on_mouse)
    print("コート上の既知点をクリック。終わったら q")
    while cv2.waitKey(30) != ord("q"):
        cv2.imshow("calib", view)
    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("frame", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", help="[[img_x, img_y, court_X, court_Y], ...](X/Y の一方は null 可)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--rims", default="[]", help="[[x, y], ...] キーフレーム上のリング中心")
    ap.add_argument("--preview", type=Path, help="コートラインを重ねた確認画像の保存先")
    args = ap.parse_args()

    img = cv2.imread(str(args.frame))
    pts = click_points(img) if args.interactive else json.loads(args.points)
    H, summary = fit_homography(pts)
    print(f"{args.name}: {json.dumps(summary, ensure_ascii=False)}")

    prof = json.loads(args.profile.read_text(encoding="utf-8"))
    # 特徴点は選手とスコアボードを除いて抽出する(動く物体・重畳表示に引きずられないように)
    exclude = [tuple(b) for b in Detector()(img).players.xyxy] + [tuple(v) for v in prof["scoreboard"].values()]
    feats = FeatureExtractor().extract(img, exclude)
    feat_file = f"{args.profile.stem}_{args.name}.npz"
    save_keyframe_features(args.profile.parent / feat_file, feats)
    print(f"{args.name}: 特徴点 {len(feats.pts)} 個 -> {feat_file}")

    kf = {"name": args.name, "source_frame": args.frame.as_posix(), "points": pts,
          "rims": json.loads(args.rims), "features": feat_file,
          "H_court": (H / H[2, 2]).round(8).tolist(), "calibration": summary}
    prof["keyframes"] = [k for k in prof.get("keyframes", []) if k["name"] != args.name] + [kf]
    args.profile.write_text(json.dumps(prof, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        for pl in court_polylines():
            p = cv2.perspectiveTransform(pl.reshape(-1, 1, 2), H).reshape(-1, 2)
            cv2.polylines(img, [np.int32(p)], False, (0, 255, 255), 2)
        for x, y, *_ in pts:
            cv2.circle(img, (int(x), int(y)), 6, (255, 0, 255), -1)
        for x, y in json.loads(args.rims):
            cv2.circle(img, (int(x), int(y)), 12, (0, 0, 255), 2)
        cv2.imwrite(str(args.preview), img)


if __name__ == "__main__":
    main()
