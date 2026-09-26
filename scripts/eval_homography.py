"""校正に使っていないフレームで、コート座標変換の誤差を測る。

各フレームをキーフレームに照合し(wbt.camera)、既知点のコート座標を画像に投影した位置と、
目視で読み取った位置との差(画素)と、画像点をコートに逆投影したときの差(m)を出す。

使い方:
    python scripts/eval_homography.py docs/results/homography_checkpoints.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from wbt.camera import CameraTracker
from wbt.detect import Detector
from wbt.profiles import load_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path)
    args = ap.parse_args()
    spec = json.loads(args.checkpoints.read_text(encoding="utf-8"))
    prof = load_profile(spec["profile"])
    det = Detector()
    px_all, m_all = [], []
    for f in spec["frames"]:
        img = cv2.imread(f["frame"])
        cam = CameraTracker(prof.keyframes)
        exclude = [tuple(b) for b in det(img).players.xyxy] + [tuple(prof.scoreboard["overlay"])]
        st = cam.update(img, exclude=exclude, cut=True)
        if not st.valid:
            print(f"{f['frame']}: キーフレームに照合できませんでした")
            continue
        pts = np.float64(f["points"])
        proj = cam.court_to_img(st, pts[:, 2:])
        back = cam.img_to_court(st, pts[:, :2])
        px = np.linalg.norm(proj - pts[:, :2], axis=1)
        m = np.linalg.norm(back - pts[:, 2:], axis=1)
        px_all += px.tolist()
        m_all += m.tolist()
        print(f"{Path(f['frame']).name}: keyframe={prof.keyframes[st.keyframe].name} inliers={st.inliers} "
              f"画素誤差 平均 {px.mean():.1f} 最大 {px.max():.1f} / コート上 平均 {m.mean():.2f}m 最大 {m.max():.2f}m")
    if px_all:
        print(f"全体({len(px_all)}点): 画素誤差 平均 {np.mean(px_all):.1f}px 最大 {np.max(px_all):.1f}px / "
              f"コート上 平均 {np.mean(m_all):.2f}m 最大 {np.max(m_all):.2f}m")


if __name__ == "__main__":
    main()
