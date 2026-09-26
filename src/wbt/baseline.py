"""既製YOLO(COCO学習済み)によるベースライン検出。"""
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class id -> 名前
PERSON, SPORTS_BALL = 0, 32
DEFAULT_CLASS_CONF = {PERSON: 0.25, SPORTS_BALL: 0.10}  # ボールは小さく信頼度が低めに出るため閾値を下げる


def run_baseline(model_name: str, frames: list[Path], out_dir: Path,
                 class_conf: dict[int, float] = DEFAULT_CLASS_CONF, imgsz: int = 1280) -> dict:
    """frames に対して検出を行い、CSV・可視化画像・summary.json を out_dir/<model> に保存する。"""
    model = YOLO(model_name)
    out_dir = out_dir / Path(model_name).stem
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    classes = sorted(class_conf)

    counts = {c: [] for c in classes}
    confs = {c: [] for c in classes}
    times = []
    with open(out_dir / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cls", "x1", "y1", "x2", "y2", "conf"])
        for p in frames:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            r = model.predict(img, classes=classes, conf=min(class_conf.values()), imgsz=imgsz, verbose=False)[0]
            times.append(time.perf_counter() - t0)
            keep = [i for i, (c, s) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()))
                    if s >= class_conf[int(c)]]
            r = r[keep]
            n = {c: 0 for c in classes}
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                c = int(c)
                w.writerow([p.name, model.names[c], round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(s, 3)])
                n[c] += 1
                confs[c].append(s)
            for c in classes:
                counts[c].append(n[c])
            cv2.imwrite(str(out_dir / "vis" / p.name), r.plot(line_width=2, font_size=12))

    times = times[1:] or times  # 初回はウォームアップを含むので除外
    summary = {
        "model": model_name,
        "imgsz": imgsz,
        "n_frames": len(frames),
        "cpu_ms_per_frame": float(np.mean(times) * 1000),
        "classes": {
            model.names[c]: {
                "conf_thresh": class_conf[c],
                "per_frame_mean": float(np.mean(counts[c])),
                "per_frame_max": int(np.max(counts[c])),
                "frames_with_detection": int(np.count_nonzero(counts[c])),
                "conf_mean": float(np.mean(confs[c])) if confs[c] else None,
            }
            for c in classes
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
