"""物体検出。モデルごとのクラスを player / ball / rim / referee に正規化する。

既製の COCO モデル(person, sports ball)と、自前で学習したモデル(player, ball, rim, referee)を
同じインターフェースで差し替えられるようにする。
"""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

CLASSES = ("player", "ball", "rim", "referee")
COCO_MAP = {"person": "player", "sports ball": "ball"}
DEFAULT_CONF = {"player": 0.10, "ball": 0.10, "rim": 0.30, "referee": 0.30}  # player は ByteTrack の2段目照合用に低めで残す


@dataclass
class Detections:
    players: Boxes          # ByteTrack にそのまま渡せる形式(numpy)
    balls: np.ndarray       # (N, 5) x1, y1, x2, y2, conf
    rims: np.ndarray        # (N, 5)
    referees: np.ndarray    # (N, 5)


class Detector:
    def __init__(self, weights: str = "yolov8s.pt", imgsz: int = 1280, conf: dict[str, float] | None = None):
        self.model = YOLO(weights)
        self.weights = weights
        self.imgsz = imgsz
        self.conf = {**DEFAULT_CONF, **(conf or {})}
        names = self.model.names
        if "player" in names.values():
            mapping = {n: n for n in CLASSES}
        else:
            mapping = COCO_MAP
        # モデルのクラスID → 正規化後のクラス名
        self.id_map = {i: mapping[n] for i, n in names.items() if n in mapping}
        self.available = set(self.id_map.values())

    def __call__(self, frame: np.ndarray) -> Detections:
        r = self.model.predict(frame, classes=list(self.id_map), conf=min(self.conf.values()),
                               imgsz=self.imgsz, verbose=False)[0]
        boxes = r.boxes.cpu().numpy()
        names = np.array([self.id_map[int(c)] for c in boxes.cls]) if len(boxes) else np.array([], dtype=str)
        thr = np.array([self.conf[n] for n in names]) if len(names) else np.array([])
        keep = boxes.conf >= thr if len(names) else np.zeros(0, bool)

        def arr(name):
            m = keep & (names == name)
            return np.hstack([boxes.xyxy[m], boxes.conf[m, None]]) if m.any() else np.zeros((0, 5), np.float32)

        players = boxes[keep & (names == "player")]
        return Detections(players=players[suppress_contained(players.xyxy, players.conf)], balls=arr("ball"),
                          rims=arr("rim"), referees=arr("referee"))


def suppress_contained(xyxy: np.ndarray, conf: np.ndarray, ratio: float = 0.8, min_conf: float = 0.3) -> np.ndarray:
    """ほぼ全体が別の(より大きく、信頼度が十分な)箱の内側にある箱を除くマスクを返す。

    COCO の person は、車いすの選手に対して「上半身だけの箱」と「車いすまで含む箱」を
    重ねて出すことがあり、同じ選手に2つのトラックが付く原因になる。
    """
    n = len(xyxy)
    keep = np.ones(n, bool)
    if n < 2:
        return keep
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for i in range(n):
        for j in range(n):
            if i == j or area[j] <= area[i] or conf[j] < min_conf:
                continue
            iw = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
            ih = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            if iw > 0 and ih > 0 and iw * ih / max(area[i], 1e-9) > ratio:
                keep[i] = False
                break
    return keep
