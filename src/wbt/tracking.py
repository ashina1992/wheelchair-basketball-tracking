"""選手トラッキング(ByteTrack)とボール軌跡の推定。"""
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .detect import Detections


class PlayerTracker:
    """ByteTrack のラッパー。ショットが切り替わったら reset() でトラックを捨てる。"""

    def __init__(self, track_buffer: int = 30):
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        cfg.track_buffer = track_buffer
        self.cfg = cfg
        self.tracker = BYTETracker(cfg)
        self._id_offset = 0   # reset 後も ID が重複しないようにずらす

    def reset(self):
        self._id_offset += 10000
        self.tracker = BYTETracker(self.cfg)

    def update(self, det: Detections, frame: np.ndarray) -> np.ndarray:
        """(N, 6) の配列 [x1, y1, x2, y2, track_id, conf] を返す。"""
        if len(det.players) == 0:
            self.tracker.update(det.players, frame)
            return np.zeros((0, 6), np.float32)
        out = self.tracker.update(det.players, frame)
        if len(out) == 0:
            return np.zeros((0, 6), np.float32)
        res = out[:, [0, 1, 2, 3, 4, 5]].astype(np.float64)
        res[:, 4] += self._id_offset
        return res


class BallTracker:
    """フレームごとのボール候補から、軌跡として一貫する1つを選ぶ。

    直前の位置と速度から予測した位置に近い候補を優先し、見つからなければ
    信頼度の高い候補から軌跡を始め直す。途切れた区間は interpolate() で後から線形補間する。
    """

    def __init__(self, gate: float = 150.0, restart_conf: float = 0.3, max_missing: int = 5):
        self.gate = gate
        self.restart_conf = restart_conf
        self.max_missing = max_missing
        self.reset()

    def reset(self):
        self.pos = None
        self.vel = np.zeros(2)
        self.missing = 0

    def update(self, balls: np.ndarray, zoom: float = 1.0) -> tuple[np.ndarray, float] | None:
        cands = [(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), float(b[4])) for b in balls]
        choice = None
        if self.pos is not None:
            pred = self.pos + self.vel * (self.missing + 1)
            gate = self.gate * zoom * (1 + self.missing)
            near = [(np.linalg.norm(c - pred), c, s) for c, s in cands]
            near = [x for x in near if x[0] < gate]
            if near:
                _, c, s = min(near, key=lambda x: x[0] - 100 * x[2])
                choice = (c, s)
        if choice is None:
            strong = [(c, s) for c, s in cands if s >= self.restart_conf]
            if strong:
                choice = max(strong, key=lambda x: x[1])
                self.vel = np.zeros(2)
                self.pos = None
        if choice is None:
            self.missing += 1
            if self.missing > self.max_missing:
                self.reset()
            return None
        c, s = choice
        if self.pos is not None:
            self.vel = 0.5 * self.vel + 0.5 * (c - self.pos) / (self.missing + 1)
        self.pos, self.missing = c, 0
        return c, s


def interpolate_ball(frames: list[dict], max_gap: int = 8):
    """frames[i]["ball"] が None の区間を、同じショット内で max_gap フレーム以内なら線形補間する。"""
    last = None
    for i, fr in enumerate(frames):
        if not fr["wide"]:
            last = None
            continue
        if fr["ball"] is None:
            continue
        if last is not None and 1 < i - last <= max_gap + 1 and fr["segment"] == frames[last]["segment"]:
            a, b = np.array(frames[last]["ball"]["xy"]), np.array(fr["ball"]["xy"])
            for j in range(last + 1, i):
                w = (j - last) / (i - last)
                frames[j]["ball"] = {"xy": (a + (b - a) * w).round(1).tolist(), "conf": 0.0, "interpolated": True}
        last = i
