"""会場プロファイル(スコアボードの位置と数字の見本、チーム色、校正済みキーフレーム)の読み書き。"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Keyframe, load_keyframe_features

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


@dataclass
class Team:
    id: str
    name: str
    color: tuple[int, int, int] | None = None   # RGB。チーム識別のクラスタをチームに対応付けるのに使う


@dataclass
class Profile:
    name: str
    frame_size: tuple[int, int]
    scoreboard: dict[str, list[int]]     # "score_a" / "score_b" / "clock" → [x0, y0, x1, y1]
    teams: list[Team]
    keyframes: list[Keyframe] = field(default_factory=list)
    path: Path | None = None
    digits: object | None = None          # スコアボードの数字の見本(wbt.digits.DigitReader)


def load_profile(name_or_path: str | Path) -> Profile:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROFILES_DIR / f"{p}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    teams = [Team(k, v["name"], tuple(v["color"]) if v.get("color") else None) for k, v in d["teams"].items()]
    kfs = []
    for k in d.get("keyframes", []):
        H = np.float64(k["H_court"])   # scripts/calibrate_keyframe.py で推定したコート→画像
        # キーフレームが写すゴールの側: 画像中心をコート座標に写した X の符号で決める
        w, h = d["frame_size"]
        cx = cv2.perspectiveTransform(np.float64([[[w / 2, h / 2]]]), np.linalg.inv(H))[0, 0, 0]
        side = 1 if cx > 0 else -1
        kfs.append(Keyframe(k["name"], H, np.float64(k["rims"]),
                            load_keyframe_features(p.parent / k["features"]), side))
    digits = None
    if d.get("digits"):   # scripts/build_digit_templates.py で作成
        from .digits import DigitReader
        digits = DigitReader.load(p.parent / d["digits"])
    return Profile(d["name"], tuple(d["frame_size"]), d["scoreboard"], teams, kfs, p, digits)


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))
