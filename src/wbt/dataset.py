"""データセット作成: 広角ショットのフレーム抽出、既製モデルによる下書きラベル、試合単位の分割。

ラベルは YOLO 形式(1行 = "class cx cy w h"、座標は画像サイズで正規化)。
CVAT には「YOLO 1.1」形式として読み込み・書き出しできる構成(obj.names / obj.data / train.txt /
obj_train_data/)で出力する。
"""
import random
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .detect import CLASSES, Detector
from .shots import is_wide_shot
from .teams import STANDING_ASPECT
from .video import detect_content_box

CLASS_ID = {n: i for i, n in enumerate(CLASSES)}   # player=0, ball=1, rim=2, referee=3


def to_yolo_lines(boxes: list[tuple[str, np.ndarray]], w: int, h: int) -> list[str]:
    lines = []
    for name, (x1, y1, x2, y2) in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASS_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def read_yolo_labels(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append((int(c), np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])))
    return out


RIM_BOX_PX = (80.0, 24.0)   # キーフレーム上でのリングの見かけの大きさ(幅, 高さ、1920x1080 基準)
COURT_MARGIN = 1.0          # コート外周からこの距離(m)以内の人物だけを下書きに残す


def prelabel(det, frame: np.ndarray, profile=None) -> list[tuple[str, np.ndarray]]:
    """検出結果を下書きラベルにする(アノテーターが修正する前提)。

    - 立っている人物(箱が縦長)は referee とする
    - profile(会場プロファイル)を渡し、フレームがその会場のキーフレームに照合できた場合は、
      コート外の人物(ベンチ・観客)を除き、リングの箱を校正結果から入れる
    """
    h, w = frame.shape[:2]
    court_ok = lambda b: True   # noqa: E731
    cam_rims: list[np.ndarray] = []
    if profile is not None and profile.keyframes:
        from .camera import CameraTracker
        from .court import court_outline

        s = profile.frame_size[0] / w   # 会場プロファイルの座標系(例: 1920 幅)への倍率
        small = cv2.resize(frame, tuple(profile.frame_size)) if s != 1 else frame
        cam = CameraTracker(profile.keyframes)
        exclude = [tuple(b * s) for b in det.players.xyxy] + [tuple(v) for v in profile.scoreboard.values()]
        st = cam.update(small, exclude=exclude, cut=True)
        if st.valid:
            outline = np.float32(court_outline(COURT_MARGIN))

            def court_ok(b):   # noqa: F811
                foot = np.array([[(b[0] + b[2]) / 2 * s, b[3] * s]])
                X, Y = cam.img_to_court(st, foot)[0]
                return cv2.pointPolygonTest(outline, (float(X), float(Y)), False) >= 0

            if not len(det.rims):
                zoom = cam.zoom(st)
                rw, rh = RIM_BOX_PX[0] * zoom / s, RIM_BOX_PX[1] * zoom / s
                for cx, cy in cam.rims_in_frame(st)[0] / s:
                    if 0 <= cx < w and 0 <= cy < h:
                        cam_rims.append(np.array([cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2]))

    boxes = []
    for b, c in zip(det.players.xyxy, det.players.conf):
        if c < 0.3 or not court_ok(b):
            continue
        aspect = (b[3] - b[1]) / max(b[2] - b[0], 1.0)
        boxes.append(("referee" if aspect > STANDING_ASPECT else "player", b))
    for b in det.balls:
        if b[4] >= 0.3:
            boxes.append(("ball", b[:4]))
    for b in det.rims:
        boxes.append(("rim", b[:4]))
    boxes += [("rim", b) for b in cam_rims]
    for b in det.referees:
        if court_ok(b[:4]):
            boxes.append(("referee", b[:4]))
    return boxes


def relabel(images: list[Path], labels_dir: Path, weights: str, profile=None, progress=None) -> int:
    """images の下書きラベルを、指定したモデル(と会場プロファイル)で作り直して labels_dir に書く。"""
    det = Detector(weights)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(images):
        img = cv2.imread(str(p))
        lines = to_yolo_lines(prelabel(det(img), img, profile), img.shape[1], img.shape[0])
        (labels_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if progress:
            progress(k + 1, len(images))
    return len(images)


def extract_candidates(video: Path, out_dir: Path, game: str, every_sec: float = 2.0,
                       detector: Detector | None = None, weights: str = "yolov8s.pt") -> int:
    """広角ショットのフレームを every_sec 間隔で抜き出し、画像と下書きラベルを保存する。

    out_dir/images/<game>_<frame>.jpg、out_dir/labels/<game>_<frame>.txt を作る。保存枚数を返す。
    """
    detector = detector or Detector(weights)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = detect_content_box(cap)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for idx in range(0, total, max(1, round(fps * every_sec))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[y0:y1, x0:x1]
        det = detector(frame)
        if not is_wide_shot(det.players.xyxy, frame.shape[0], det.players.conf):
            continue
        stem = f"{game}_{idx:06d}"
        cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = to_yolo_lines(prelabel(det, frame), frame.shape[1], frame.shape[0])
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    cap.release()
    return n


def export_cvat_yolo(src_dir: Path, zip_path: Path):
    """images/ と labels/ から、CVAT の「YOLO 1.1」形式でインポートできる zip を作る。"""
    imgs = sorted((src_dir / "images").glob("*.jpg"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASSES)}\ntrain = data/train.txt\nnames = data/obj.names\n"
                               "backup = backup/\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{p.name}" for p in imgs) + "\n")
        for p in imgs:
            lab = src_dir / "labels" / f"{p.stem}.txt"
            z.writestr(f"obj_train_data/{p.stem}.txt", lab.read_text() if lab.exists() else "")
    return len(imgs)


def import_cvat_yolo(zip_path: Path, images_dir: Path, out_dir: Path) -> int:
    """CVAT から「YOLO 1.1」で書き出した zip のラベルを、画像と対にして out_dir/images, labels に置く。"""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        names = z.read("obj.names").decode().split()
        if names != list(CLASSES):
            raise ValueError(f"クラス定義が一致しません: {names}")
        for info in z.infolist():
            p = Path(info.filename)
            if p.suffix != ".txt" or p.parent.name != "obj_train_data":
                continue
            img = images_dir / f"{p.stem}.jpg"
            if not img.exists():
                continue
            shutil.copy(img, out_dir / "images" / img.name)
            (out_dir / "labels" / p.name).write_bytes(z.read(info))
            n += 1
    return n


def _matches(game: str, names: list[str]) -> bool:
    """game が names のどれかと一致するか、その後に "_<番号>" が付いた名前か。

    1試合を前半・後半など複数ファイルに分けて録画した場合(Japan_v_Poland_men_1, _2)も、
    "Japan_v_Poland_men" と指定すれば同じ試合としてまとめて扱える。
    """
    return any(game == n or (game.startswith(n + "_") and game[len(n) + 1:].isdigit()) for n in names)


def split_by_game(src_dir: Path, out_dir: Path, val_games: list[str], test_games: list[str]) -> dict:
    """ファイル名の先頭(<game>_)で試合を判別し、試合単位で train/val/test に分けて data.yaml を作る。"""
    counts = {"train": 0, "val": 0, "test": 0}
    for img in sorted((src_dir / "images").glob("*.jpg")):
        game = img.stem.rsplit("_", 1)[0]
        split = "test" if _matches(game, test_games) else "val" if _matches(game, val_games) else "train"
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = src_dir / sub / f"{img.stem}{ext}"
            dst = out_dir / sub / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy(src, dst)
        counts[split] += 1
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n", encoding="utf-8")
    return counts


def sample_for_annotation(src_dir: Path, n: int, seed: int = 0, exclude: set[str] = frozenset()) -> list[Path]:
    """アノテーション対象を n 枚選ぶ(試合ごとに均等になるように)。

    前半・後半など複数ファイルに分けて録画した試合(<game>_1, <game>_2)は1試合として数える。
    exclude に含まれるファイル名(拡張子なし)は選ばない(前の回ですでにアノテーションした画像)。
    """
    by_game: dict[str, list[Path]] = {}
    for img in sorted((src_dir / "images").glob("*.jpg")):
        if img.stem in exclude:
            continue
        game = re.sub(r"_\d+$", "", img.stem.rsplit("_", 1)[0])
        by_game.setdefault(game, []).append(img)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_game)))
    out = []
    for imgs in by_game.values():
        out += rng.sample(imgs, min(per, len(imgs)))
    return out
