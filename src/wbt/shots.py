"""広角の試合ショットかどうかの簡易判定(会場プロファイルがない映像用)。

解析パイプラインでは、会場のキーフレームに照合できたかどうかで広角ショットを判定する(wbt.camera)。
データセット作成のように、まだ校正していない会場の映像を扱うときはこちらの簡易判定を使う。
広角ショットでは人物が小さく多数写り、サイドラインカメラやクローズアップでは手前の人物が大きく写る。
観客も小さく写るため、中央値ではなく「最も大きい人物」の大きさで判定する。
"""
import numpy as np

MIN_PEOPLE = 6
MAX_HEIGHT = 0.3   # 信頼度の高い人物の箱の高さの最大値(画像の高さに対する比)
# Japan v Poland(42枚)での確認: 広角25枚はすべて 0.17〜0.27、それ以外は1枚を除き 0.39 以上


def is_wide_shot(person_boxes: np.ndarray, frame_height: int, conf: np.ndarray | None = None,
                 min_conf: float = 0.5) -> bool:
    boxes = np.asarray(person_boxes).reshape(-1, 4)
    if conf is not None:
        boxes = boxes[np.asarray(conf) >= min_conf]
    if len(boxes) < MIN_PEOPLE:
        return False
    return float(np.max(boxes[:, 3] - boxes[:, 1])) / frame_height <= MAX_HEIGHT
