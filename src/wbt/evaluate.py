"""評価: 検出のクラス別 AP と、イベント検出の適合率・再現率。

検出の評価は、既製の COCO モデル(person→player、sports ball→ball に読み替え)と自前モデルを
同じ test セット・同じ計算方法で比べられるよう、ultralytics の val に頼らず自前で計算する。
"""
import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 4), b: (M, 4) の xyxy → (N, M)"""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(preds: list[tuple[int, float, np.ndarray]], gts: dict[int, np.ndarray],
                      iou_thr: float = 0.5) -> dict:
    """1クラス分の AP(全点補間)。

    preds: [(画像番号, 信頼度, xyxy)]、gts: {画像番号: (K, 4)}
    """
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return {"ap": None, "n_gt": 0, "precision": None, "recall": None}
    preds = sorted(preds, key=lambda p: -p[1])
    used = {k: np.zeros(len(g), bool) for k, g in gts.items()}
    tp = np.zeros(len(preds))
    for i, (img, _, box) in enumerate(preds):
        g = gts.get(img)
        if g is None or len(g) == 0:
            continue
        ov = iou(box, g)[0]
        j = int(np.argmax(ov))
        if ov[j] >= iou_thr and not used[img][j]:
            used[img][j] = True
            tp[i] = 1
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(preds) + 1) if len(preds) else np.zeros(0)
    # 全点補間: 適合率を右から単調非増加にしてから、再現率の増分で積分する
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "n_gt": n_gt, "n_pred": len(preds),
            "recall": float(recall[-1]) if len(recall) else 0.0,
            "precision": float(precision[-1]) if len(precision) else None}


def detection_report(all_preds: dict[int, list[tuple[int, float, np.ndarray]]],
                     all_gts: dict[int, dict[int, np.ndarray]], names: dict[int, str]) -> dict:
    """クラスごとの AP50 と、AP50:95(IoU 0.50〜0.95 の平均)。"""
    out = {}
    for c, name in names.items():
        p, g = all_preds.get(c, []), all_gts.get(c, {})
        ap50 = average_precision(p, g, 0.5)
        aps = [average_precision(p, g, t)["ap"] for t in np.arange(0.5, 0.96, 0.05)]
        out[name] = {**ap50, "ap50_95": float(np.mean(aps)) if ap50["ap"] is not None else None}
    return out


def match_events(pred: list[dict], gt: list[dict], tol: float = 2.0, types=None) -> dict:
    """イベントを種別・チームが一致し、時刻差が tol 秒以内のもの同士で1対1に対応付け、種別ごとの P/R を返す。"""
    types = types or sorted({e["type"] for e in gt} | {e["type"] for e in pred})
    report = {}
    for typ in types:
        P = [e for e in pred if e["type"] == typ]
        G = [e for e in gt if e["type"] == typ]
        used = [False] * len(G)
        tp = 0
        for e in sorted(P, key=lambda e: -e.get("confidence", 1.0)):
            cands = [(abs(e["t"] - g["t"]), j) for j, g in enumerate(G)
                     if not used[j] and g.get("team") == e.get("team") and abs(e["t"] - g["t"]) <= tol]
            if cands:
                used[min(cands)[1]] = True
                tp += 1
        report[typ] = {"tp": tp, "n_pred": len(P), "n_gt": len(G),
                       "precision": tp / len(P) if P else None, "recall": tp / len(G) if G else None}
    return report
