"""スコアボード(得点・試合時計・ショットクロック)の読み取りと、時間方向の整合チェック。"""
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

_WIN_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if not shutil.which("tesseract") and _WIN_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_WIN_TESSERACT)


def _ocr(frame: np.ndarray, box, whitelist: str) -> str:
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # 白抜き文字を黒文字にそろえる(Tesseract は白地に黒文字が得意)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if b.mean() < 127:
        b = 255 - b
    return pytesseract.image_to_string(b, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}").strip()


MAX_PERIOD_MIN = 12   # 1ピリオドの最大の長さ(分)。車いすバスケは10分


def parse_clock(s: str) -> float | None:
    """'7:20' → 440.0、'59.9' → 59.9(残り1分未満の表示)。"""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m and int(m.group(1)) <= MAX_PERIOD_MIN and int(m.group(2)) < 60:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})\.(\d)", s)
    if m:
        return float(s)
    return None


SWAP_WEIGHT = 0.5   # 数字を入れ替えた候補の重み(読んだままの候補は 1.0)


def clock_candidates(s: str) -> list[tuple[float, float]]:
    """OCR の読みと、取り違えやすい数字(1↔7)を入れ替えた読みから、時計の候補 (値, 重み) を返す。"""
    cands = {}
    raw = parse_clock(s)
    if raw is not None:
        cands[raw] = 1.0
    for i, ch in enumerate(s):
        if ch in "17":
            v = parse_clock(s[:i] + ("7" if ch == "1" else "1") + s[i + 1:])
            if v is not None and v not in cands:
                cands[v] = SWAP_WEIGHT
    return sorted(cands.items())


SHOT_CLOCK_MAX = 24
OREB_RESET = 14   # FIBA: オフェンスリバウンド等ではショットクロックが 14 秒に戻る
MERGE_SEC = 2.5   # この間隔以内に続いたリセットは1回とみなす(表示の切り替わり途中の値を除く)


def read_scoreboard(frame: np.ndarray, rois: dict, digits=None) -> dict:
    """得点・試合時計・ショットクロックを読む。digits(wbt.digits.DigitReader)があれば得点とショットクロックに使う。

    Tesseract はこのスコアボードのフォントで「7」を「1」と読み、「27」を「21」と誤読することがあった
    (得点は減らないという整合チェックで捨てられ、得点の変化を取りこぼした)。
    """
    def read_digits(key):
        if digits is None:
            return _ocr(frame, rois[key], "0123456789")
        x0, y0, x1, y1 = rois[key]
        return digits.read(frame[y0:y1, x0:x1])

    a = read_digits("score_a")
    b = read_digits("score_b")
    c = _ocr(frame, rois["clock"], "0123456789:.") if "clock" in rois else ""
    shot = None
    if "shot_clock" in rois:
        s = read_digits("shot_clock")
        if s.isdigit() and int(s) <= SHOT_CLOCK_MAX:
            shot = int(s)
    return {
        "score_a": int(a) if a.isdigit() and len(a) <= 3 else None,
        "score_b": int(b) if b.isdigit() and len(b) <= 3 else None,
        "clock_candidates": clock_candidates(c),
        "shot_clock": shot,
    }


def _stable_shot_clock(ts: list[float], values: list[int | None], tol: float = 1.5) -> list[int | None]:
    """ショットクロックの読みから、時間の流れと整合する値だけを残す。

    - 直前に採用した値から見て「止まっている」か「経過時間だけ減った」値はそのまま採用する
    - 増えた値はリセット(24 か 14 に戻った)とみなすが、次の読みがその値から整合するときだけ採用する
      (1回だけの誤読をリセットと取り違えないため)
    """
    n = len(values)
    out: list[int | None] = [None] * n
    prev_t, prev_v = None, None
    idx = [i for i in range(n) if values[i] is not None]
    for k, i in enumerate(idx):
        v, t = values[i], ts[i]

        def confirmed_by_next():
            for j in idx[k + 1:k + 3]:
                dt = ts[j] - t
                if dt > 3.0:
                    break
                if abs(values[j] - v) <= tol or abs(values[j] - (v - dt)) <= tol:
                    return True
            return False

        def plausible_reset(dt):
            # リセット後の値は 24 か 14 から、前の読みからの経過時間以内しか減っていないはず
            return any(0 <= r - v <= dt + tol for r in (SHOT_CLOCK_MAX, OREB_RESET))

        if prev_v is None:
            if confirmed_by_next():
                out[i], prev_t, prev_v = v, t, v
            continue
        dt = t - prev_t
        if abs(v - prev_v) <= tol or abs(v - (prev_v - dt)) <= tol or (v < prev_v and prev_v - v <= dt + tol):
            out[i], prev_t, prev_v = v, t, v
        elif v > prev_v + tol and plausible_reset(dt) and confirmed_by_next():
            out[i], prev_t, prev_v = v, t, v
    return out


def shot_clock_resets(series: list[dict]) -> list[dict]:
    """ショットクロックが増えた(リセットされた)時刻と、戻った値(24 か 14)を返す。

    読みの間隔があるため、リセット直後の読みは 23 や 13 のこともある。14 を超えていれば 24 へのリセットとみなす。
    時刻は、リセット前の最後の読みとリセット後の最初の読みの中間とする。

    MERGE_SEC 以内に続いたリセットは1回にまとめ、最後の値を採る。リングに当たった直後に一瞬 14 が表示され、
    1秒ほどで 24 に変わることがある(USA v Japan では、公式記録上すべてディフェンスリバウンドの場面で
    「14 → 24」が6回あった)。時刻は最初のリセットの時刻とする。
    """
    out = []
    prev = None
    for r in series:
        v = r.get("shot_clock")
        if v is None:
            continue
        if prev is not None and v > prev["shot_clock"] + 1.5:
            to = SHOT_CLOCK_MAX if v > OREB_RESET else OREB_RESET
            t = round((prev["t"] + r["t"]) / 2, 2)
            if out and t - out[-1]["t_last"] <= MERGE_SEC:
                out[-1]["to"], out[-1]["t_last"] = to, t
            else:
                out.append({"t": t, "to": to, "t_last": t})
        prev = r
    return [{"t": x["t"], "to": x["to"]} for x in out]


def _stable_scores(values: list[int | None], persist: int = 2, max_jump: int = 3) -> list[int | None]:
    """誤読を除いた得点の系列を返す。

    得点は減らないので、確定値より小さい読みは捨てる。確定値より大きい読みは、
    同じ値が persist 回連続し、かつ増分が max_jump 以下のときだけ採用する。
    最初の確定値は、同じ値が persist 回連続したときに決める。
    """
    out, cur, cand, run, start = [], None, None, 0, 0
    for i, v in enumerate(values):
        if v is None:
            out.append(cur)
            continue
        if cur is not None and v == cur:
            cand, run = None, 0
        elif cur is None or (cur < v <= cur + max_jump):
            if v == cand:
                run += 1
            else:
                cand, run, start = v, 1, i
            if run >= persist:
                cur, cand, run = v, None, 0
                out[start:] = [v] * (i - start)   # 最初に読めた時点までさかのぼって確定する
        out.append(cur)
    return out


def _stable_clock(ts: list[float], cands: list[list[tuple[float, float]]], tol: float = 1.5) -> list[float | None]:
    """各時刻の候補 (値, 重み) から、時間の流れと整合する(止まっている、または経過時間だけ減っている)値を選ぶ。

    最初の値は、前後数回の読みで重みの合計が最も大きい候補とする。以降は、直前に採用した値から
    見て「止まっている」か「経過時間だけ減った」値に最も近い候補を採用し、外れる場合は捨てる。
    """
    n = len(cands)
    out: list[float | None] = [None] * n
    prev_t, prev_c, misses = None, None, 0
    for i in range(n):
        if not cands[i]:
            continue
        if prev_c is None:
            window = [cw for j in range(i, min(n, i + 5)) for cw in cands[j]]
            support = {c: sum(w for v, w in window if abs(v - c) <= 3) for c, _ in cands[i]}
            best = max(support, key=support.get)
            if support[best] >= 2.5:
                out[i], prev_t, prev_c = best, ts[i], best
            continue
        dt = ts[i] - prev_t
        err, _, c = min((min(abs(c - prev_c), abs(c - (prev_c - dt))), -w, c) for c, w in cands[i])
        if err <= tol:
            out[i], prev_t, prev_c, misses = c, ts[i], c, 0
        else:
            misses += 1
            if misses >= 6:   # ピリオドの切り替わり等で時計が飛んだとみなし、初期化し直す
                prev_c, misses = None, 0
    return out


def smooth_scoreboard(readings: list[dict]) -> list[dict]:
    """readings: [{"t", "score_a", "score_b", "clock"}] を整合チェック済みの系列にする。"""
    ts = [r["t"] for r in readings]
    a = _stable_scores([r["score_a"] for r in readings])
    b = _stable_scores([r["score_b"] for r in readings])
    c = _stable_clock(ts, [r["clock_candidates"] for r in readings])
    sc = _stable_shot_clock(ts, [r.get("shot_clock") for r in readings])
    return [{"t": t, "score_a": sa, "score_b": sb, "clock": cl, "shot_clock": s,
             "visible": r["score_a"] is not None and r["score_b"] is not None}
            for t, sa, sb, cl, s, r in zip(ts, a, b, c, sc, readings)]


def score_changes(series: list[dict]) -> list[dict]:
    """得点が増えた時刻・チーム・増分を返す。時刻は、新しい値を最初に読めた時刻とする。"""
    out = []
    prev = {"a": None, "b": None}
    for r in series:
        for team in ("a", "b"):
            v = r[f"score_{team}"]
            if v is None:
                continue
            if prev[team] is not None and v > prev[team]:
                out.append({"t": r["t"], "team": team, "points": v - prev[team], "score": v})
            prev[team] = v
    return out


def format_clock(sec: float | None) -> str | None:
    if sec is None:
        return None
    if sec < 60 and sec != int(sec):
        return f"{sec:.1f}"
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"
