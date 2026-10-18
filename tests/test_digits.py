import cv2
import numpy as np

from wbt.digits import DigitReader, glyphs, harvest


def render(text: str, w: int = 90, h: int = 45) -> np.ndarray:
    """赤地に白文字のスコアボード風の画像を作る。"""
    img = np.full((h, w, 3), (40, 30, 200), np.uint8)
    cv2.putText(img, text, (4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def test_glyphs_skip_colon_and_split_digits():
    assert len(glyphs(render("7:25"))) == 3
    assert len(glyphs(render("14", w=60))) == 2


def test_reader_learns_from_harvested_clock_and_reads_new_values():
    samples = []
    for text in ["7:25", "7:16", "7:09", "6:38", "5:47", "4:02"]:
        samples += harvest(render(text), text)
    reader = DigitReader(np.array([g for g, _ in samples]), np.array([d for _, d in samples]))
    assert reader.read(render("14", w=60)) == "14"
    assert reader.read(render("5", w=40)) == "5"
    assert reader.read(render("23", w=60)) == "23"


def test_reader_returns_empty_for_unknown_shapes():
    samples = harvest(render("7:25"), "7:25")
    reader = DigitReader(np.array([g for g, _ in samples]), np.array([d for _, d in samples]), max_dist=0.05)
    assert reader.read(render("8", w=40)) == ""
