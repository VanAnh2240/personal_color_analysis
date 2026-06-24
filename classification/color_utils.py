"""
classification/color_utils.py
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple, Optional

RGB    = Tuple[int, int, int]
BGRArr = np.ndarray


# 1.  Colour-space helpers
def rgb_to_lab(rgb: RGB) -> np.ndarray:
    bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)


def lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def _hsv_value_255(rgb: RGB) -> float:
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 2])


def _hsv_saturation_255(rgb: RGB) -> float:
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 1])


# 2.  Pure-numpy k-means
def _kmeans_numpy(pixels: np.ndarray, k: int,
                  max_iter: int = 50, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    N   = len(pixels)

    centers = [pixels[rng.integers(N)].copy()]
    for _ in range(1, k):
        c_arr   = np.array(centers, dtype=np.float32)
        sq_dist = np.sum((pixels[:, None] - c_arr[None]) ** 2, axis=2)
        min_d   = sq_dist.min(axis=1)
        total   = min_d.sum()
        centers.append(
            pixels[rng.integers(N)].copy() if total == 0
            else pixels[rng.choice(N, p=min_d / total)].copy()
        )
    centers = np.array(centers, dtype=np.float32)

    for _ in range(max_iter):
        labels = np.sum((pixels[:, None] - centers[None]) ** 2, axis=2).argmin(axis=1)
        new_c  = np.array([
            pixels[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ], dtype=np.float32)
        if np.allclose(centers, new_c, atol=0.5):
            break
        centers = new_c
    return centers


# 3.  Dominant-colour extraction
def _brightness(rgb: RGB) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def extract_dominant_color(
    bgr_image: BGRArr,
    mask: np.ndarray,
    k: int = 3,
    min_brightness: float = 15.0,
    max_brightness: float = 240.0,
    prefer_bright: bool = True,
    random_state: int = 42,
) -> Optional[RGB]:
    region_pixels = bgr_image[mask == 255]
    if len(region_pixels) < k:
        return None

    rgb_pixels = region_pixels[:, ::-1].astype(np.float32)
    centers    = _kmeans_numpy(rgb_pixels, k=k, random_state=random_state)
    candidates: List[RGB] = [tuple(int(c) for c in ctr) for ctr in centers]

    valid = [c for c in candidates if min_brightness <= _brightness(c) <= max_brightness]
    if not valid:
        valid = candidates

    best_rgb, best_score = None, float("inf")
    for candidate in valid:
        diff   = rgb_pixels - np.array(candidate, dtype=np.float32)
        rmse   = float(np.sqrt(np.mean(diff ** 2)))
        bright = _brightness(candidate)
        weight = (1.0 + (1.0 - bright / 255.0) * 0.5 if prefer_bright
                  else 1.0 + (bright / 255.0) * 0.5)
        score  = rmse * weight
        if score < best_score:
            best_score = score
            best_rgb   = candidate
    return best_rgb


def classify_hue(skin_rgb: RGB) -> str:
    """
    S (Subtone/Hue) metric — warm vs cool.
    """
    lab = rgb_to_lab(skin_rgb)
    a = float(lab[1]) - 128
    b = float(lab[2]) - 128
    warm_score = b - 0.5 * a
    return "warm" if warm_score > 8 else "cool"


def classify_chroma(
        skin_rgb: RGB,
        eyes_rgb: RGB,
        lips_rgb: Optional[RGB] = None,
        threshold: float = 130.0
    ) -> str:
    """
    I (Intensity / chroma) metric: bright vs muted.
    """
    weights: list[float] = []
    sats: list[float] = []

    sats.append(_hsv_saturation_255(skin_rgb));  weights.append(0.50)
    if eyes_rgb is not None:
        sats.append(_hsv_saturation_255(eyes_rgb)); weights.append(0.30)
    if lips_rgb is not None:
        sats.append(_hsv_saturation_255(lips_rgb)); weights.append(0.20)

    total_w = sum(weights)
    avg_sat = sum(s * w for s, w in zip(sats, weights)) / total_w

    return "bright" if avg_sat >= threshold else "muted"


def classify_value(
    skin_rgb: RGB,
    threshold: float = 170.0,
) -> str:
    """
    V (Value) metric — light vs dark.
    """
    v = _hsv_value_255(skin_rgb)
    return "light" if v > threshold else "dark"

def classify_contrast(
    skin_rgb: RGB,
    hair_rgb:  Optional[RGB],
    threshold: float = 70.0,
) -> Optional[str]:
    """
    C (Contrast) metric — high vs low.
    """
    if hair_rgb is None:
        return None
    v_skin = _hsv_value_255(skin_rgb)
    v_hair = _hsv_value_255(hair_rgb)

    diff =  abs(v_skin - v_hair)
    return "high" if diff >= threshold else "low"


