from __future__ import annotations

import os
import glob
from typing import List, Tuple

import cv2
import numpy as np


def load_images(images_dir: str) -> List[np.ndarray]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        raise ValueError("Need at least 2 readable images.")

    return images


def estimate_intrinsics(image_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    f = 0.9 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    return np.array(
        [[f, 0.0, cx],
         [0.0, f, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]
