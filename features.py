from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class FeatureSet:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray


def create_sift() -> cv2.SIFT:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT not available. Install opencv-contrib-python.")
    return cv2.SIFT_create(nfeatures=10000)


def extract_features(image: np.ndarray, sift: cv2.SIFT) -> FeatureSet:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("No features detected in an image.")
    return FeatureSet(keypoints=keypoints, descriptors=descriptors)
