from __future__ import annotations

from typing import List

import cv2
import numpy as np

from features import FeatureSet


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def build_correspondences(
    feat1: FeatureSet,
    feat2: FeatureSet,
    matches,
):
    pts1 = np.float32([feat1.keypoints[m.queryIdx].pt for m in matches])
    pts2 = np.float32([feat2.keypoints[m.trainIdx].pt for m in matches])

    if len(pts1) < 8:
        raise RuntimeError("Not enough matches for geometry estimation.")

    return pts1, pts2


def choose_best_pair(features: List[FeatureSet]):
    best_i, best_j, best_matches = -1, -1, []
    best_score = -1

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            matches = match_features(features[i].descriptors, features[j].descriptors)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_i, best_j, best_matches = i, j, matches

    if best_score < 8:
        raise RuntimeError("Could not find a strong image pair for initialization.")

    return best_i, best_j, best_matches
