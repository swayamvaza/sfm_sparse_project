from __future__ import annotations

import cv2
import numpy as np

from features import FeatureSet


def triangulate_between_anchor_and_new(
    anchor_img,
    anchor_feat: FeatureSet,
    new_feat: FeatureSet,
    anchor_R: np.ndarray,
    anchor_t: np.ndarray,
    new_R: np.ndarray,
    new_t: np.ndarray,
    K: np.ndarray,
    ratio: float = 0.75,
):
    from matching import match_features

    matches = match_features(anchor_feat.descriptors, new_feat.descriptors, ratio=ratio)
    if len(matches) < 8:
        return np.empty((0, 3)), np.empty((0, 3))

    pts_anchor = np.float32([anchor_feat.keypoints[m.queryIdx].pt for m in matches])
    pts_new = np.float32([new_feat.keypoints[m.trainIdx].pt for m in matches])

    P1 = K @ np.hstack([anchor_R, anchor_t])
    P2 = K @ np.hstack([new_R, new_t])

    pts4d_h = cv2.triangulatePoints(P1, P2, pts_anchor.T, pts_new.T)
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T

    valid = np.isfinite(pts3d).all(axis=1) & (pts3d[:, 2] > 0)
    pts3d = pts3d[valid]
    pts_anchor = pts_anchor[valid]

    h, w = anchor_img.shape[:2]
    pts_int = np.round(pts_anchor).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h - 1)
    colors = anchor_img[pts_int[:, 1], pts_int[:, 0], ::-1].astype(np.float32) / 255.0

    return pts3d, colors
