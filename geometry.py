from __future__ import annotations

import cv2
import numpy as np

from features import FeatureSet
from matching import match_features


def estimate_pose_and_triangulate(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
):
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    inlier_mask = mask.ravel().astype(bool) & pose_mask.ravel().astype(bool)
    return E, inlier_mask, R, t


def pnp_register_image(
    new_feat: FeatureSet,
    anchor_feat: FeatureSet,
    anchor_kp_indices: np.ndarray,
    anchor_points_3d: np.ndarray,
    K: np.ndarray,
    ratio: float = 0.75,
):
    matches = match_features(anchor_feat.descriptors, new_feat.descriptors, ratio=ratio)
    if len(matches) < 6:
        return False, None, None, None, None, None

    anchor_to_3d = {int(kp_id): idx for idx, kp_id in enumerate(anchor_kp_indices)}

    obj_pts = []
    img_pts = []
    used_anchor_ids = []

    for m in matches:
        kp_id = int(m.queryIdx)
        if kp_id not in anchor_to_3d:
            continue
        obj_pts.append(anchor_points_3d[anchor_to_3d[kp_id]])
        img_pts.append(new_feat.keypoints[m.trainIdx].pt)
        used_anchor_ids.append(kp_id)

    if len(obj_pts) < 6:
        return False, None, None, None, None, None

    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K,
        None,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 6:
        return False, None, None, None, None, None

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    inliers = inliers.ravel()
    return (
        True,
        R,
        tvec,
        img_pts[inliers],
        obj_pts[inliers],
        np.asarray(used_anchor_ids, dtype=int)[inliers],
    )
