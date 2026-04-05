import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
from features import FeatureSet, create_sift, extract_features
from geometry import estimate_pose_and_triangulate, pnp_register_image
from matching import build_correspondences, choose_best_pair
from triangulation import triangulate_between_anchor_and_new
from utils import load_images, estimate_intrinsics
from visualization import save_point_cloud_ply, visualize_point_cloud
from bundle_adjustment import bundle_adjustment


@dataclass
class Track:
    point_3d: np.ndarray
    color: np.ndarray
    observations: dict  # {image_idx: keypoint_idx}
    error: float = 0.0


@dataclass
class ReconstructionResult:
    points_3d: np.ndarray
    colors: np.ndarray
    R: np.ndarray
    t: np.ndarray
    inlier_mask: np.ndarray


@dataclass
class ReconstructionState:
    images: List[np.ndarray]
    features: List[FeatureSet]
    K: np.ndarray
    registered_poses: dict
    tracks: List[Track]


def sample_colors(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    pts_int = np.clip(np.round(pts).astype(int), [[0, 0]], [[w - 1, h - 1]])
    colors = image[pts_int[:, 1], pts_int[:, 0], ::-1]
    return colors.astype(np.float32) / 255.0


def run_two_view(images_dir: str, out_ply: Optional[str] = None) -> ReconstructionResult:
    """Simple two-view reconstruction."""
    images = load_images(images_dir)
    sift = create_sift()
    print(f"Loaded {len(images)} images")
    features = [extract_features(img, sift) for img in images]
    print("Extracted SIFT features")

    i, j, matches = choose_best_pair(features)
    print(f"Best pair selected: image {i} and image {j} with {len(matches)} raw matches")

    pts1, pts2 = build_correspondences(features[i], features[j], matches)
    K = estimate_intrinsics(images[i].shape)

    _, inlier_mask, R, t = estimate_pose_and_triangulate(pts1, pts2, K)
    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    pts4d_h = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
    points_3d = (pts4d_h[:3] / pts4d_h[3]).T

    valid = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0)
    points_3d = points_3d[valid]
    inlier_pts1 = inlier_pts1[valid]
    colors = sample_colors(images[i], inlier_pts1)

    result = ReconstructionResult(points_3d=points_3d, colors=colors, R=R, t=t, inlier_mask=inlier_mask)

    print(f"Reconstructed {len(points_3d)} points")
    print("Pose: R =\n", R, "\nt =\n", t)

    if out_ply is not None:
        save_point_cloud_ply(points_3d, colors, out_ply)
        print(f"Saved to: {out_ply}")

    # visualize_point_cloud(points_3d, colors)  # Commented out for non-interactive mode
    return result


def run_incremental(images_dir: str, out_ply: Optional[str] = None):
    """Full incremental multi-view SfM with track management and bundle adjustment."""
    images = load_images(images_dir)
    sift = create_sift()
    features = [extract_features(img, sift) for img in images]
    K = estimate_intrinsics(images[0].shape)

    # 1. SELECT AND TRIANGULATE SEED PAIR
    i, j, matches = choose_best_pair(features)
    print(f"\nSeed pair: images {i} and {j}, {len(matches)} matches")

    pts1, pts2 = build_correspondences(features[i], features[j], matches)
    _, inlier_mask, R, t = estimate_pose_and_triangulate(pts1, pts2, K)

    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Filter valid points
    valid = np.isfinite(pts3d).all(axis=1) & (pts3d[:, 2] > 0)
    pts3d = pts3d[valid]
    inlier_pts1 = inlier_pts1[valid]
    colors_seed = sample_colors(images[i], inlier_pts1)

    # Create initial tracks
    good_match_idx = np.where(inlier_mask)[0][valid]
    inlier_indices = np.where(inlier_mask)[0]
    inlier_trainIdx = np.array([matches[k].trainIdx for k in inlier_indices])
    match_indices_j = inlier_trainIdx[valid]
    
    tracks = []
    for pt_idx in range(len(pts3d)):
        track = Track(
            point_3d=pts3d[pt_idx],
            color=colors_seed[pt_idx],
            observations={
                i: int(good_match_idx[pt_idx]),
                j: int(match_indices_j[pt_idx])
            }
        )
        tracks.append(track)

    state = ReconstructionState(
        images=images,
        features=features,
        K=K,
        registered_poses={
            i: (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)),
            j: (R, t),
        },
        tracks=tracks,
    )

    print(f"Initialized {len(tracks)} tracks from seed pair")

    # 2. INCREMENTAL REGISTRATION (sorted by proximity to seed pair)
    remaining_images = [idx for idx in range(len(images)) if idx not in state.registered_poses]
    # Sort by distance to closest seed image
    remaining_images.sort(key=lambda idx: min(abs(idx - i), abs(idx - j)))
    
    for idx in remaining_images:

        print(f"\nRegistering image {idx}...")

        # Collect 3D-2D correspondences from existing tracks
        obj_pts, img_pts = [], []
        track_indices = []
        
        for track_idx, track in enumerate(state.tracks):
            for reg_idx, kpt_idx in track.observations.items():
                if reg_idx in state.registered_poses:
                    obj_pts.append(track.point_3d)
                    kpt = features[reg_idx].keypoints[kpt_idx]
                    img_pts.append(kpt.pt)
                    track_indices.append(track_idx)
                    break

        if len(obj_pts) < 4:
            print(f"  Skipping: < 4 2D-3D correspondences")
            continue

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        # PnP-RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, None,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or rvec is None:
            print(f"  PnP failed")
            continue

        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.reshape(3, 1).astype(np.float64)
        state.registered_poses[idx] = (R_new, t_new)
        print(f"  Registered with {len(inliers) if inliers is not None else 0} inliers")

        # TRIANGULATE with registered images
        for reg_idx in state.registered_poses:
            if reg_idx >= idx or reg_idx == idx:
                continue

            # Match features
            desc1 = features[reg_idx].descriptors
            desc2 = features[idx].descriptors
            
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                continue

            bf = cv2.BFMatcher()
            matches_list = bf.knnMatch(desc1, desc2, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_pair in matches_list:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 4:
                continue

            pts1 = np.array([features[reg_idx].keypoints[m.queryIdx].pt for m in good_matches])
            pts2 = np.array([features[idx].keypoints[m.trainIdx].pt for m in good_matches])

            # PRIORITY 2: Epipolar geometry validation
            F, epipolar_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None or np.sum(epipolar_mask) < 4:
                continue
            epipolar_mask = epipolar_mask.ravel().astype(bool)
            pts1_epi = pts1[epipolar_mask]
            pts2_epi = pts2[epipolar_mask]
            good_matches_epi = [good_matches[k] for k in np.where(epipolar_mask)[0]]

            # Triangulate
            R_reg, t_reg = state.registered_poses[reg_idx]
            R_idx, t_idx = state.registered_poses[idx]

            P1 = K @ np.hstack([R_reg, t_reg])
            P2 = K @ np.hstack([R_idx, t_idx])
            pts4d = cv2.triangulatePoints(P1, P2, pts1_epi.T, pts2_epi.T)
            pts3d_new = (pts4d[:3] / pts4d[3]).T

            # Filter validity
            valid_new = np.isfinite(pts3d_new).all(axis=1) & (pts3d_new[:, 2] > 0)
            if not np.any(valid_new):
                continue

            pts3d_new = pts3d_new[valid_new]
            pts1_valid = pts1_epi[valid_new]
            good_matches_valid = [good_matches_epi[k] for k in np.where(valid_new)[0]]

            # PRIORITY 3: Track reuse - extend existing tracks instead of creating duplicates
            n_new, n_extended = 0, 0
            used_track_ids = set()
            
            for pt_idx, pt3d in enumerate(pts3d_new):
                m = good_matches_valid[pt_idx]
                kpt_idx_reg = m.queryIdx
                kpt_idx_new = m.trainIdx
                
                # Check if keypoint in reg_idx already belongs to a track
                existing_track_idx = None
                for t_idx, track in enumerate(state.tracks):
                    if reg_idx in track.observations and track.observations[reg_idx] == kpt_idx_reg:
                        existing_track_idx = t_idx
                        break
                
                if existing_track_idx is not None and existing_track_idx not in used_track_ids:
                    # Extend existing track
                    state.tracks[existing_track_idx].observations[idx] = kpt_idx_new
                    state.tracks[existing_track_idx].point_3d = pt3d
                    state.tracks[existing_track_idx].color = sample_colors(images[idx], pts1_valid[pt_idx:pt_idx+1])[0]
                    used_track_ids.add(existing_track_idx)
                    n_extended += 1
                else:
                    # Create new track
                    color = sample_colors(images[idx], pts1_valid[pt_idx:pt_idx+1])[0]
                    track = Track(
                        point_3d=pt3d,
                        color=color,
                        observations={
                            reg_idx: kpt_idx_reg,
                            idx: kpt_idx_new,
                        }
                    )
                    state.tracks.append(track)
                    n_new += 1

            print(f"  +{n_new} new, +{n_extended} extended from image {reg_idx}")

    # 3. BUNDLE ADJUSTMENT
    if len(state.tracks) > 0:
        print("\nRunning bundle adjustment...")
        state.registered_poses, state.tracks = bundle_adjustment(
            state.registered_poses, state.tracks, K, features
        )

    # 4. FILTER BY REPROJECTION ERROR
    print("Filtering by reprojection error...")
    filtered_tracks = []
    for track in state.tracks:
        errors = []
        for cam_id, kpt_idx in track.observations.items():
            R, t = state.registered_poses[cam_id]
            P = K @ np.hstack([R, t])
            pt_h = np.append(track.point_3d, 1.0)
            proj = P @ pt_h
            pt_2d = proj[:2] / proj[2]
            kpt = features[cam_id].keypoints[kpt_idx]
            err = np.linalg.norm(pt_2d - kpt.pt)
            errors.append(err)
        
        mean_err = np.mean(errors)
        track.error = mean_err
        # PRIORITY 1: Stricter reprojection filtering (1.0 pixel)
        if mean_err < 1.0:
            filtered_tracks.append(track)

    state.tracks = filtered_tracks
    print(f"Kept {len(state.tracks)} after filtering")

    # 5. OUTPUT
    if len(state.tracks) == 0:
        print("ERROR: No 3D points!")
        return state

    points = np.array([t.point_3d for t in state.tracks])
    colors = np.array([t.color for t in state.tracks])

    print(f"\nFinal: {len(points)} 3D points, {len(state.registered_poses)} cameras")

    if out_ply is not None:
        try:
            save_point_cloud_ply(points, colors, out_ply)
            print(f"Saved to: {out_ply}")
        except Exception as e:
            print(f"ERROR saving: {e}")

    # visualize_point_cloud(points, colors)  # Commented out for non-interactive mode
    return state
