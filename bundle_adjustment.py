from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def project(points, camera_params, K):
    """Project 3D points to 2D using camera parameters."""
    import cv2
    points_proj = np.zeros((len(points), 2))
    for i in range(len(points)):
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:]
        proj, _ = cv2.projectPoints(np.array([points[i]]), rvec, tvec, K, None)
        points_proj[i] = proj.ravel()
    return points_proj

def ba_objective(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()

def bundle_adjustment(camera_poses: dict, tracks: list, K: np.ndarray, features: list):
    """Refine camera poses and 3D points."""
    import cv2
    
    n_cameras = len(camera_poses)
    n_points = len(tracks)
    
    # Map camera keys to indices
    camera_map = {cam_id: i for i, cam_id in enumerate(camera_poses.keys())}
    camera_map_inv = {i: cam_id for cam_id, i in camera_map.items()}
    
    # Prepare params
    camera_params = np.zeros((n_cameras, 6))
    for cam_id, (R, t) in camera_poses.items():
        rvec, _ = cv2.Rodrigues(R)
        camera_params[camera_map[cam_id]] = np.hstack((rvec.ravel(), t.ravel()))
        
    points_3d = np.zeros((n_points, 3))
    
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for i, track in enumerate(tracks):
        points_3d[i] = track.point_3d
        for cam_id, feat_idx in track.observations.items():
            if cam_id in camera_map:
                camera_indices.append(camera_map[cam_id])
                point_indices.append(i)
                # fetch 2D point from features
                pt2d = features[cam_id].keypoints[feat_idx].pt
                points_2d.append(pt2d)
                
    camera_indices = np.array(camera_indices, dtype=int)
    point_indices = np.array(point_indices, dtype=int)
    points_2d = np.array(points_2d)

    if len(points_2d) == 0:
        return camera_poses, tracks

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    res = least_squares(ba_objective, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))
    
    # unpack
    opt_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    opt_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    
    # update camera_poses
    new_camera_poses = {}
    for i in range(n_cameras):
        cam_id = camera_map_inv[i]
        rvec = opt_camera_params[i, :3]
        t = opt_camera_params[i, 3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        new_camera_poses[cam_id] = (R, t)
        
    # update tracks
    for i, track in enumerate(tracks):
        track.point_3d = opt_points_3d[i]
        
    return new_camera_poses, tracks

