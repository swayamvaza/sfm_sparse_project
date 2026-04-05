from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, out_path: str) -> None:
    if o3d is None:
        raise RuntimeError("open3d is not installed.")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pc)


def visualize_point_cloud(points: np.ndarray, colors: np.ndarray) -> None:
    if o3d is None:
        print("Open3D not installed; skipping visualization.")
        return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries([pc])
