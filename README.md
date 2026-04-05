# Sparse 3D Reconstruction Project

A simple, interview-friendly multi-view 3D reconstruction pipeline using:

- SIFT feature extraction
- Feature matching
- Epipolar geometry
- Pose recovery
- Triangulation
- Incremental sparse reconstruction
- Open3D visualization

## Files

- `main.py` — entry point
- `utils.py` — image loading and camera intrinsics
- `features.py` — SIFT extraction
- `matching.py` — matching and image pair selection
- `geometry.py` — essential matrix, pose recovery, PnP registration
- `triangulation.py` — triangulation helpers
- `visualization.py` — point cloud save + display
- `bundle_adjustment.py` — placeholder / optional refinement hook
- `reconstruction.py` — two-view and incremental reconstruction pipeline

## Install

```bash
pip install opencv-contrib-python numpy open3d
```

Optional for bundle adjustment:
```bash
pip install scipy
```

## Run

Two-view demo:
```bash
python main.py --images_dir path/to/images --mode two_view
```

Incremental mode:
```bash
python main.py --images_dir path/to/images --mode incremental
```
