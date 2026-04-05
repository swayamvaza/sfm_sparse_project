from __future__ import annotations

import argparse

from reconstruction import run_two_view, run_incremental


def main():
    parser = argparse.ArgumentParser(description="Sparse 3D reconstruction project")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder containing overlapping images")
    parser.add_argument("--out_ply", type=str, default="sparse_cloud.ply", help="Output PLY file")
    parser.add_argument("--mode", type=str, default="two_view", choices=["two_view", "incremental"])
    args = parser.parse_args()

    if args.mode == "incremental":
        run_incremental(args.images_dir, args.out_ply)
    else:
        run_two_view(args.images_dir, args.out_ply)


if __name__ == "__main__":
    main()
