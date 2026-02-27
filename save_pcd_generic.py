"""
save_pcd_generic.py
===================
Convert Aria MPS semidense_points.csv.gz to .ply using Open3D.

Usage:
  python save_pcd_generic.py \
      --input_path  .../slam/semidense_points.csv.gz \
      --output_path .../semidense_points.ply
"""

import argparse

import numpy as np
import open3d as o3d
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence


def main():
    pa = argparse.ArgumentParser(description="Convert MPS semidense points to PLY")
    pa.add_argument("--input_path",  type=str, required=True,
                    help="Path to semidense_points.csv.gz")
    pa.add_argument("--output_path", type=str, required=True,
                    help="Output .ply path")
    pa.add_argument("--inv_dist_std", type=float, default=0.001,
                    help="Inverse distance std threshold (lower = stricter)")
    pa.add_argument("--dist_std",    type=float, default=0.015,
                    help="Distance std threshold (lower = stricter)")
    args = pa.parse_args()

    print(f"Reading points from {args.input_path} ...")
    points = mps.read_global_point_cloud(args.input_path)

    filtered = filter_points_from_confidence(
        points, args.inv_dist_std, args.dist_std)
    print(f"Filtered {len(points)} → {len(filtered)} points")

    xyz = np.array([p.position_world for p in filtered])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud(args.output_path, pcd)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
