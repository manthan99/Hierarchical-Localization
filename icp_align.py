"""
icp_align.py
============
Apply coarse T_ariaWorld_from_blkWorld to BLK scan, refine with ICP against
Aria semidense point cloud, save the aligned BLK scan and ICP stats.

Usage:
  python icp_align.py \
      --blk_pcd       .../blk/blk_scan.pcd \
      --aria_pcd      .../semidense_points.ply \
      --transform     .../alignment/T_ariaWorld_from_blkWorld.npy \
      --output_pcd    .../blk_scan_aria_leader.ply \
      --output_stats  .../alignment/icp_stats.txt \
      --voxel_size    0.02
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    pa = argparse.ArgumentParser(description="ICP-refine BLK→Aria alignment")
    pa.add_argument("--blk_pcd",      type=Path, required=True)
    pa.add_argument("--aria_pcd",     type=Path, required=True)
    pa.add_argument("--transform",    type=Path, required=True,
                    help="T_ariaWorld_from_blkWorld .npy file")
    pa.add_argument("--output_pcd",   type=Path, required=True,
                    help="Output: BLK scan in Aria frame (PLY)")
    pa.add_argument("--output_stats", type=Path, default=None,
                    help="Output: ICP stats JSON")
    pa.add_argument("--voxel_size",   type=float, default=0.02,
                    help="Voxel downsampling for ICP [m] (default: 2cm)")
    pa.add_argument("--icp_threshold", type=float, default=0.5,
                    help="ICP max correspondence distance [m]")
    args = pa.parse_args()

    args.output_pcd.parent.mkdir(parents=True, exist_ok=True)

    # Load transform
    T_aria_from_blk = np.load(str(args.transform))
    print(f"Coarse T_ariaWorld_from_blkWorld:")
    for row in T_aria_from_blk:
        print(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")

    # Load point clouds
    print(f"\nLoading BLK scan: {args.blk_pcd}")
    pcd_blk = o3d.io.read_point_cloud(str(args.blk_pcd))
    print(f"  {len(pcd_blk.points):,} points")

    print(f"Loading Aria semidense: {args.aria_pcd}")
    pcd_aria = o3d.io.read_point_cloud(str(args.aria_pcd))
    print(f"  {len(pcd_aria.points):,} points")

    # Apply coarse transform: BLK → Aria frame
    print("\nApplying coarse transform …")
    pcd_blk_in_aria = pcd_blk.transform(T_aria_from_blk)

    # Save coarse-only aligned BLK (before ICP)
    coarse_path = args.output_pcd.with_stem(args.output_pcd.stem + "_coarse")
    o3d.io.write_point_cloud(str(coarse_path), pcd_blk_in_aria)
    print(f"  Saved coarse-only BLK: {coarse_path}")

    # Downsample for ICP
    print(f"Downsampling at {args.voxel_size*100:.0f} cm …")
    blk_ds = pcd_blk_in_aria.voxel_down_sample(args.voxel_size)
    aria_ds = pcd_aria.voxel_down_sample(args.voxel_size)
    print(f"  BLK downsampled:  {len(blk_ds.points):,}")
    print(f"  Aria downsampled: {len(aria_ds.points):,}")

    # Estimate normals
    print("Estimating normals …")
    radius = args.voxel_size * 3
    blk_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    aria_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # ICP: Aria (source) → BLK (target)
    # BLK is denser → better normals → use as target for point-to-plane.
    # We get T that moves Aria→BLK, then invert to move BLK→Aria.
    print(f"Running point-to-plane ICP (threshold={args.icp_threshold} m) …")
    print(f"  Source: Aria ({len(aria_ds.points):,} pts)  →  Target: BLK ({len(blk_ds.points):,} pts)")
    icp_result = o3d.pipelines.registration.registration_icp(
        aria_ds, blk_ds,       # Aria=source, BLK=target
        args.icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,
            relative_fitness=1e-6,
            relative_rmse=1e-6,
        ),
    )

    T_aria_to_blk = icp_result.transformation   # moves Aria → BLK
    T_icp = np.linalg.inv(T_aria_to_blk)        # invert: correction for BLK → Aria
    fitness = icp_result.fitness
    inlier_rmse = icp_result.inlier_rmse

    print(f"\n  ICP Results:")
    print(f"    Fitness:     {fitness:.4f} ({fitness*100:.1f}% of Aria points matched)")
    print(f"    Inlier RMSE: {inlier_rmse:.4f} m ({inlier_rmse*100:.2f} cm)")
    print(f"    ICP correction (for BLK, inverted):")
    for row in T_icp:
        print(f"      [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")

    # Apply ICP refinement to BLK
    pcd_blk_final = pcd_blk_in_aria.transform(T_icp)

    # Save ICP-refined aligned BLK
    o3d.io.write_point_cloud(str(args.output_pcd), pcd_blk_final)
    print(f"\n  Saved aligned BLK (ICP-refined): {args.output_pcd}")

    # Compute final transform
    T_final = T_icp @ T_aria_from_blk
    np.save(str(args.output_pcd.with_name("T_ariaWorld_from_blkWorld_refined.npy")),
            T_final)

    # Save stats
    stats = {
        "blk_pcd": str(args.blk_pcd),
        "aria_pcd": str(args.aria_pcd),
        "blk_points": len(pcd_blk.points),
        "aria_points": len(pcd_aria.points),
        "voxel_size_m": args.voxel_size,
        "icp_threshold_m": args.icp_threshold,
        "icp_fitness": float(fitness),
        "icp_inlier_rmse_m": float(inlier_rmse),
        "icp_inlier_rmse_cm": float(inlier_rmse * 100),
        "T_coarse": T_aria_from_blk.tolist(),
        "T_icp_correction": T_icp.tolist(),
        "T_refined": T_final.tolist(),
    }

    stats_path = args.output_stats or args.output_pcd.with_name("icp_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved ICP stats: {stats_path}")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
