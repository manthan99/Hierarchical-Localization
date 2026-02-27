"""
fast_align_pipeline.py
======================
Fast end-to-end pipeline to compute T_ariaWorld_from_blkWorld.

1. SfM on subsampled Aria images (every Nth frame)
2. Umeyama Sim(3): SfM → Aria metric world
3. Localize ALL BLK images in the Aria SfM
4. Compute T_ariaWorld_from_blkWorld per image
5. Report stats & save best transform

Usage
-----
python fast_align_pipeline.py \
    --aria_img_dir  /media/.../images/aria_undistorted \
    --aria_yaml_dir /media/.../images/aria \
    --blk_dir       /media/.../images/blk \
    --output_dir    outputs/fast_align \
    --aria_subsample 10 \
    --num_retrieval  5
"""

import argparse
import csv
import shutil
from pathlib import Path

import h5py
import numpy as np
import pycolmap
import yaml
from scipy.spatial.transform import Rotation as Rot

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    localize_sfm,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_aria_yaml_pose(yml_path):
    with open(yml_path) as f:
        raw = yaml.safe_load(f)
    ext = raw["extrinsics_world_camera"]
    t = np.array(ext["translation"], dtype=np.float64)
    q = ext["quaternion_xyzw"]
    R = Rot.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_blk_yaml_pose(yml_path):
    with open(yml_path) as f:
        raw = yaml.safe_load(f)
    pos = np.array(raw["Position"], dtype=np.float64)
    R_wc = np.array(raw["Rotation"], dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = pos
    return T


def umeyama(src, dst):
    """Sim(3): dst ≈ s*R@src + t. Returns s, R, t, T(4x4)."""
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    sig = np.sum(sc ** 2) / n
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sig
    t = mu_d - s * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return s, R, t, T


def rotation_angle_deg(R_mat):
    cos_a = np.clip((np.trace(R_mat) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_a))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--aria_img_dir",   type=Path, required=True)
    pa.add_argument("--aria_yaml_dir",  type=Path, required=True)
    pa.add_argument("--blk_dir",        type=Path, required=True)
    pa.add_argument("--output_dir",     type=Path, default=Path("outputs/fast_align"))
    pa.add_argument("--aria_subsample", type=int, default=10,
                    help="Use every Nth Aria frame for SfM")
    pa.add_argument("--num_retrieval",  type=int, default=5)
    args = pa.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sfm_dir = out / "sfm"

    feature_conf  = extract_features.confs["superpoint_aachen"]
    matcher_conf  = match_features.confs["superglue"]
    matcher_conf = {**matcher_conf, "model": {**matcher_conf["model"], "weights": "indoor"}}
    retrieval_conf = extract_features.confs["netvlad"]

    # ── Step 1: Subsampled Aria SfM ──────────────────────────────────────────
    print("=" * 70)
    print(f"Step 1: Aria SfM (every {args.aria_subsample}th frame)")
    print("=" * 70)

    all_aria = sorted(args.aria_img_dir.glob("*.jpg"))
    aria_subset = all_aria[::args.aria_subsample]
    aria_names = [f.name for f in aria_subset]
    print(f"  Total Aria images: {len(all_aria)}")
    print(f"  Using: {len(aria_names)} images")

    # Extract features
    print("  Extracting features …")
    aria_features = extract_features.main(
        feature_conf, args.aria_img_dir, out, image_list=aria_names)

    # Extract global descriptors
    print("  Extracting global descriptors …")
    aria_global = extract_features.main(
        retrieval_conf, args.aria_img_dir, out, image_list=aria_names)

    # Retrieval pairs (for SfM)
    sfm_pairs = out / "pairs-netvlad-sfm.txt"
    print("  Generating SfM pairs …")
    pairs_from_retrieval.main(aria_global, sfm_pairs, num_matched=10)

    # Match
    print("  Matching …")
    sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], out)

    # Reconstruct
    print("  Running SfM …")
    rec = reconstruction.main(
        sfm_dir, args.aria_img_dir, sfm_pairs, aria_features, sfm_matches,
        image_list=aria_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    print(f"  SfM: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    # ── Step 2: Umeyama SfM → Aria world ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 2: Sim(3) SfM → Aria world")
    print("=" * 70)

    sfm_positions = {}
    for img_id, img in rec.images.items():
        w2c = img.cam_from_world().inverse()
        sfm_positions[img.name] = np.array(w2c.translation)

    aria_positions = {}
    for yml in sorted(args.aria_yaml_dir.glob("*.yaml")):
        fname = yml.stem + ".jpg"
        if fname in sfm_positions:
            aria_positions[fname] = load_aria_yaml_pose(yml)[:3, 3]

    common = sorted(set(sfm_positions) & set(aria_positions))
    print(f"  Common frames: {len(common)}")

    pts_sfm = np.array([sfm_positions[n] for n in common])
    pts_aria = np.array([aria_positions[n] for n in common])

    scale, R_sa, t_sa, T_aria_from_sfm = umeyama(pts_sfm, pts_aria)
    res = np.linalg.norm(
        (T_aria_from_sfm[:3, :3] @ pts_sfm.T + T_aria_from_sfm[:3, 3:]).T - pts_aria,
        axis=1)
    print(f"  Scale: {scale:.6f}")
    print(f"  Residuals: mean={res.mean():.4f} m, median={np.median(res):.4f} m")
    np.save(str(out / "T_ariaWorld_from_sfm.npy"), T_aria_from_sfm)

    # ── Step 3: Localize BLK images ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 3: Localize ALL BLK images")
    print("=" * 70)

    images_root = args.blk_dir.parent
    blk_subdir = args.blk_dir.name  # e.g. 'blk' or 'images'
    blk_images = sorted(args.blk_dir.glob("*.jpg"))
    blk_names = [f"{blk_subdir}/{p.name}" for p in blk_images]
    print(f"  BLK images: {len(blk_names)} (subdir: {blk_subdir})")

    # Query list with intrinsics
    query_file = out / "blk_queries.txt"
    with open(query_file, "w") as f:
        for jpg in blk_images:
            yml = jpg.with_suffix(".yml")
            if not yml.exists():
                continue
            with open(yml) as yf:
                raw = yaml.safe_load(yf)
            W, H = int(raw["Width"]), int(raw["Height"])
            fx, fy = float(raw["FocalLength_X_Pixel"]), float(raw["FocalLength_Y_Pixel"])
            cx, cy = float(raw["PrincipalPoint_X_Pixel"]), float(raw["PrincipalPoint_Y_Pixel"])
            f.write(f"{blk_subdir}/{jpg.name} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # Extract BLK features (to separate dir)
    blk_tmp = out / "blk_tmp"
    blk_tmp.mkdir(exist_ok=True)
    print("  Extracting BLK features …")
    blk_features = extract_features.main(
        feature_conf, images_root, blk_tmp, image_list=blk_names)

    # Merge features
    merged_feat = out / (feature_conf["output"] + "_merged.h5")
    print("  Merging features …")
    shutil.copy2(str(aria_features), str(merged_feat))
    with h5py.File(str(blk_features), "r") as src, \
         h5py.File(str(merged_feat), "a") as dst:
        def _copy(name, obj):
            if isinstance(obj, h5py.Dataset):
                parent = "/".join(name.split("/")[:-1])
                if parent and parent not in dst:
                    dst.create_group(parent)
                if name not in dst:
                    src.copy(name, dst[parent] if parent else dst)
        src.visititems(_copy)

    # BLK global descriptors
    print("  Extracting BLK global descriptors …")
    blk_global = extract_features.main(
        retrieval_conf, images_root, blk_tmp, image_list=blk_names)

    # Merge global
    merged_global = out / (retrieval_conf["output"] + "_merged.h5")
    shutil.copy2(str(aria_global), str(merged_global))
    with h5py.File(str(blk_global), "r") as src, \
         h5py.File(str(merged_global), "a") as dst:
        def _copy2(name, obj):
            if isinstance(obj, h5py.Dataset):
                parent = "/".join(name.split("/")[:-1])
                if parent and parent not in dst:
                    dst.create_group(parent)
                if name not in dst:
                    src.copy(name, dst[parent] if parent else dst)
        src.visititems(_copy2)

    # Retrieval: BLK queries → Aria DB
    loc_pairs = out / "pairs-blk-loc.txt"
    aria_db_names = [img.name for img in rec.images.values()]
    print(f"  Retrieval (top {args.num_retrieval}) …")
    pairs_from_retrieval.main(
        merged_global, loc_pairs,
        num_matched=args.num_retrieval,
        query_list=blk_names,
        db_list=aria_db_names)

    # Match
    print("  Matching (SuperGlue) …")
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"] + "_merged", out)

    # Localize
    results = out / "BLK_localization.txt"
    print("  Localizing …")
    localize_sfm.main(
        reference_sfm=rec,
        queries=query_file,
        retrieval=loc_pairs,
        features=merged_feat,
        matches=loc_matches,
        results=results,
    )

    # ── Step 4: Compute T_ariaWorld_from_blkWorld ────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 4: T_ariaWorld_from_blkWorld")
    print("=" * 70)

    loc_poses = {}
    with open(results) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            name = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            rot = Rot.from_quat([qx, qy, qz, qw])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = [tx, ty, tz]
            loc_poses[name] = T

    print(f"  Localized: {len(loc_poses)} images")

    transforms = []
    names = []
    for name, T_cam_from_sfm in loc_poses.items():
        img_name = Path(name).name
        stem = Path(img_name).stem
        yml = args.blk_dir / (stem + ".yml")
        if not yml.exists():
            continue

        T_blkWorld_from_cam = load_blk_yaml_pose(yml)

        # SfM → Aria world
        T_sfm_from_cam = np.linalg.inv(T_cam_from_sfm)
        pos_sfm = T_sfm_from_cam[:3, 3]
        pos_aria = scale * R_sa @ pos_sfm + t_sa
        R_aria_from_cam = R_sa @ T_sfm_from_cam[:3, :3]

        T_aria_from_cam = np.eye(4)
        T_aria_from_cam[:3, :3] = R_aria_from_cam
        T_aria_from_cam[:3, 3] = pos_aria

        T_aria_from_blk = T_aria_from_cam @ np.linalg.inv(T_blkWorld_from_cam)
        transforms.append(T_aria_from_blk)
        names.append(name)

    transforms = np.array(transforms)
    print(f"  Valid transforms: {len(transforms)}")

    # Consistency: deviations from first
    T_ref = transforms[0]
    all_dt, all_dr = [], []
    for T_i in transforms:
        dT = np.linalg.inv(T_ref) @ T_i
        all_dt.append(np.linalg.norm(dT[:3, 3]))
        all_dr.append(rotation_angle_deg(dT[:3, :3]))
    all_dt, all_dr = np.array(all_dt), np.array(all_dr)

    # Inlier filtering
    t_thresh = max(2 * np.median(all_dt), 1.0)
    r_thresh = max(2 * np.median(all_dr), 5.0)
    inlier_mask = (all_dt <= t_thresh) & (all_dr <= r_thresh)
    n_inliers = inlier_mask.sum()
    print(f"  Inliers: {n_inliers}/{len(transforms)} "
          f"(dt<{t_thresh:.1f}m, dr<{r_thresh:.1f}°)")

    if n_inliers >= 3:
        inlier_T = transforms[inlier_mask]
        t_mean = inlier_T[:, :3, 3].mean(0)
        quats = [Rot.from_matrix(T[:3, :3]).as_quat() for T in inlier_T]
        quats = np.array(quats)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]
        q_mean = quats.mean(0)
        q_mean /= np.linalg.norm(q_mean)

        T_best = np.eye(4, dtype=np.float64)
        T_best[:3, :3] = Rot.from_quat(q_mean).as_matrix()
        T_best[:3, 3] = t_mean
    else:
        T_best = T_ref

    euler = Rot.from_matrix(T_best[:3, :3]).as_euler('xyz', degrees=True)

    print(f"\n  Inlier stats:")
    dt_in = all_dt[inlier_mask]
    dr_in = all_dr[inlier_mask]
    print(f"    Δt: mean={dt_in.mean():.4f}, std={dt_in.std():.4f} m")
    print(f"    Δr: mean={dr_in.mean():.4f}, std={dr_in.std():.4f}°")

    print(f"\n{'=' * 70}")
    print("RESULT: T_ariaWorld_from_blkWorld")
    for row in T_best:
        print(f"  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}, {row[3]:12.8f}]")
    print(f"  Euler (xyz): [{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]")
    print(f"  Translation: [{T_best[0,3]:.4f}, {T_best[1,3]:.4f}, {T_best[2,3]:.4f}]")

    np.save(str(out / "T_ariaWorld_from_blkWorld.npy"), T_best)
    np.savetxt(str(out / "T_ariaWorld_from_blkWorld.txt"), T_best, fmt="%.10f")
    print(f"\n  Saved to {out / 'T_ariaWorld_from_blkWorld.npy'}")


if __name__ == "__main__":
    main()
