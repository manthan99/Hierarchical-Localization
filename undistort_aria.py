"""
undistort_aria.py
=================
Undistort raw Aria RGB fisheye images to a pinhole camera model.

Camera model: Fisheye624 (6 radial + 2 tangential + 4 thin-prism coefficients).
Projection params from Aria calibration:
  [f, cx, cy, k1..k6, p1, p2, s1..s4]

Usage
-----
python undistort_aria.py \
    --input_dir  /media/.../images/aria \
    --output_dir /media/.../images/aria_undistorted
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ── Aria RGB camera calibration (Fisheye624) ──────────────────────────────────
# From: CameraCalibration(label: camera-rgb, model name: Fisheye624)
ARIA_F  = 610.91
ARIA_CX = 714.756
ARIA_CY = 704.552
ARIA_W  = 1408
ARIA_H  = 1408

# Distortion coefficients: k1..k6, p1, p2, s1..s4
ARIA_K = np.array([0.40396, -0.49252, 0.206644, 1.05278, -1.61978, 0.621768])
ARIA_P = np.array([-5.09913e-05, 0.000200007])
ARIA_S = np.array([-4.33914e-05, -0.000582142, -0.00166294, 0.000525241])


# ── Fisheye624 forward projection ────────────────────────────────────────────

def fisheye624_project(pts_3d: np.ndarray) -> np.ndarray:
    """
    Forward-project 3D rays (Nx3) to distorted pixel coordinates (Nx2)
    using the Fisheye624 model.
    """
    x = pts_3d[:, 0] / pts_3d[:, 2]
    y = pts_3d[:, 1] / pts_3d[:, 2]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)

    # Radial distortion (even powers of theta)
    k1, k2, k3, k4, k5, k6 = ARIA_K
    theta2 = theta * theta
    theta_d = theta * (1.0
                       + k1 * theta2
                       + k2 * theta2**2
                       + k3 * theta2**3
                       + k4 * theta2**4
                       + k5 * theta2**5
                       + k6 * theta2**6)

    # Scale factor (avoid division by zero at image centre)
    scale = np.where(r > 1e-8, theta_d / r, 1.0)
    xd = scale * x
    yd = scale * y

    # Tangential + thin prism distortion
    p1, p2 = ARIA_P
    s1, s2, s3, s4 = ARIA_S
    rd2 = xd**2 + yd**2
    rd4 = rd2 * rd2

    xd2 = xd + 2*p1*xd*yd + p2*(rd2 + 2*xd**2) + s1*rd2 + s2*rd4
    yd2 = yd + p1*(rd2 + 2*yd**2) + 2*p2*xd*yd + s3*rd2 + s4*rd4

    # To pixel
    u = ARIA_F * xd2 + ARIA_CX
    v = ARIA_F * yd2 + ARIA_CY

    return np.stack([u, v], axis=-1)


# ── Build undistortion remap ─────────────────────────────────────────────────

def build_undistort_map(
    out_w: int, out_h: int,
    out_fx: float, out_fy: float,
    out_cx: float, out_cy: float,
) -> tuple:
    """
    For every pixel in the undistorted (pinhole) output image,
    compute where it falls in the distorted fisheye input.

    Returns (map_x, map_y) suitable for cv2.remap().
    """
    # Grid of output pixel coordinates
    u_out = np.arange(out_w, dtype=np.float64)
    v_out = np.arange(out_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_out, v_out)

    # Back-project to 3D rays via pinhole model
    x = (uu - out_cx) / out_fx
    y = (vv - out_cy) / out_fy
    z = np.ones_like(x)

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    # Forward-project through fisheye to get source (distorted) pixel
    src_px = fisheye624_project(pts)   # (N, 2)

    map_x = src_px[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_px[:, 1].reshape(out_h, out_w).astype(np.float32)

    return map_x, map_y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Undistort raw Aria fisheye images to pinhole.")
    pa.add_argument("--input_dir",  type=Path, required=True,
                    help="Directory of raw .jpg Aria images")
    pa.add_argument("--output_dir", type=Path, required=True,
                    help="Output directory for undistorted images")
    pa.add_argument("--out_focal",  type=float, default=None,
                    help="Output focal length (default: same as Aria, 610.91). "
                         "Increase to zoom in / decrease to get wider FOV.")
    pa.add_argument("--out_size",   type=int, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="Output image size (default: same as input, 1408x1408)")
    args = pa.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_w = args.out_size[0] if args.out_size else ARIA_W
    out_h = args.out_size[1] if args.out_size else ARIA_H
    out_f = args.out_focal if args.out_focal else ARIA_F
    out_cx = out_w / 2.0
    out_cy = out_h / 2.0

    print(f"Output pinhole: {out_w}x{out_h}, f={out_f:.2f}, "
          f"cx={out_cx:.1f}, cy={out_cy:.1f}")

    # Build remap once (it's the same for every image)
    print("Building undistortion map …")
    map_x, map_y = build_undistort_map(out_w, out_h, out_f, out_f, out_cx, out_cy)

    # Process images
    images = sorted(args.input_dir.glob("*.jpg"))
    if not images:
        images = sorted(args.input_dir.glob("*.png"))
    print(f"Found {len(images)} images in {args.input_dir}")

    for img_path in tqdm(images, desc="Undistorting"):
        out_path = args.output_dir / img_path.name
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] Cannot read {img_path.name}")
            continue

        undist = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(out_path), undist)

    # Save the new intrinsics for reference
    intrinsics_file = args.output_dir / "intrinsics.txt"
    with open(intrinsics_file, "w") as f:
        f.write(f"# Undistorted pinhole intrinsics\n")
        f.write(f"model: PINHOLE\n")
        f.write(f"width: {out_w}\n")
        f.write(f"height: {out_h}\n")
        f.write(f"fx: {out_f}\n")
        f.write(f"fy: {out_f}\n")
        f.write(f"cx: {out_cx}\n")
        f.write(f"cy: {out_cy}\n")

    print(f"\n✅ Done! Undistorted images in {args.output_dir}")
    print(f"   Intrinsics saved to {intrinsics_file}")
    print(f"\n   Next steps:")
    print(f"   1. Update pipeline_blk_aria.ipynb to use 'aria_undistorted' instead of 'aria'")
    print(f"   2. The fx/fy/cx/cy in localize_blk.py are already correct (pinhole, f={out_f})")


if __name__ == "__main__":
    main()
