"""
save_images_generic.py
======================
Extract Aria RGB frames + per-frame YAML metadata from a VRS file.

Outputs:
  - frame_XXXX.jpg  (one per second of recording)
  - frame_XXXX.yaml (intrinsics, T_world_camera, T_device_camera)

Usage:
  python save_images_generic.py \
      --vrs_path          .../leader_trimmed.vrs \
      --trajectory_path   .../slam/closed_loop_trajectory.csv \
      --online_calib_path .../slam/online_calibration.jsonl \
      --output_dir        .../aria_raw
"""

import argparse
import os

import cv2
import numpy as np
import yaml
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.mps.utils import get_nearest_pose


def get_nearest_online_calib(calib_list_ts, target_ts_ns):
    return min(calib_list_ts, key=lambda pair: abs(pair[1] - target_ts_ns))[0]


def sanitize(v):
    """Recursively convert numpy / pybind types to plain Python for YAML."""
    if isinstance(v, np.ndarray):
        return v.ravel().tolist()
    if hasattr(v, 'tolist') and not isinstance(v, (str, bytes)):
        try:
            return np.asarray(v).ravel().tolist()
        except Exception:
            try:
                return v.tolist()
            except Exception:
                return str(v)
    if isinstance(v, dict):
        return {kk: sanitize(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        lst = [sanitize(x) for x in v]
        # flatten singleton-list wrappers
        while isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], list):
            lst = lst[0]
        return lst
    if hasattr(v, 'name'):
        try:
            return v.name
        except Exception:
            return str(v)
    if hasattr(v, 'value') and not isinstance(v, bool):
        try:
            return int(v)
        except Exception:
            pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def main():
    pa = argparse.ArgumentParser(description="Extract Aria images + metadata from VRS")
    pa.add_argument("--vrs_path",          type=str, required=True)
    pa.add_argument("--trajectory_path",   type=str, required=True)
    pa.add_argument("--online_calib_path", type=str, required=True)
    pa.add_argument("--output_dir",        type=str, required=True)
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize
    provider = data_provider.create_vrs_data_provider(args.vrs_path)
    closed_loop_traj = mps.read_closed_loop_trajectory(args.trajectory_path)
    online_calibs = mps.read_online_calibration(args.online_calib_path)

    online_calibs_ts = [
        (c, int(c.tracking_timestamp.total_seconds() * 1e9))
        for c in online_calibs
    ]

    stream_id = provider.get_stream_id_from_label("camera-rgb")
    first_ts_ns = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    last_ts_ns = provider.get_last_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    duration_sec = (last_ts_ns - first_ts_ns) / 1e9

    print(f"Exporting {int(duration_sec)} frames ...")

    for sec in range(int(duration_sec)):
        query_ts_ns = first_ts_ns + int(sec * 1e9)

        image_data = provider.get_image_data_by_time_ns(
            stream_id, query_ts_ns, TimeDomain.DEVICE_TIME)
        if not image_data or not image_data[0]:
            continue

        actual_ts_ns = image_data[1].capture_timestamp_ns
        img_array = image_data[0].to_numpy_array()

        current_calib_entry = get_nearest_online_calib(online_calibs_ts, actual_ts_ns)
        rgb_calib = next(
            (c for c in current_calib_entry.camera_calibs
             if c.get_label() == "camera-rgb"), None)

        pose_info = get_nearest_pose(closed_loop_traj, actual_ts_ns)

        if rgb_calib and pose_info:
            T_world_device = pose_info.transform_world_device
            T_device_camera = rgb_calib.get_transform_device_camera()
            T_world_camera = T_world_device @ T_device_camera

            metadata = {
                "timestamp_ns": actual_ts_ns,
                "image_filename": f"frame_{sec:04d}.jpg",
                "intrinsics": {
                    "model": rgb_calib.get_model_name(),
                    "focal_length": rgb_calib.get_focal_lengths().tolist(),
                    "principal_point": rgb_calib.get_principal_point().tolist(),
                    "projection_params": rgb_calib.get_projection_params().tolist(),
                    "image_size": rgb_calib.get_image_size().tolist(),
                },
                "extrinsics_world_camera": {
                    "translation": T_world_camera.translation().tolist(),
                    "quaternion_xyzw": T_world_camera.rotation().to_quat().tolist(),
                },
                "extrinsics_device_camera": {
                    "translation": T_device_camera.translation().tolist(),
                    "quaternion_xyzw": T_device_camera.rotation().to_quat().tolist(),
                },
            }
            sanitized = sanitize(metadata)

            cv2.imwrite(
                os.path.join(args.output_dir, metadata["image_filename"]),
                cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            with open(os.path.join(args.output_dir, f"frame_{sec:04d}.yaml"),
                      "w", encoding="utf-8") as f:
                yaml.dump(sanitized, f, default_flow_style=False, sort_keys=False)

    print("Done.")


if __name__ == "__main__":
    main()
