#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_full_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline: Aria VRS → images → undistort → SfM → BLK localization → TF → ICP
#
# Usage:
#   ./run_full_pipeline.sh <data_root> <traj_name> <output_root>
#
# Example:
#   ./run_full_pipeline.sh /cluster/work/cvg/data/CoMind_clean \
#       f0b49334-565a-4d30-a010-15d92e511c9a \
#       /cluster/work/cvg/data/processed
#
# Output structure:
#   <output_root>/<traj_name>/
#   ├── aria_raw/           # Step 0: extracted frames + YAML
#   ├── aria_undistorted/   # Step 2: undistorted frames
#   ├── semidense_points.ply  # Step 1: Aria point cloud
#   ├── alignment/          # Step 3: SfM + localization
#   │   ├── sfm/
#   │   ├── T_ariaWorld_from_blkWorld.npy
#   │   └── icp_stats.json
#   └── blk_scan_aria_leader.ply  # Step 4: BLK in Aria frame
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <data_root> <traj_name> <output_root>"
    echo ""
    echo "  data_root   : parent directory containing trajectory folders (input)"
    echo "                e.g. /cluster/work/cvg/data/CoMind_clean"
    echo "  traj_name   : trajectory UUID folder name"
    echo "                e.g. f0b49334-565a-4d30-a010-15d92e511c9a"
    echo "  output_root : parent directory for processed outputs"
    echo "                e.g. /cluster/work/cvg/data/processed"
    exit 1
fi

DATA_ROOT="$1"
TRAJ="$2"
OUTPUT_ROOT="$3"

TRAJ_DIR="${DATA_ROOT}/${TRAJ}"     # input data
OUT_DIR="${OUTPUT_ROOT}/${TRAJ}"    # all outputs go here

# Where this script lives (= the hloc repo root)
HLOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Input paths ───────────────────────────────────────────────────────────────
# VRS_FILE="${TRAJ_DIR}/trimmed_vrs/leader_trimmed.vrs"
# MPS_DIR="${TRAJ_DIR}/mps_leader_trimmed_vrs/slam"
# TRAJECTORY_CSV="${MPS_DIR}/closed_loop_trajectory.csv"
# ONLINE_CALIB="${MPS_DIR}/online_calibration.jsonl"
# SEMIDENSE_CSV="${MPS_DIR}/semidense_points.csv.gz"
# BLK_IMG_DIR="${TRAJ_DIR}/blk/images"
VRS_FILE="${TRAJ_DIR}/trimmed_vrs/helper_trimmed.vrs"
MPS_DIR="${TRAJ_DIR}/multislam_output/helper_trimmed/slam"
TRAJECTORY_CSV="${MPS_DIR}/closed_loop_trajectory.csv"
ONLINE_CALIB="${MPS_DIR}/online_calibration.jsonl"
SEMIDENSE_CSV="${MPS_DIR}/semidense_points.csv.gz"
BLK_IMG_DIR="${TRAJ_DIR}/blk/images"

# ── Output paths (all under OUT_DIR) ──────────────────────────────────────────
ARIA_RAW="${OUT_DIR}/aria_raw"
ARIA_UNDIST="${OUT_DIR}/aria_undistorted"
SEMIDENSE_PLY="${OUT_DIR}/semidense_points.ply"
ALIGN_OUT="${OUT_DIR}/alignment"

mkdir -p "${OUT_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Missing required file: $1"
        exit 1
    fi
}

echo "Pipeline for: ${TRAJ}"
echo "Input data:   ${TRAJ_DIR}"
echo "Output dir:   ${OUT_DIR}"
echo "HLOC dir:     ${HLOC_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 0: Extract Aria RGB images + metadata from VRS
# ══════════════════════════════════════════════════════════════════════════════
log "Step 0: Extract Aria images from VRS"

if [ -d "${ARIA_RAW}" ] && [ "$(find "${ARIA_RAW}" -maxdepth 1 -name '*.jpg' | head -1)" ]; then
    echo "  → Already extracted ($(ls "${ARIA_RAW}"/*.jpg | wc -l) images), skipping."
else
    check_file "${VRS_FILE}"
    check_file "${TRAJECTORY_CSV}"
    check_file "${ONLINE_CALIB}"

    python3 "${HLOC_DIR}/save_images_generic.py" \
        --vrs_path          "${VRS_FILE}" \
        --trajectory_path   "${TRAJECTORY_CSV}" \
        --online_calib_path "${ONLINE_CALIB}" \
        --output_dir        "${ARIA_RAW}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Save semidense point cloud
# ══════════════════════════════════════════════════════════════════════════════
log "Step 1: Save Aria semidense point cloud"

if [ -f "${SEMIDENSE_PLY}" ]; then
    echo "  → Already exists, skipping."
else
    check_file "${SEMIDENSE_CSV}"

    python3 "${HLOC_DIR}/save_pcd_generic.py" \
        --input_path  "${SEMIDENSE_CSV}" \
        --output_path "${SEMIDENSE_PLY}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Undistort Aria images (fisheye → pinhole)
# ══════════════════════════════════════════════════════════════════════════════
log "Step 2: Undistort Aria images"

if [ -d "${ARIA_UNDIST}" ] && [ "$(find "${ARIA_UNDIST}" -maxdepth 1 -name '*.jpg' | head -1)" ]; then
    echo "  → Already undistorted ($(ls "${ARIA_UNDIST}"/*.jpg | wc -l) images), skipping."
else
    python3 "${HLOC_DIR}/undistort_aria.py" \
        --input_dir  "${ARIA_RAW}" \
        --output_dir "${ARIA_UNDIST}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Compute T_ariaWorld_from_blkWorld
# ══════════════════════════════════════════════════════════════════════════════
log "Step 3: Compute alignment (SfM + localize + TF)"

mkdir -p "${ALIGN_OUT}"

python3 "${HLOC_DIR}/fast_align_pipeline.py" \
    --aria_img_dir  "${ARIA_UNDIST}" \
    --aria_yaml_dir "${ARIA_RAW}" \
    --blk_dir       "${BLK_IMG_DIR}" \
    --output_dir    "${ALIGN_OUT}" \
    --aria_subsample 10 \
    --num_retrieval 5

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: ICP — transform BLK scan into Aria frame and refine
# ══════════════════════════════════════════════════════════════════════════════
log "Step 4: ICP alignment (BLK → Aria frame)"

BLK_PCD=$(find "${TRAJ_DIR}/blk" -maxdepth 1 -name "*.pcd" -o -name "*.ply" | head -1)
if [ -z "${BLK_PCD}" ]; then
    echo "  WARNING: No BLK .pcd/.ply found in ${TRAJ_DIR}/blk/"
    echo "  Skipping ICP step."
else
    ALIGNED_BLK="${OUT_DIR}/blk_scan_aria_leader.ply"

    python3 "${HLOC_DIR}/icp_align.py" \
        --blk_pcd       "${BLK_PCD}" \
        --aria_pcd      "${SEMIDENSE_PLY}" \
        --transform     "${ALIGN_OUT}/T_ariaWorld_from_blkWorld.npy" \
        --output_pcd    "${ALIGNED_BLK}" \
        --output_stats  "${ALIGN_OUT}/icp_stats.json" \
        --voxel_size    0.01 \
        --icp_threshold  0.1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
log "✅ PIPELINE COMPLETE"
echo ""
echo "  Trajectory:     ${TRAJ}"
echo "  Input data:     ${TRAJ_DIR}"
echo "  Output:         ${OUT_DIR}"
echo ""
echo "  Aria raw:       ${ARIA_RAW}"
echo "  Aria undist:    ${ARIA_UNDIST}"
echo "  Semidense PCD:  ${SEMIDENSE_PLY}"
echo "  Alignment:      ${ALIGN_OUT}"
echo "  Transform:      ${ALIGN_OUT}/T_ariaWorld_from_blkWorld.npy"
if [ -n "${BLK_PCD:-}" ] && [ -f "${ALIGNED_BLK:-}" ]; then
echo "  BLK in Aria:    ${ALIGNED_BLK}"
echo "  ICP stats:      ${ALIGN_OUT}/icp_stats.json"
fi
