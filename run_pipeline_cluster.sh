#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline_cluster.sh
# ═══════════════════════════════════════════════════════════════════════════════
# Copy input data to $TMPDIR, run the full alignment pipeline there,
# then copy final outputs back to the output directory.
#
# Usage:
#   ./run_pipeline_cluster.sh <data_root> <traj_name> <output_root>
#
# Example:
#   ./run_pipeline_cluster.sh /cluster/work/cvg/data/CoMind_clean \
#       f0b49334-565a-4d30-a010-15d92e511c9a \
#       /cluster/work/cvg/data/processed
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <data_root> <traj_name> <output_root>"
    exit 1
fi

DATA_ROOT="$1"
TRAJ="$2"
OUTPUT_ROOT="$3"

TRAJ_DIR="${DATA_ROOT}/${TRAJ}"
HLOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use $TMPDIR if set (cluster), otherwise create a temp dir
WORK="${TMPDIR:-$(mktemp -d)}"
WORK_TRAJ="${WORK}/${TRAJ}"

echo "═══════════════════════════════════════════════════════════"
echo "  Trajectory:  ${TRAJ}"
echo "  Input:       ${TRAJ_DIR}"
echo "  Output:      ${OUTPUT_ROOT}/${TRAJ}"
echo "  TMPDIR work: ${WORK_TRAJ}"
echo "═══════════════════════════════════════════════════════════"

# ── Step A: Copy inputs to TMPDIR ─────────────────────────────────────────────
echo ""
echo ">>> Copying inputs to TMPDIR ..."

mkdir -p "${WORK_TRAJ}"

# 1. VRS file
echo "  trimmed_vrs/leader_trimmed.vrs"
mkdir -p "${WORK_TRAJ}/trimmed_vrs"
cp "${TRAJ_DIR}/trimmed_vrs/leader_trimmed.vrs" \
   "${WORK_TRAJ}/trimmed_vrs/leader_trimmed.vrs"

# 2. MPS SLAM data
echo "  mps_leader_trimmed_vrs/slam/"
mkdir -p "${WORK_TRAJ}/mps_leader_trimmed_vrs/slam"
cp "${TRAJ_DIR}/mps_leader_trimmed_vrs/slam/closed_loop_trajectory.csv" \
   "${WORK_TRAJ}/mps_leader_trimmed_vrs/slam/"
cp "${TRAJ_DIR}/mps_leader_trimmed_vrs/slam/online_calibration.jsonl" \
   "${WORK_TRAJ}/mps_leader_trimmed_vrs/slam/"
cp "${TRAJ_DIR}/mps_leader_trimmed_vrs/slam/semidense_points.csv.gz" \
   "${WORK_TRAJ}/mps_leader_trimmed_vrs/slam/"

# 3. BLK images + yml + pcd
echo "  blk/"
mkdir -p "${WORK_TRAJ}/blk/images"
cp "${TRAJ_DIR}"/blk/images/*.jpg "${WORK_TRAJ}/blk/images/" 2>/dev/null || true
cp "${TRAJ_DIR}"/blk/images/*.yml "${WORK_TRAJ}/blk/images/" 2>/dev/null || true
# Copy BLK point cloud (pcd/ply at top level of blk/)
cp "${TRAJ_DIR}"/blk/*.pcd "${WORK_TRAJ}/blk/" 2>/dev/null || true
cp "${TRAJ_DIR}"/blk/*.ply "${WORK_TRAJ}/blk/" 2>/dev/null || true

echo ">>> Copy done."
echo ""

# ── Step B: Run the full pipeline in TMPDIR ───────────────────────────────────
echo ">>> Running pipeline in TMPDIR ..."

# Use WORK as both data_root and output_root so everything stays in TMPDIR
"${HLOC_DIR}/run_full_pipeline.sh" "${WORK}" "${TRAJ}" "${WORK}"

echo ""
echo ">>> Pipeline done."

# ── Step C: Copy outputs back ─────────────────────────────────────────────────
echo ""
echo ">>> Copying outputs to ${OUTPUT_ROOT}/${TRAJ} ..."

FINAL_OUT="${OUTPUT_ROOT}/${TRAJ}"
mkdir -p "${FINAL_OUT}"

# Copy the key outputs
cp -r "${WORK_TRAJ}/aria_raw"           "${FINAL_OUT}/" 2>/dev/null || true
cp -r "${WORK_TRAJ}/aria_undistorted"   "${FINAL_OUT}/" 2>/dev/null || true
cp -r "${WORK_TRAJ}/alignment"          "${FINAL_OUT}/" 2>/dev/null || true
cp    "${WORK_TRAJ}/semidense_points.ply"       "${FINAL_OUT}/" 2>/dev/null || true
cp    "${WORK_TRAJ}/blk_scan_aria_leader.ply"   "${FINAL_OUT}/" 2>/dev/null || true

echo ">>> Done."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  COMPLETE: ${TRAJ}"
echo "  Outputs at: ${FINAL_OUT}"
echo "═══════════════════════════════════════════════════════════"
