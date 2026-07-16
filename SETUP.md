# CoMind Aria ↔ BLK Alignment Pipeline — Setup & Run Guide

Pipeline: extract Aria RGB frames from VRS → undistort (fisheye → pinhole) → SfM on
Aria frames (hloc: SuperPoint + SuperGlue + NetVLAD) → localize BLK images →
compute `T_ariaWorld_from_blkWorld` → ICP-refine BLK scan into the Aria frame.

Entry point: [`Hierarchical-Localization/run_pipeline_cluster_leader.sh`](Hierarchical-Localization/run_pipeline_cluster_leader.sh)
(copies inputs to `$TMPDIR`, runs `run_full_pipeline_leader.sh` there, copies outputs back).

---

## 1. Workspace setup

```bash
mkdir -p <workspace>/cvg_comind && cd <workspace>/cvg_comind

# hloc fork — contains the pipeline scripts
# --recursive is required: it pulls third_party/ submodules, including
# SuperGluePretrainedNetwork with the bundled SuperPoint/SuperGlue weights.
git clone --recursive git@github.com:manthan99/Hierarchical-Localization.git

# If already cloned without --recursive:
#   cd Hierarchical-Localization && git submodule update --init --recursive

# OPTIONAL: projectaria_tools source. The pipeline only needs the Python API,
# which is installed from PyPI as a prebuilt wheel (no source build needed).
# Clone only if you want the C++ tools / to build from source.
git clone git@github.com:facebookresearch/projectaria_tools.git
```

> ⚠️ **Note (as of 2026-07-16):** `run_pipeline_cluster_leader.sh`,
> `run_full_pipeline_leader.sh`, and the `process_dataset*.sh` SLURM wrappers are
> currently **untracked** in the fork — a fresh clone will not contain them until
> they are committed and pushed to `manthan99/Hierarchical-Localization`.

### Expected data layout

```
<data_root>/<traj_uuid>/
├── trimmed_vrs/leader_trimmed.vrs
└── mps_leader_trimmed_vrs/slam/
    ├── closed_loop_trajectory.csv
    ├── online_calibration.jsonl
    └── semidense_points.csv.gz

<blk_root>/<traj_uuid>/blk/
├── images/*.jpg + *.yml      # BLK images with camera poses
└── *.pcd or *.ply            # BLK laser scan (top level of blk/)
```

Paths used in previous runs:
- `data_root`   = `/cluster/work/cvg/data/CoMind_clean`
- `blk_root`    = `/cluster/work/cvg/data/CoMind_clean/patelm/blk_data`
- `output_root` = `/cluster/work/cvg/data/CoMind_clean/patelm/output_leader`

---

## 2. Conda environment

Requires miniconda/mamba. On the ETH cluster, load `eth_proxy` first for internet
access (needed by pip and the git-based lightglue dependency):

```bash
module load eth_proxy
conda create -n hloc python=3.11 -y
conda activate hloc

# PyTorch with CUDA 12.4 (RTX 4090 / cluster GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Aria Python API (prebuilt wheel — no source build required), Open3D, YAML
pip install projectaria-tools==2.1.1 open3d pyyaml

# hloc + its requirements (pycolmap, kornia, opencv, h5py, lightglue, …)
cd <workspace>/cvg_comind/Hierarchical-Localization
pip install -e .
```

Notes:
- **No COLMAP binary needed** — the `pycolmap` pip wheel covers SfM and localization.
- SuperPoint/SuperGlue weights ship with the `third_party/SuperGluePretrainedNetwork`
  submodule (`models/weights/*.pth`) — nothing to download.
- **NetVLAD weights** (`Pitts30K_struct.mat`, ~250 MB) are downloaded on first use
  from `https://cvg-data.inf.ethz.ch` into `torch.hub.get_dir()/netvlad/`
  (currently cached at `/cluster/work/rsl/patelm/.cache/hub/netvlad/`). To
  pre-download so compute jobs don't need internet:
  ```bash
  python -c "from hloc import extractors; from hloc.utils.base_model import dynamic_load; dynamic_load(extractors, 'netvlad')({'name': 'netvlad'})"
  ```

### Verify

```bash
conda activate hloc
python -c "
import torch, cv2, open3d, pycolmap, projectaria_tools.core.mps
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction, localize_sfm
print('cuda:', torch.cuda.is_available())
print('OK')"
```

Verified working set (2026-07-16): python 3.11, torch 2.6.0+cu124,
numpy 2.4.4, opencv 5.0.0, open3d 0.19.0, pycolmap 4.1.0,
projectaria-tools 2.1.1, hloc 1.5.

---

## 3. Running the pipeline (single trajectory)

```bash
conda activate hloc
module load eth_proxy   # only needed if NetVLAD weights are not yet cached

cd <workspace>/cvg_comind/Hierarchical-Localization
./run_pipeline_cluster_leader.sh <data_root> <blk_root> <traj_uuid> <output_root>

# Example:
./run_pipeline_cluster_leader.sh \
    /cluster/work/cvg/data/CoMind_clean \
    /cluster/work/cvg/data/CoMind_clean/patelm/blk_data \
    f0b49334-565a-4d30-a010-15d92e511c9a \
    /cluster/work/cvg/data/CoMind_clean/patelm/output_leader
```

Requirements & behavior:
- **GPU required** (SuperPoint/SuperGlue/NetVLAD inference). Previous jobs requested
  1 GPU with 23 GB (`--gres=gpumem:23G`); a 24 GB RTX 4090 works.
- Uses `$TMPDIR` as the working directory when set (SLURM sets it per job; request
  enough with `--tmp`, previously 200 GB). Outside SLURM it falls back to `mktemp -d`
  — note that fallback dir is **not** auto-cleaned.
- Steps are resumable: each step is skipped if its output already exists in the
  work dir (relevant when rerunning with a persistent output dir via
  `run_full_pipeline_leader.sh` directly).

### Outputs (`<output_root>/<traj_uuid>/`)

```
aria_raw/                     # extracted frames + per-frame YAML (pose, intrinsics)
aria_undistorted/             # pinhole-undistorted frames
semidense_points.ply          # Aria MPS semidense point cloud
alignment/
├── sfm/                      # hloc/COLMAP reconstruction
├── T_ariaWorld_from_blkWorld.npy
└── icp_stats.json
blk_scan_aria_leader.ply      # BLK scan transformed into Aria world frame
```

---

## 4. SLURM batch processing

*To be documented (follow-up).* The existing `process_dataset*.sh` wrappers show the
pattern: `sbatch` with 32 CPUs × 4000 MB, `--tmp=200G`, `--gpus=1 --gres=gpumem:23G`,
8 h walltime, `module load eth_proxy`, activate the environment, then loop
`run_pipeline_cluster_leader.sh` over trajectory UUIDs. They still reference the old
`.venv` — replace with `conda activate hloc`.
