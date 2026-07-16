#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=200G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23G
#SBATCH --job-name=hloc
#SBATCH --output=/cluster/work/rsl/patelm/ws/cvg/cvg_comind/Hierarchical-Localization/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/ws/cvg/cvg_comind/Hierarchical-Localization/slurm_output/%x_%j.err

# Load modules and environment
source ~/.bashrc
module load eth_proxy
cd /cluster/work/rsl/patelm/ws/cvg/cvg_comind
conda activate hloc

# Define the list of trajectories as a bash array

DATASET_LIST=(
    "0b9598ec-b2a0-4211-86c3-d77d2c82cb8f"
    "11bd710a-d8f3-4e3c-88e4-d97d7bc8b3f5"
    "313483e1-e51a-44f6-8e73-3d80d56fe08e"
    "39097cb5-de7a-49d4-8217-b2c30e19a3ae"
)

# DATASET_LIST=(
#     "0b9598ec-b2a0-4211-86c3-d77d2c82cb8f"
#     "11bd710a-d8f3-4e3c-88e4-d97d7bc8b3f5"
#     "313483e1-e51a-44f6-8e73-3d80d56fe08e"
#     "4621cc65-f05d-46fb-aea2-7245d513e959"
#     "482e5b50-8c69-4d03-b6d8-e4932a981711"
#     "4fbe9724-b960-4d6c-a016-6b545557b7de"
#     "87f944b2-ff9d-47fe-bf59-6c374f139356"
#     "8f8b5bf7-a8ab-41f2-a912-d42139e603e1"
#     "b19c41f5-719d-4a93-92e2-0555cad8a607"
#     "f794039a-f9c7-49f1-82f4-39f93c561523"
# )

# DATASET_LIST=(
#     "0b9598ec-b2a0-4211-86c3-d77d2c82cb8f"
#     "11bd710a-d8f3-4e3c-88e4-d97d7bc8b3f5"
#     "1326e688-9e84-4d1a-aef7-a588ca2cc47d" # Issue
#     "21c13149-ca54-45dc-94a1-bd74a1c8a27e" # ISSUE
#     "313483e1-e51a-44f6-8e73-3d80d56fe08e"
#     "39097cb5-de7a-49d4-8217-b2c30e19a3ae"
#     "4621cc65-f05d-46fb-aea2-7245d513e959"
#     "482e5b50-8c69-4d03-b6d8-e4932a981711"
#     "4a93a8f7-305e-489b-b4e7-c31253507ade"
#     "4fbe9724-b960-4d6c-a016-6b545557b7de"
#     "58dd70af-0c19-486e-9bdc-24b5c22b49e6"
#     "63f25fa8-4573-48da-84e4-4cd5ea1c2701"
#     "7b04b3ef-43ed-4c46-ab3d-0d3baa25352e"
#     "87f944b2-ff9d-47fe-bf59-6c374f139356"
#     "8ee6c694-38be-4a90-ad5a-75ef4449e972"
#     "8f8b5bf7-a8ab-41f2-a912-d42139e603e1"
#     "9a667685-1cdd-4097-9138-9138faf388b2"
#     "b19c41f5-719d-4a93-92e2-0555cad8a607"
#     "b1ade277-a777-43cf-92a6-ede187252a5c"
#     "c4944bc8-994d-4ed2-9d99-9812c3275bf0"
#     "d1c6ae3a-3f7b-4193-9923-2e029d9513fd" # ISSUE
#     "d7f489e1-227a-4de7-aa07-7b7d37afd139"
#     "d8fcc3dd-e9bb-42e2-b8fd-b2ec5c5e54f1"
#     "e5a6754a-5149-4cc7-821a-bb967d8857bf"
#     "f0b49334-565a-4d30-a010-15d92e511c9a"
#     "f794039a-f9c7-49f1-82f4-39f93c561523"
# )

# Root directory where trajectories live
DATA_ROOT="/cluster/work/cvg/data/CoMind_clean"
BLK_ROOT="/cluster/work/cvg/data/CoMind_clean/patelm/blk_data"
# Final output storage
OUT_ROOT="/cluster/work/cvg/data/CoMind_clean/patelm/output_leader"

cd Hierarchical-Localization

# Loop over the array
for traj in "${DATASET_LIST[@]}"; do
    echo "--------------------------------------------------------"
    echo "Processing $traj"
    
    # Call the cluster wrapper script
    # Args: <data_root> <traj_name> <output_root>
    ./run_pipeline_cluster_leader.sh "$DATA_ROOT" "$BLK_ROOT" "$traj" "$OUT_ROOT"
done

echo "Batch processing finished."


