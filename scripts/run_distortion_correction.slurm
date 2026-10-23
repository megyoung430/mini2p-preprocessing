#!/bin/bash
#SBATCH --job-name=mini2p-distortion
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=distortion-%j.out
#SBATCH --error=distortion-%j.err

set -euo pipefail

# Edit these four values for the cluster and recording.
REPOSITORY_DIR="/path/to/mini2p-preprocessing"
CONDA_ENV="mini2p-preprocessing"
INPUT_TIFF="/path/to/run-01/input_dynamic_bidiphase_destriped.tif"
OUTPUT_TIFF="/path/to/run-01/input_dynamic_bidiphase_destriped_undistorted.tif"
TRANSFORM_MAT="/path/to/transform_matrix312_A049.mat"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python "${REPOSITORY_DIR}/scripts/distortion_correct_full_recording.py" \
  --input "${INPUT_TIFF}" \
  --output "${OUTPUT_TIFF}" \
  --transform "${TRANSFORM_MAT}" \
  --channels 2 \
  --z-um 0 \
  --order 1 \
  --crop-padding 1 \
  --overwrite
