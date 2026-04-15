#!/bin/bash
#SBATCH --job-name=DS_6050_Project_group11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --account=shakeri_ds6050
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=YOUR_EMAIL@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=train_models_%j.log

echo "========================================="
echo "Started at $(date)"
echo "========================================="

echo "Hello $USER, running on node $(hostname)."
echo "Job ID: $SLURM_JOB_ID"

# ============================================================
# USER CONFIGURATION — edit these before submitting
# ============================================================
PROJECT_DIR=/path/to/your/project
CONTAINER=/path/to/your/pytorch-2.9.0.sif
# ============================================================

cd $PROJECT_DIR

module purge
module load apptainer
module load pytorch/2.9.0

echo "========================================="
echo "Starting max_len=512 at $(date)"
echo "========================================="
apptainer exec --nv $CONTAINER python train_all_models.py --scheme wide_neutral --max_len 512 --model all --seeds 42 43 44

echo "========================================="
echo "Starting max_len=256 at $(date)"
echo "========================================="
apptainer exec --nv $CONTAINER python train_all_models.py --scheme wide_neutral --max_len 256 --model bilstm distilbert bert --seeds 42 43 44

echo "========================================="
echo "Starting max_len=128 at $(date)"
echo "========================================="
apptainer exec --nv $CONTAINER python train_all_models.py --scheme wide_neutral --max_len 128 --model bilstm distilbert bert --seeds 42 43 44

echo "========================================="
echo "All runs complete at $(date)"
echo "========================================="
