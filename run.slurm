#!/bin/bash

#SBATCH --job-name=train_models

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --partition=gpu

#SBATCH --account=shakeri_ds6050

#SBATCH --time=08:00:00

#SBATCH --gres=gpu:a100:1

#SBATCH --output=/home/wqp7qy/ondemand/data/sys/myjobs/projects/default/1/train_models_%j.log

echo "Hello $USER, running on node $(hostname)."
echo "Job ID: $SLURM_JOB_ID"

# --- Move to project directory ---
cd /home/wqp7qy/ondemand/data/sys/myjobs/projects/default/1


# --- Load modules ---

module purge

module load apptainer

module load pytorch/2.9.0



# --- Run training inside the container ---

apptainer exec --nv $CONTAINERDIR/pytorch-2.9.0.sif python train_all_models.py \

    --scheme wide_neutral \

    --max_len 256 \

    --force_resplit \

    --model all



echo "Job finished at $(date)"
