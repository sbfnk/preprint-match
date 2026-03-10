#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --output=precompute_%j.out
#SBATCH --error=precompute_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd ~/medrxiv

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medrxiv

# Embed and score (papers.json already fetched locally)
python3 precompute.py --skip-fetch --all
