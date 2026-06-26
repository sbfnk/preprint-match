#!/bin/bash
#SBATCH --job-name=rebuild-v5
#SBATCH --output=rebuild-v5_%j.out
#SBATCH --error=rebuild-v5_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# v5 model rebuild on the LSHTM GPU cluster.
# Matches the deployed v4 recipe: hard-negative contrastive fine-tuning on
# full text, then ensemble (alpha=0.1) + C=10 logistic regression, min-papers=10.
#
# Prerequisite: labeled_dataset.json (WITH full_text) present in ~/medrxiv/.
# After this job: run precompute with model-v5 + adapter, upload artifacts to
# the data-latest release, point refresh.yml at model-v5, redeploy.

cd ~/medrxiv/
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medrxiv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

echo "=== Step 1: Hard-negative fine-tune SPECTER2 (full text) ==="
date
# batch-size 8 / max-chunks 4 fit the A40 48GB (16 OOMs); see memory notes.
python3 finetune_embeddings.py \
  --input labeled_dataset_v5.json \
  --output-dir finetuned-specter2-v5/ \
  --hard-negatives \
  --epochs 3 --batch-size 8 --max-chunks 4 --lr 2e-5 \
  --temperature 0.05 --val-size 0.1 --test-size 0.2 \
  --checkpoint-every 500
echo "Fine-tune done"
date

echo ""
echo "=== Step 2: Save model-v5 (ensemble alpha=0.1, C=10, min-papers=10) ==="
date
python3 save_model.py \
  --model-dir model-v5 \
  --embeddings-dir finetuned-specter2-v5/embeddings \
  --dataset labeled_dataset_v5.json \
  --alpha 0.1 --classifier-C 10.0 --min-papers 10 --pca-components 256
echo "Saved model-v5/"
date

echo ""
echo "=== Done. Next (manual / separate job): ==="
echo "  python3 precompute.py --skip-fetch --all \\"
echo "    --model-dir model-v5 \\"
echo "    --adapter-path finetuned-specter2-v5/best_adapter \\"
echo "    --dataset labeled_dataset_v5.json"
echo "  then upload model-v5.tar.gz + adapter-v5.tar.gz to the data-latest"
echo "  release, point refresh.yml at model-v5, and redeploy."
