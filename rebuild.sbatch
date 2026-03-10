#!/bin/bash
#SBATCH --job-name=rebuild
#SBATCH --output=rebuild_%j.out
#SBATCH --error=rebuild_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Full rebuild pipeline: embeddings → fine-tune → evaluate → save model
# Run from ~/medrxiv/ on the LSHTM cluster
# Prerequisites: labeled_dataset.json (with full text) synced to cluster

cd ~/medrxiv/
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medrxiv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e  # Exit on first error

echo "=== Step 3: Generate baseline embeddings ==="
echo "Started: $(date)"
python3 generate_embeddings.py \
  --input labeled_dataset.json \
  --mode full-text --model specter2 \
  --output-dir embeddings/full-text-specter2 \
  --stride 256
echo "Baseline embeddings done: $(date)"

echo ""
echo "=== Step 4: Fine-tune SPECTER2 ==="
echo "Started: $(date)"
python3 finetune_embeddings.py \
  --input labeled_dataset.json \
  --output-dir finetuned-specter2-v2/ \
  --epochs 3 --batch-size 8 --max-chunks 4 --lr 2e-5 \
  --val-size 0.1 --test-size 0.2 \
  --checkpoint-every 500
echo "Fine-tuning done: $(date)"

echo ""
echo "=== Step 5: Evaluation ==="
echo "Started: $(date)"

# kNN with original embeddings
python3 evaluate_knn.py \
  --embeddings-dir embeddings/full-text-specter2 \
  --val-size 0.1 \
  --output knn_results_orig_v2.json
echo "kNN (original) done"

# kNN with fine-tuned embeddings
python3 evaluate_knn.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 \
  --output knn_results_ft_v2.json
echo "kNN (fine-tuned) done"

# Classifier
python3 train_classifier.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 \
  --output classifier_results_v2.json
echo "Classifier done"

# Ensemble (both methods)
python3 ensemble_predict.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 --method both \
  --output ensemble_results_v2.json
echo "Ensemble done"

# Ensemble with min-papers=10
python3 ensemble_predict.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 --method interpolation --min-papers 10 \
  --output ensemble_results_ft_v2_min10.json
echo "Ensemble (min-papers=10) done"

# Calibration
python3 calibrate.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --output calibration_results_v2.json
echo "Calibration done"

echo ""
echo "=== Evaluation complete: $(date) ==="
echo "Check results files for optimal alpha, then run save_model.py manually"
echo ""
echo "Next steps (manual):"
echo "  1. Check ensemble_results_v2.json for best alpha"
echo "  2. python3 save_model.py --model-dir model-v2 \\"
echo "       --embeddings-dir finetuned-specter2-v2/embeddings \\"
echo "       --alpha <best> --min-papers 10"
echo "  3. python3 precompute.py --skip-fetch --all \\"
echo "       --model-dir model-v2 \\"
echo "       --adapter-path finetuned-specter2-v2/best_adapter"
