#!/bin/bash
# Upload model artifacts and prediction state to a GitHub release.
# Run once after training, or whenever model changes.
#
# Usage:
#   ./scripts/upload_data.sh
#
# Requires: gh CLI authenticated

set -euo pipefail

TAG="data-latest"
REPO="sbfnk/medrxiv-journal-prediction"

echo "Packaging artifacts..."

# Model directory
tar czf /tmp/model.tar.gz -C . model/

# Adapter (find whichever exists)
ADAPTER=""
for d in finetuned-specter2-v2-hardneg/best_adapter finetuned-specter2/best_adapter; do
    if [ -d "$d" ]; then
        ADAPTER="$d"
        break
    fi
done
if [ -z "$ADAPTER" ]; then
    echo "No adapter found, skipping"
else
    tar czf /tmp/adapter.tar.gz -C . "$ADAPTER"
fi

# Prediction state (papers + embeddings for incremental refresh)
tar czf /tmp/predictions-state.tar.gz \
    predictions/papers.json \
    predictions/embeddings.npz

# Labeled dataset (needed by precompute.py)
gzip -c labeled_dataset.json > /tmp/labeled_dataset.json.gz

echo "Creating/updating release ${TAG}..."

# Create release if it doesn't exist
if ! gh release view "$TAG" --repo "$REPO" &>/dev/null; then
    gh release create "$TAG" --repo "$REPO" \
        --title "Model and prediction data" \
        --notes "Artifacts for automated refresh pipeline. Updated by scripts/upload_data.sh." \
        --latest=false
fi

# Upload assets (overwrite existing)
echo "Uploading assets..."
gh release upload "$TAG" --repo "$REPO" --clobber \
    /tmp/model.tar.gz \
    /tmp/predictions-state.tar.gz \
    /tmp/labeled_dataset.json.gz

if [ -n "$ADAPTER" ]; then
    gh release upload "$TAG" --repo "$REPO" --clobber /tmp/adapter.tar.gz
fi

echo "Done. Assets uploaded to release ${TAG}."
ls -lh /tmp/model.tar.gz /tmp/adapter.tar.gz /tmp/predictions-state.tar.gz /tmp/labeled_dataset.json.gz 2>/dev/null
