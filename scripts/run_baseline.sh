#!/bin/bash
# Run the complete BIG-MRI baseline: preprocessing, labels, training,
# prediction, and evaluation.
#
# Set the four paths below, then run this script from the repository root:
#
#     bash scripts/run_baseline.sh
#
# Every step can also be run on its own. See preprocessing/README.md and
# src/README.md for the details of each script.
set -e

RAW_IMAGES=/path/to/BIG-MRI      # Downloaded dataset, with one folder per patient
CLINICAL=/path/to/clinical.xlsx  # Clinical table of the dataset
IMAGES=/path/to/preprocessed     # Folder for the preprocessed images
OUTPUT=/path/to/output           # Folder for the labels, models, and results

# 1. Preprocess the images.
python preprocessing/preprocess_images_v2.py \
    --input-root "$RAW_IMAGES" \
    --output-root "$IMAGES"

# 2. Create the label table, which also contains the data split.
python preprocessing/create_labels.py \
    --clinical "$CLINICAL" \
    --output "$OUTPUT/labels.csv"

# 3. Train one model per cross-validation fold.
for FOLD in 0 1 2 3; do
    python src/train.py \
        --labels "$OUTPUT/labels.csv" \
        --image-root "$IMAGES" \
        --output-dir "$OUTPUT/fold$FOLD" \
        --fold "$FOLD"
done

# 4. Predict with the ensemble of the four models.
python src/predict.py \
    --model "$OUTPUT"/fold*/best_model.pt \
    --image-root "$IMAGES" \
    --output "$OUTPUT/predictions.csv"

# 5. Evaluate on the independent test set.
python src/evaluate.py \
    --predictions "$OUTPUT/predictions.csv" \
    --labels "$OUTPUT/labels.csv" \
    --fold test \
    --output "$OUTPUT/metrics.json"
