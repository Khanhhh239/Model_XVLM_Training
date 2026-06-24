#!/bin/bash
# Quick start script để test training pipeline local trước khi chạy trên Kaggle

set -e  # Exit on error

echo "========================================="
echo "STAGE 1 Training - Quick Start Test"
echo "========================================="

# 1. Check requirements
echo ""
echo "[1/5] Checking Python environment..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers installed')" || echo "⚠️  Missing transformers"
python -c "import albumentations; print(f'✓ Albumentations installed')" || echo "⚠️  Missing albumentations"

# 2. Check data files
echo ""
echo "[2/5] Checking data files..."
DATA_DIR="data"
REQUIRED_FILES=(
    "train_30k_hard.jsonl"
    "train_30k_hard_vitpose.json"
    "boxes_30k.jsonl"
    "checkpoints/best.pth"
)

MISSING_FILES=0
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$FILE" ]; then
        echo "  ✓ $DATA_DIR/$FILE"
    else
        echo "  ⚠️  MISSING: $DATA_DIR/$FILE"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "❌ Missing $MISSING_FILES required file(s)!"
    echo "   Please prepare data according to KAGGLE_SETUP_GUIDE.md"
    exit 1
fi

# 3. Check image folder
echo ""
echo "[3/5] Checking image folder..."
if [ -d "$DATA_DIR/train_webp" ]; then
    NUM_IMAGES=$(find "$DATA_DIR/train_webp" -name "*.webp" | wc -l)
    echo "  ✓ Found $NUM_IMAGES .webp images"
    if [ $NUM_IMAGES -lt 30000 ]; then
        echo "  ⚠️  Expected ~30K images, found only $NUM_IMAGES"
    fi
else
    echo "  ❌ Missing $DATA_DIR/train_webp folder!"
    exit 1
fi

# 4. Validate config
echo ""
echo "[4/5] Validating config..."
python -c "
import yaml
from pathlib import Path

config_path = 'configs/stage1_30k_kaggle_t4.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

print(f'  ✓ Config loaded from {config_path}')
print(f'    - Batch size: {cfg[\"train\"][\"batch_size\"]}')
print(f'    - Epochs: {cfg[\"optim\"][\"epochs\"]}')
print(f'    - XBM enabled: {cfg[\"loss\"][\"xbm_enabled\"]}')
print(f'    - Box head: {cfg[\"model\"][\"bbox_enabled\"]}')
print(f'    - Anomaly head: {cfg[\"model\"][\"anomaly_enabled\"]}')
"

# 5. Run sanity check (overfit one batch)
echo ""
echo "[5/5] Running sanity check (overfit one batch)..."
echo "      This will take ~5 minutes..."
echo ""

python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth \
    --overfit-one-batch

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SUCCESS! Pipeline is working correctly."
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Review config: configs/stage1_30k_kaggle_t4.yaml"
    echo "  2. (Optional) Test full training locally for 1 epoch:"
    echo "     python scripts/train.py --config configs/stage1_30k_kaggle_t4.yaml --init-from data/checkpoints/best.pth"
    echo "  3. Upload to Kaggle and run full training (see KAGGLE_SETUP_GUIDE.md)"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ FAILED! Check logs above for errors."
    echo "========================================="
    exit 1
fi
