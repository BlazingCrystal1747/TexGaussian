#!/bin/bash
# 计算 GT 与生成结果之间的指标，参数与 scripts/eval_metrics.py 保持一致

set -euo pipefail

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-metric}"
conda activate "$ENV_NAME"

# 默认参数，可通过环境变量或位置参数覆盖
# 位置参数: $1=EXPERIMENT_NAME, $2=METRICS(可选，优先级最高)
EXPERIMENT_NAME="${1:-exp003_test_batch}"
BASE_GT_DIR="${BASE_GT_DIR:-"../datasets/texverse_rendered/eval_lit"}"
BASE_GEN_DIR="${BASE_GEN_DIR:-"../experiments/${EXPERIMENT_NAME}/texverse_gen_renders"}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-50}"
CLIP_MODEL="${CLIP_MODEL:-"ViT-B/32"}"
OUTPUT="${OUTPUT:-"../experiments/${EXPERIMENT_NAME}/metrics_${EXPERIMENT_NAME}.json"}"

# 指标选择:
# - 预设: all | pixel (psnr,ssim,lpips) | dist (fid,kid) | semantic (clip)
# - 自定义逗号列表: 例如 psnr,ssim,clip
METRICS="${METRICS:-all}"

echo "=============================================="
echo "Evaluate Metrics"
echo "Experiment:    $EXPERIMENT_NAME"
echo "GT Dir:        $BASE_GT_DIR"
echo "Gen Dir:       $BASE_GEN_DIR"
echo "Batch size:    $BATCH_SIZE"
echo "Device:        $DEVICE"
echo "KID subset:    $KID_SUBSET_SIZE"
echo "CLIP model:    $CLIP_MODEL"
echo "Metrics:       $METRICS"
echo "Output:        $OUTPUT"
echo "Conda env:     $ENV_NAME"
echo "=============================================="

CUDA_VISIBLE_DEVICES=1 python scripts/eval_metrics.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --base_gt_dir "$BASE_GT_DIR" \
  --base_gen_dir "$BASE_GEN_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --kid_subset_size "$KID_SUBSET_SIZE" \
  --clip_model "$CLIP_MODEL" \
  --metrics "$METRICS" \
  --output "$OUTPUT"
