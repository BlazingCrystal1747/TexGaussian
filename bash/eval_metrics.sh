#!/bin/bash
# 计算 GT 与生成结果之间的指标，参数与 scripts/eval_metrics.py 保持一致

# 不使用 set -e，防止脚本自动退出
set -uo pipefail

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-metric}"
conda activate "$ENV_NAME"

# 默认参数，可通过环境变量或位置参数覆盖
# 位置参数: $1=EXPERIMENT_NAME, $2=METRICS(可选，优先级最高)
EXPERIMENT_NAME="${1:-exp003_test_batch_short}"
BASE_GT_DIR="${BASE_GT_DIR:-"../datasets/texverse_rendered/test"}"
BASE_GEN_DIR="${BASE_GEN_DIR:-"../experiments/${EXPERIMENT_NAME}/texverse_gen_renders"}"
LIT_SUBDIR="${LIT_SUBDIR:-"lit"}"
UNLIT_SUBDIR="${UNLIT_SUBDIR:-"unlit"}"
# 降低默认批量大小以避免 OOM (从 8 改为 4)
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-50}"
CLIP_MODEL="${CLIP_MODEL:-"ViT-B/32"}"
LONGCLIP_MODEL="${LONGCLIP_MODEL:-"../third_party/Long-CLIP/checkpoints/longclip-L.pt"}"
LONGCLIP_ROOT="${LONGCLIP_ROOT:-"../third_party/Long-CLIP"}"
LONGCLIP_CONTEXT_LENGTH="${LONGCLIP_CONTEXT_LENGTH:-248}"
OUTPUT="${OUTPUT:-"../experiments/${EXPERIMENT_NAME}/metrics_${EXPERIMENT_NAME}.json"}"
PROMPTS_FILE="${PROMPTS_FILE:-"../experiments/${EXPERIMENT_NAME}/generated_manifest.tsv"}"

# 启用 CUDA 同步调用（更稳定但更慢）
export CUDA_LAUNCH_BLOCKING=1

# 指标选择:
# - 预设: all | pixel (psnr,ssim,lpips) | dist (fid,kid) | semantic (clip)
# - 自定义逗号列表: 例如 psnr,ssim,clip
METRICS="${METRICS:-all}"

echo "=============================================="
echo "Evaluate Metrics"
echo "Experiment:    $EXPERIMENT_NAME"
echo "GT Dir:        $BASE_GT_DIR"
echo "Gen Dir:       $BASE_GEN_DIR"
echo "Lit Subdir:    $LIT_SUBDIR"
echo "Unlit Subdir:  $UNLIT_SUBDIR"
echo "Batch size:    $BATCH_SIZE"
echo "Device:        $DEVICE"
echo "KID subset:    $KID_SUBSET_SIZE"
echo "CLIP model:    $CLIP_MODEL"
echo "LongCLIP ckpt: $LONGCLIP_MODEL"
echo "LongCLIP root: $LONGCLIP_ROOT"
echo "LongCLIP ctx:  $LONGCLIP_CONTEXT_LENGTH"
echo "Prompts file:  $PROMPTS_FILE"
echo "Metrics:       $METRICS"
echo "Output:        $OUTPUT"
echo "Conda env:     $ENV_NAME"
echo "=============================================="

# 创建日志文件
LOG_DIR="../experiments/${EXPERIMENT_NAME}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_metrics_$(date +%Y%m%d_%H%M%S).log"
echo "Log file:      $LOG_FILE"
echo "=============================================="

# 设置 PyTorch 内存管理选项以提高稳定性
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 禁用 tokenizers 并行以防止死锁
export TOKENIZERS_PARALLELISM="false"
# 可选：如果仍然崩溃，取消下面这行的注释以启用同步 CUDA 调用（会更慢但更稳定）
# export CUDA_LAUNCH_BLOCKING=1

# 运行评估脚本，同时输出到终端和日志文件
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_metrics.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --base_gt_dir "$BASE_GT_DIR" \
  --base_gen_dir "$BASE_GEN_DIR" \
  --lit_subdir "$LIT_SUBDIR" \
  --unlit_subdir "$UNLIT_SUBDIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --kid_subset_size "$KID_SUBSET_SIZE" \
  --clip_model "$CLIP_MODEL" \
  --longclip_model "$LONGCLIP_MODEL" \
  --longclip_root "$LONGCLIP_ROOT" \
  --longclip_context_length "$LONGCLIP_CONTEXT_LENGTH" \
  --prompts_file "$PROMPTS_FILE" \
  --metrics "$METRICS" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete!"
else
    echo "Evaluation FAILED with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "=============================================="