#!/bin/bash

# ========================================================
# 脚本名称: render_gt_dataset_train.sh
# 功能: 使用 render_gt_dataset.py 渲染训练集的 Ground Truth (Unlit Emission)
# 模式: Train (无光照，输出 albedo/rough/metal/normal)
# ========================================================

# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate blender

# 2. 设置路径变量

# 输入: 划分好的训练集清单
MANIFEST_PATH="../experiments/common_splits/test.tsv"

# 输出: 数据集根目录 (脚本会自动创建 train_unlit 子文件夹)
OUT_ROOT="../datasets/texverse_rendered"

echo "=========================================="
echo "Start Rendering Train Set GT (Unlit Mode)"
echo "Manifest: $MANIFEST_PATH"
echo "Output:   $OUT_ROOT/train_unlit"
echo "=========================================="

# 3. 执行 Python 渲染命令
python ./scripts/render_gt_dataset.py \
  --manifest "$MANIFEST_PATH" \
  --out-root "$OUT_ROOT" \
  --mode train \
  --resolution 512 \
  --views 64 \
  --seed 42 \
  --save-blend

echo "Done."

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate texgaussian
