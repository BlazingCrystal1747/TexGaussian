#!/bin/bash

# ========================================================
# 脚本名称: run_render_test_gt.sh
# 功能: 使用 render_gt_dataset.py 渲染测试集的 Ground Truth
# 模式: Eval (Lit PBR + HDRI)
# ========================================================

# 1. 初始化 Conda 环境 (参考你的 render_pbr_eval.sh)
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate blender

# 2. 设置路径变量 (根据你的项目结构)

# 输入: 划分好的测试集清单
MANIFEST_PATH="../experiments/common_splits/test.tsv"

# 输出: 数据集根目录 (脚本会自动创建 eval_lit 子文件夹)
OUT_ROOT="../datasets/texverse_rendered"

# 资源: HDRI 环境贴图 (Eval 模式必须)
# 使用你参考脚本中的路径
HDRI_PATH="../datasets/hdri/rogland_sunset_4k.exr"

echo "=========================================="
echo "Start Rendering Test Set GT (Eval Mode)"
echo "Manifest: $MANIFEST_PATH"
echo "Output:   $OUT_ROOT/eval_lit"
echo "HDRI:     $HDRI_PATH"
echo "=========================================="

# 3. 执行 Python 渲染命令
# 注意：这里使用的是新脚本 render_gt_dataset.py 的参数命名
python ./scripts/render_gt_dataset.py \
  --manifest "$MANIFEST_PATH" \
  --out-root "$OUT_ROOT" \
  --mode eval \
  --hdri "$HDRI_PATH" \
  --resolution 512 \
  --views 20 \
  --samples 64 \
  --hdri-strength 1.0 \
  --seed 42 \
  --save-blend

echo "Done."