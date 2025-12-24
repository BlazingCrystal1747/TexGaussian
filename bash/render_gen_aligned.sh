#!/bin/bash

# ========================================================
# 脚本名称: render_gen_aligned.sh
# 功能: 读取 GT 渲染时保存的 transforms.json，相同视角渲染生成的贴图/模型
# 模式: 支持 beauty（PBR+HDRI）和 unlit（各贴图 emission），默认 beauty
# 需要: manifest.tsv 至少包含 obj_id、mesh、albedo，可选 rough/metal/normal/transforms
# ========================================================

# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate blender

# 2. 路径配置（按需修改）

# 生成结果的 manifest（路径支持相对路径）
MANIFEST_PATH="../experiments/exp003_test_batch/generated_manifest.tsv"

# GT 渲染结果根目录（内部包含 eval_lit/train_unlit/.../transforms.json）
GT_ROOT="../datasets/texverse_rendered"
TRANSFORMS_SUBDIR="eval_lit"   # 如果想用训练相机，改成 train_unlit

# 输出目录
OUT_ROOT="../experiments/exp003_test_batch/texverse_gen_renders"

# HDRI（beauty 模式必需，unlit 可忽略）
HDRI_PATH="../datasets/hdri/rogland_sunset_4k.exr"

echo "=============================================="
echo "Render Generated Assets with GT Cameras"
echo "Manifest:   $MANIFEST_PATH"
echo "GT Root:    $GT_ROOT/$TRANSFORMS_SUBDIR"
echo "Output:     $OUT_ROOT"
echo "HDRI:       $HDRI_PATH"
echo "Mode:       beauty"
echo "=============================================="

# 3. 执行渲染
CUDA_VISIBLE_DEVICES=0 python ./scripts/render_gen_aligned.py \
  --manifest "$MANIFEST_PATH" \
  --gt-root "$GT_ROOT" \
  --transforms-subdir "$TRANSFORMS_SUBDIR" \
  --out-root "$OUT_ROOT" \
  --mode beauty \
  --hdri "$HDRI_PATH" \
  --samples 64 \
  --hdri-strength 1.0 \
  --save-blend

echo "Done."

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
# 方式1：source conda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate texgaussian
