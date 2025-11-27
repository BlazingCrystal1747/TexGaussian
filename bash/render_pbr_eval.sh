# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
# 方式1：source conda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate blender

python scripts/render_pbr_eval.py -- \
  --manifest ../experiments/common_splits/test.tsv \
  --out-dir ../experiments/test/textrures \
  --hdri ../datasets/hdri/rogland_sunset_4k.exr \
  --res 512 \
  --save-blend

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
# 方式1：source conda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate texgaussian