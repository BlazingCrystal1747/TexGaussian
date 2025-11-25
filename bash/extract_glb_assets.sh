# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
# 方式1：source conda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate blender

python scripts/extract_glb_assets.py \
  --tsv "../datasets/texverse/downloaded_manifest.tsv" \
  --data-root "../datasets/texverse" \
  --out-root "../datasets/texverse_extracted"

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
# 方式1：source conda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate texgaussian