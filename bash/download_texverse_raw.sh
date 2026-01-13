# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-texgaussian}"
conda activate "$ENV_NAME"

CUDA_VISIBLE_DEVICES=0 python scripts/download_texverse_raw.py --total-num 6000 --resume-from-manifest