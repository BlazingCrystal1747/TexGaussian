#（可选）只编译 3090/4090 的算力，缩短时间 #但其实没用
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

# 让 nvcc / cicc / ptxas 都在 PATH 里
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"

# 头/库路径（确保 crypt.h 等能被找到）
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# ================= 配置区 =================

# 实验名称 (将作为文件夹名创建在 experiments 下)
EXP_NAME="exp001_test_batch"

# TSV 路径 (建议绝对路径，或相对于 texGaussian 的路径)
BATCH_TSV="../experiments/common_splits/test.tsv"

# 输出根目录 (指向 project_root/experiments/EXP_NAME)
# 假设脚本在 project_root/texGaussian 下运行
OUTPUT_ROOT="../experiments/${EXP_NAME}"

# ==========================================

echo "Starting Batch Inference..."
echo "Config: ${BATCH_TSV}"
echo "Output: ${OUTPUT_ROOT}"
echo "Textures will be stored under: ${OUTPUT_ROOT}/textures"

CUDA_VISIBLE_DEVICES=0 python3 texture.py objaverse \
--tsv_path "${BATCH_TSV}" \
--ckpt_path ./assets/ckpts/PBR_model.safetensors \
--output_dir "${OUTPUT_ROOT}" \
--save_image False
