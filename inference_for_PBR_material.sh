# 让 nvcc / cicc / ptxas 都在 PATH 里
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"

# 头/库路径（确保 crypt.h 等能被找到）
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

#（可选）只编译 3090/4090 的算力，缩短时间
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

CUDA_VISIBLE_DEVICES=0 python3 texture.py objaverse \
--texture_name test5 \
--ckpt_path ./assets/ckpts/PBR_model.safetensors \
--output_dir ./texture_mesh \
--save_image False \
--mesh_path ./meshes/suzanne.obj \
--text_prompt "Gold leaf, thin irregular coverage, warm speculars, subtle normal wrinkling" \
