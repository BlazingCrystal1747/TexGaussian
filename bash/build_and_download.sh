# 生成清单并逐文件下载 2K glb
python scripts/build_texverse_manifests.py \
  --data-root ../datasets/texverse --out-dir ../datasets/texverse/splits \
  --train-num 1600 --val-num 300 --test-num 400 \
  --download
