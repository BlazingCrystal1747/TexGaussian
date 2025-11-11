# 生成清单并逐文件下载 2K glb
python scripts/build_texverse_manifests.py \
  --data-root dataset/texverse --out-dir dataset/texverse/splits \
  --train-num 800 --val-num 150 --test-num 200 \
  --download