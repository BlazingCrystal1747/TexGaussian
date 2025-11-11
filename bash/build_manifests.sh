# 前置：运行fetch_texverse_meta.sh，把 metadata.json / caption.json / TexVerse_pbr_id_list.txt 放到 dataset/texverse/
python scripts/build_texverse_manifests.py \
  --data-root dataset/texverse --out-dir dataset/texverse/splits \
  --train-num 800 --val-num 150 --test-num 200