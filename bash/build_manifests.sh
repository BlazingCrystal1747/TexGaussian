# 前置：运行fetch_texverse_meta.sh，把 metadata.json / caption.json / TexVerse_pbr_id_list.txt 放到 ../datasets/texverse/
python scripts/build_texverse_manifests.py \
  --data-root ../datasets/texverse --out-dir ../datasets/texverse/splits \
  --train-num 1600 --val-num 300 --test-num 400
