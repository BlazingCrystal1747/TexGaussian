export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = "YiboZhang2001/TexVerse"
out  = "./dataset/texverse"
os.makedirs(out, exist_ok=True)

files = [
    "TexVerse_id_list.txt",
    "TexVerse_pbr_id_list.txt",
    "metadata.json",          # 建议一起下
    "caption.json"            # 建议一起下
]
for fn in files:
    try:
        hf_hub_download(repo_id=repo, repo_type="dataset",
                        filename=fn, local_dir=out,
                        local_dir_use_symlinks=False)
        print("[OK]", fn)
    except Exception as e:
        print("[ERR]", fn, e)
PY