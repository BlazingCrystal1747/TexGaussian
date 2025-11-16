# -*- coding: utf-8 -*-
"""
texverse_tsv.py

最小 TexVerse TSV 数据集读取器。
TSV 列顺序（严格）：
    rel_glb<TAB>caption<TAB>workflow<TAB>obj_id<TAB>license

返回样本字段：
    - glb_path: str              # 绝对路径（不检查存在性）
    - text_tokens: Any           # 由外部 clip_tokenizer 产生
    - workflow_id: LongTensor    # 0=MR, 1=SG（未知一律按 MR）
    - bucket_id: LongTensor      # 0..9或10(misc)
    - caption: str
    - obj_id: str
    - license: str
"""

import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

# —— 支持“直接运行本文件”的导入回退 ——
try:
    from .bucket_rules import caption_to_bucket  # 包内相对导入（推荐用 -m 运行）
except Exception:
    import sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from core.datasets.bucket_rules import caption_to_bucket  # 直接运行时的绝对导入


class TexVerseTSV(Dataset):
    """从 TSV 读取 TexVerse 清单，供 TexGaussian 训练/验证/推理。"""

    WF_MAP: Dict[str, int] = {"MR": 0, "SG": 1}

    def __init__(self, data_root: str, tsv_path: str, clip_tokenizer):
        """
        Args:
            data_root: 数据根目录（TSV 中的 rel_glb 会与此拼接）
            tsv_path:  TSV 文件路径（绝对或相对）
            clip_tokenizer: 可调用对象，输入字符串 -> 文本 token（项目里已有封装）
        """
        if not callable(clip_tokenizer):
            raise TypeError("clip_tokenizer must be callable(text)->tokens")

        self.data_root: str = os.path.abspath(data_root)
        self.tsv_path: str = os.path.abspath(tsv_path)
        self.clip_tokenizer = clip_tokenizer

        if not os.path.exists(self.tsv_path):
            raise FileNotFoundError(f"TSV not found: {self.tsv_path}")

        self._rows: List[Dict[str, Any]] = []
        self._load_rows()

    # ---------- internal ----------

    def _load_rows(self) -> None:
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 5:
                    raise ValueError(f"Line {ln}: expected 5 columns, got {len(parts)}")

                rel_glb, caption, workflow, obj_id, license_str = parts
                wf_id = self.WF_MAP.get((workflow or "").strip().upper(), 0)

                row = {
                    "rel_glb": rel_glb,
                    "glb_path": os.path.abspath(os.path.join(self.data_root, rel_glb)),
                    "caption": caption,
                    "workflow_id": wf_id,
                    "bucket_id": caption_to_bucket(caption),
                    "obj_id": obj_id,
                    "license": license_str,
                }
                self._rows.append(row)

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self._rows[i]
        tokens = self.clip_tokenizer(row["caption"])
        sample: Dict[str, Any] = {
            "glb_path": row["glb_path"],
            "text_tokens": tokens,
            "workflow_id": torch.tensor(row["workflow_id"], dtype=torch.long),
            "bucket_id": torch.tensor(row["bucket_id"], dtype=torch.long),
            "caption": row["caption"],
            "obj_id": row["obj_id"],
            "license": row["license"],
        }
        return sample

    def __repr__(self) -> str:
        return f"TexVerseTSV(n={len(self)}, tsv='{self.tsv_path}', root='{self.data_root}')"


if __name__ == "__main__":
    """
    最小自测：
      1) 默认查找：../datasets/texverse/splits/train.tsv 作为 DEMO_TSV
      2) 可通过环境变量覆盖：
            DEMO_TSV=<path/to/splits/texverse/train.tsv>
            DEMO_DATA_ROOT=<path/to/datasets/texverse>
    """
    # 推断仓库根与默认路径
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_tsv = os.path.join(repo_root, "..", "datasets", "texverse", "splits", "train.tsv")
    default_data_root = os.path.join(repo_root, "..", "datasets", "texverse")

    demo_tsv = os.environ.get("DEMO_TSV", default_tsv)
    data_root = os.environ.get("DEMO_DATA_ROOT", default_data_root)

    if not os.path.exists(demo_tsv):
        print(f"[WARN] DEMO_TSV not found: {demo_tsv}")
        print("Set DEMO_TSV=<path/to/splits/texverse/train.tsv> if your path differs.")
        raise SystemExit(0)

    dummy_tok = lambda s: s
    ds = TexVerseTSV(data_root=data_root, tsv_path=demo_tsv, clip_tokenizer=dummy_tok)
    print(ds)
    if len(ds) > 0:
        s = ds[0]
        print("keys:", list(s.keys()))
        print("wf_id:", s["workflow_id"].item(), "bucket_id:", s["bucket_id"].item())
        print("glb_path:", s["glb_path"])
