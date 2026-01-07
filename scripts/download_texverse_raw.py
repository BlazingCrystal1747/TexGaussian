#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[Step 1] download_texverse_filtered.py
功能：基于 metadata.json 筛选 Metalness PBR 资产，直接下载 1k 版本 glb。
"""

import os
import sys
import json
import argparse
import random
import csv
import struct
import re

# 确保能找到 repo 根目录下的 external.clip
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 兼容 packaging 新版本不自动暴露 version 属性的问题
try:
    import packaging
    import packaging.version as _packaging_version
    if not hasattr(packaging, "version"):
        packaging.version = _packaging_version
except Exception:
    pass

try:
    from external.clip import tokenize
except ImportError:
    print("Error: clip (external.clip.tokenize) not found. Please ensure CLIP is installed and accessible.")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: Please install huggingface_hub (pip install huggingface_hub)")
    sys.exit(1)

# 必需的元数据文件
REQUIRED_META = ["metadata.json", "caption.json"]
# 设定面数上限，避免显存溢出 (TexGaussian 建议)
MAX_FACE_COUNT = 200000 

def strict_validate(text: str):
    """严格校验文本长度，不允许截断；超长则抛错。"""
    try:
        tokenize(text)
    except RuntimeError as e:
        raise ValueError(f"Text too long for CLIP context: {text}") from e

def parse_captions_natural(text: str) -> tuple[str, str]:
    if text is None:
        return "", ""
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "", ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s for s in sentences if s]
    if not sentences:
        return "", ""
    if len(sentences) >= 2:
        short_caption = f"{sentences[0]} {sentences[1]}"
    else:
        short_caption = sentences[0]
    return short_caption, cleaned

def log_skip(log_f, obj_id, reason, detail=""):
    message = f"SKIP\t{obj_id}\t{reason}"
    if detail:
        message += f"\t{detail}"
    log_f.write(message + "\n")
    log_f.flush()

def validate_glb_strict(glb_path):
    """
    严格校验 GLB 结构 (Single Mesh, Single Material)
    二进制快速解析，无需加载几何体
    """
    try:
        with open(glb_path, "rb") as f:
            # 1. 校验 Magic (glTF)
            magic, _, _ = struct.unpack('<3I', f.read(12))
            if magic != 0x46546C67: return False 
            
            # 2. 读取 Chunk 0 (JSON)
            chunk_len, chunk_type = struct.unpack('<2I', f.read(8))
            if chunk_type != 0x4E4F534A: return False # 'JSON'
            
            json_bytes = f.read(chunk_len)
            data = json.loads(json_bytes.decode('utf-8'))
            
            # 3. 核心筛选逻辑
            meshes = data.get("meshes", [])
            materials = data.get("materials", [])
            
            # 必须只有一个 Mesh
            if len(meshes) != 1: return False
            
            # 必须只有一个 Primitive (即单一材质槽)
            primitives = meshes[0].get("primitives", [])
            if len(primitives) != 1: return False
                
            # 必须只有一个材质定义
            if len(materials) != 1: return False
                
            return True
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="YiboZhang2001/TexVerse-1K")
    parser.add_argument("--data-root", default="../datasets/texverse")
    parser.add_argument("--out-dir", default="", help="默认同 data-root")
    parser.add_argument("--total-num", type=int, default=2000, help="目标下载数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if not args.out_dir: args.out_dir = args.data_root
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 确保元数据存在
    print(f"Checking metadata in {args.data_root}...")
    for fn in REQUIRED_META:
        if not os.path.exists(os.path.join(args.data_root, fn)):
            print(f"[Downloading] {fn}...")
            try:
                hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=fn, local_dir=args.data_root)
            except Exception as e:
                sys.exit(f"Fatal: {e}")

    # 2. 加载元数据
    print("Loading Metadata...")
    with open(os.path.join(args.data_root, "metadata.json"), "r", encoding="utf-8") as f: 
        meta = json.load(f)
    with open(os.path.join(args.data_root, "caption.json"), "r", encoding="utf-8") as f: 
        caps = json.load(f)
    
    log_path = os.path.join(args.out_dir, "download_texverse_raw.log")
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("event\tobj_id\treason\tdetail\n")

        # 3. 筛选候选列表
        candidates = []
        print("Filtering candidates (Target: 1k, Metalness PBR)...")
        
        for obj_id, m in meta.items():
            # A. PBR 类型筛选: TexGaussian 仅支持 Metalness 流程 [cite: 58, 190]
            # JSON中 key 为 "pbrType"
            pbr_type = m.get("pbrType")
            if pbr_type != "metalness": 
                continue
                
            # B. 寻找 1k 路径
            paths = m.get("glb_paths", [])
            target_path = None
            for p in paths:
                if "_1024.glb" in p: # 匹配 1k 文件名
                    target_path = p
                    break
            
            if not target_path: 
                continue
            
            # C. 获取 Caption
            c_data = caps.get(obj_id)
            if isinstance(c_data, dict): 
                cap = c_data.get("material_desc") or c_data.get("caption")
            else: 
                cap = c_data
            if cap is None or cap == "":
                log_skip(log_f, obj_id, "empty caption")
                continue
            
            # D. 面数筛选
            face_count = m.get("faceCount", 0) or 0
            if int(face_count) > MAX_FACE_COUNT: 
                continue

            candidates.append({
                "obj_id": obj_id, 
                "rel_glb": target_path, 
                "caption": cap
            })

        random.seed(args.seed)
        random.shuffle(candidates)
        
        # 4. 下载与校验
        print(f"Start verifying... Candidates: {len(candidates)}, Target: {args.total_num}")
        valid_list = []
        
        for item in candidates:
            if len(valid_list) >= args.total_num: 
                break

            short_caption, long_caption = parse_captions_natural(item["caption"])
            if not short_caption or not long_caption:
                log_skip(log_f, item["obj_id"], "empty caption")
                continue
            try:
                strict_validate(short_caption)
            except ValueError as e:
                log_skip(log_f, item["obj_id"], "caption_short too long (tokenized)", str(e))
                continue
            except Exception as e:
                log_skip(log_f, item["obj_id"], "caption_short validation error", str(e))
                continue
            
            try:
                # 下载具体文件
                local_path = hf_hub_download(
                    repo_id=args.repo, repo_type="dataset", 
                    filename=item["rel_glb"], local_dir=args.data_root
                )
                
                # 严格校验
                if validate_glb_strict(local_path):
                    item_clean = {
                        "obj_id": item["obj_id"],
                        "rel_glb": item["rel_glb"],
                        "caption_short": short_caption,
                        "caption_long": long_caption,
                    }
                    valid_list.append(item_clean)
                    if len(valid_list) % 10 == 0:
                        print(f"Qualified: {len(valid_list)}/{args.total_num}")
                else:
                    # 不合格则删除以节省空间
                    try:
                        os.remove(local_path)
                    except FileNotFoundError:
                        pass
                    log_skip(log_f, item["obj_id"], "glb invalid", item["rel_glb"])
                    
            except Exception as e:
                log_skip(log_f, item["obj_id"], "download fail", str(e))
                print(f"[Fail] {item['obj_id']}: {e}")

        # 5. 生成清单
        out_tsv = os.path.join(args.out_dir, "downloaded_manifest.tsv")
        with open(out_tsv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["rel_glb", "caption_short", "caption_long", "obj_id"])
            for item in valid_list:
                writer.writerow([item["rel_glb"], item["caption_short"], item["caption_long"], item["obj_id"]])
                
    print(f"[Step 1 Done] Collected {len(valid_list)} valid Metalness models.")
    print(f"Manifest saved to {out_tsv}")

if __name__ == "__main__":
    main()
