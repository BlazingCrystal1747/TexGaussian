#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[Step 3] split_dataset.py
功能：基于 Step 2 产出的 manifest_extracted.tsv 先完成数据划分，供后续渲染脚本直接读取。
逻辑：
   - 只检查 Step 2 产出的 mesh/texture 路径是否存在，不检查渲染结果
   - 输出 train.tsv / val.tsv / test.tsv，列保留用于渲染的路径信息
"""

import os
import csv
import random
import argparse
from typing import List


def resolve_path(path: str, base_dir: str) -> str:
    """Allow relative paths in the manifest by resolving against its directory."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))

def main():
    parser = argparse.ArgumentParser()
    
    # ================= 配置区域 =================
    
    # 输入：Step 2 生成的总表
    parser.add_argument("--manifest", 
                        default="../datasets/texverse_extracted/manifest_extracted.tsv", 
                        help="Path to the extracted manifest from Step 2 (expects caption_short/long).")

    # 输出：默认放在 experiments/common_splits，作为所有实验的公共参考
    parser.add_argument("--out-dir", 
                        default="../experiments/common_splits", 
                        help="Directory to save train.tsv, val.tsv, test.tsv")

    # 划分比例（未指定 --counts 时生效）
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Train Val Test ratios (used when --counts is not set)")
    
    # 随机种子：固定为 42 以保证可复现性
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for splitting (Keep it fixed for fair comparison!)")

    # Optional: limit how many samples to take from the manifest before splitting
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of samples from manifest before splitting (0 = all)")

    # 指定 train/val/test 的数量（优先级高于 --ratios）
    parser.add_argument("--counts", nargs=3, type=int, default=None,
                        help="Train Val Test counts (overrides --ratios)")
    
    args = parser.parse_args()

    # ================= 1. 环境检查 =================
    manifest_path = os.path.abspath(args.manifest)
    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found at: {manifest_path}")
        print("请先运行 Step 2 (提取脚本) 生成数据列表。")
        return
    
    manifest_dir = os.path.dirname(manifest_path)

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"========== Generating Common Splits ==========")
    print(f"Input Manifest: {manifest_path}")
    print(f"Output Directory: {os.path.abspath(args.out_dir)}")
    print(f"Random Seed: {args.seed}")

    # ================= 2. 读取并校验数据 =================
    valid_data = []
    fieldnames_in = []

    print(f"Reading and validating data...")
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fieldnames_in = reader.fieldnames or []
            
            if "obj_id" not in fieldnames_in:
                print(f"[Error] Manifest missing column: 'obj_id'")
                return

            mesh_col = "mesh_path" if "mesh_path" in fieldnames_in else ("mesh" if "mesh" in fieldnames_in else None)
            albedo_col = "albedo" if "albedo" in fieldnames_in else None
            rough_col = "rough" if "rough" in fieldnames_in else ("roughness" if "roughness" in fieldnames_in else None)
            metal_col = "metal" if "metal" in fieldnames_in else ("metallic" if "metallic" in fieldnames_in else None)
            normal_col = "normal" if "normal" in fieldnames_in else None

            if mesh_col is None or albedo_col is None or rough_col is None or metal_col is None or normal_col is None:
                print("[Error] Manifest missing required texture columns (need mesh/albedo/rough/metal/normal).")
                return
            required_captions = ("caption_short", "caption_long")
            missing = [name for name in required_captions if name not in fieldnames_in]
            if missing:
                print(f"[Error] Manifest missing required caption columns: {', '.join(missing)}")
                print(f"Available columns: {', '.join(fieldnames_in) if fieldnames_in else '(none)'}")
                return
            print("Using caption columns: caption_short, caption_long")

            for row in reader:
                oid = row.get("obj_id")
                if not oid:
                    continue

                mesh_path = resolve_path(row.get(mesh_col), manifest_dir)
                albedo_path = resolve_path(row.get(albedo_col), manifest_dir)
                rough_path = resolve_path(row.get(rough_col), manifest_dir)
                metal_path = resolve_path(row.get(metal_col), manifest_dir)
                normal_path = resolve_path(row.get(normal_col), manifest_dir)

                required_paths = [mesh_path, albedo_path, rough_path, metal_path, normal_path]
                if not all(required_paths):
                    continue
                if not all(os.path.exists(p) for p in required_paths):
                    continue

                entry = {
                    "obj_id": oid,
                    "mesh": mesh_path,
                    "albedo": albedo_path,
                    "rough": rough_path,
                    "metal": metal_path,
                    "normal": normal_path,
                }
                entry["caption_short"] = row.get("caption_short", "")
                entry["caption_long"] = row.get("caption_long", "")
                valid_data.append(entry)

    except Exception as e:
        print(f"[Error] Failed to read manifest: {e}")
        return

    out_fieldnames = [
        "obj_id",
        "mesh",
        "albedo",
        "rough",
        "metal",
        "normal",
        "caption_short",
        "caption_long",
    ]

    total = len(valid_data)
    print(f"Total valid samples: {total}")

    if total == 0:
        print("[Error] No valid data found! Check your paths.")
        return

    if args.max_samples < 0:
        print("[Error] --max-samples must be >= 0")
        return

    if args.counts is not None:
        if any(c < 0 for c in args.counts):
            print("[Error] --counts must be >= 0")
            return
        if sum(args.counts) == 0:
            print("[Error] --counts sum must be > 0")
            return

    # ================= 3. 随机切分 =================
    random.seed(args.seed)
    random.shuffle(valid_data)

    if args.max_samples and args.max_samples < total:
        valid_data = valid_data[:args.max_samples]
        total = len(valid_data)
        print(f"Limiting split to {total} samples via --max-samples.")

    if args.counts is not None:
        total_requested = sum(args.counts)
        if total_requested > total:
            print(f"[Error] --counts sum ({total_requested}) exceeds available samples ({total})")
            return
        if total_requested < total:
            valid_data = valid_data[:total_requested]
            total = len(valid_data)
            print(f"Limiting split to {total} samples via --counts.")
        n_train, n_val, n_test = args.counts
    else:
        total_r = sum(args.ratios)
        n_train = int(total * (args.ratios[0] / total_r))
        n_val = int(total * (args.ratios[1] / total_r))
        # Test 拿剩余所有，避免舍入误差丢数据
        n_test = total - n_train - n_val
    
    splits = {
        "train": valid_data[:n_train],
        "val":   valid_data[n_train : n_train + n_val],
        "test":  valid_data[n_train + n_val : n_train + n_val + n_test]
    }

    # ================= 4. 保存结果 =================
    for name, data in splits.items():
        fname = os.path.join(args.out_dir, f"{name}.tsv")
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(data)
        print(f"  -> Saved {name:<5}: {len(data):>6} samples to {fname}")

    # 保存一份元数据信息，方便回溯
    info_path = os.path.join(args.out_dir, "split_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Created Date: {os.popen('date').read().strip()}\n")
        f.write(f"Source Manifest: {manifest_path}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Ratios: {args.ratios}\n")
        f.write(f"Max Samples: {args.max_samples}\n")
        f.write(f"Counts Requested: {args.counts}\n")
        f.write(f"Counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}\n")

    print(f"\n[Success] Common splits generated at: {args.out_dir}")
    print(f"下一步：在你的训练脚本中，直接读取 {os.path.join(args.out_dir, 'train.tsv')} 即可。")

if __name__ == "__main__":
    main()
