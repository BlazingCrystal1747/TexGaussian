#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[Step 3] split_dataset.py
功能：生成数据集的官方划分 (Common Split)
设计理念：
   GT (Ground Truth) 只有一份，实验有无数个。
   为了公平比较，所有实验应使用同一套 Train/Val/Test 划分。
   本脚本默认将划分文件生成到 experiments/common_splits 文件夹。
"""

import os
import csv
import random
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    
    # ================= 配置区域 =================
    
    # 输入：Step 2 生成的总表
    parser.add_argument("--manifest", 
                        default="../datasets/texverse_extracted/manifest_extracted.tsv", 
                        help="Path to the extracted manifest from Step 2")

    # 输出：默认放在 experiments/common_splits，作为所有实验的公共参考
    parser.add_argument("--out-dir", 
                        default="../experiments/common_splits", 
                        help="Directory to save train.tsv, val.tsv, test.tsv")

    # 划分比例
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], 
                        help="Train Val Test ratios (default: 0.8 0.1 0.1)")
    
    # 随机种子：固定为 42 以保证可复现性
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for splitting (Keep it fixed for fair comparison!)")
    
    args = parser.parse_args()

    # ================= 1. 环境检查 =================
    if not os.path.exists(args.manifest):
        print(f"[Error] Manifest not found at: {os.path.abspath(args.manifest)}")
        print("请先运行 Step 2 (提取脚本) 生成数据列表。")
        return

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"========== Generating Common Splits ==========")
    print(f"Input Manifest: {args.manifest}")
    print(f"Output Directory: {os.path.abspath(args.out_dir)}")
    print(f"Random Seed: {args.seed}")

    # ================= 2. 读取并校验数据 =================
    valid_data = []
    # 训练必须的 5 个通道
    required_cols = ['mesh', 'albedo', 'rough', 'metal', 'normal', 'glb_path']

    print(f"Reading and validating data...")
    try:
        with open(args.manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fieldnames = reader.fieldnames
            
            # 检查表头
            for k in required_cols:
                if k not in fieldnames:
                    print(f"[Error] Manifest missing column: '{k}'")
                    return

            # 遍历检查每一行文件的物理存在性
            for row in reader:
                # 这是一个耗时但必要的步骤，防止训练中途崩溃
                if all(os.path.exists(row[k]) for k in required_cols):
                    valid_data.append(row)
                # else:
                #    print(f"Skipping {row['obj_id']}: missing files")

    except Exception as e:
        print(f"[Error] Failed to read manifest: {e}")
        return
    
    total = len(valid_data)
    print(f"Total valid samples: {total}")

    if total == 0:
        print("[Error] No valid data found! Check your paths.")
        return

    # ================= 3. 随机切分 =================
    random.seed(args.seed)
    random.shuffle(valid_data)

    total_r = sum(args.ratios)
    n_train = int(total * (args.ratios[0] / total_r))
    n_val = int(total * (args.ratios[1] / total_r))
    # Test 拿剩余所有，避免舍入误差丢数据
    
    splits = {
        "train": valid_data[:n_train],
        "val":   valid_data[n_train : n_train + n_val],
        "test":  valid_data[n_train + n_val :]
    }

    # ================= 4. 保存结果 =================
    for name, data in splits.items():
        fname = os.path.join(args.out_dir, f"{name}.tsv")
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            # 保持和输入一样的表头
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(data)
        print(f"  -> Saved {name:<5}: {len(data):>6} samples to {fname}")

    # 保存一份元数据信息，方便回溯
    info_path = os.path.join(args.out_dir, "split_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Created Date: {os.popen('date').read().strip()}\n")
        f.write(f"Source Manifest: {os.path.abspath(args.manifest)}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Ratios: {args.ratios}\n")
        f.write(f"Counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}\n")

    print(f"\n[Success] Common splits generated at: {args.out_dir}")
    print(f"下一步：在你的训练脚本中，直接读取 {os.path.join(args.out_dir, 'train.tsv')} 即可。")

if __name__ == "__main__":
    main()