#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_glb_assets.py
功能：批量处理 GLB 文件，提取 Mesh (.obj) 和 PBR 贴图 (.png)
工具：Trimesh + Numpy + Pillow (无需 Blender)
输入：数据列表 manifest TSV
输出：
    1. /raw_assets/: [隔离区] 
       - *.mtl (保留原名，防止 Blender 自动加载)
       - *.png (Trimesh生成的原始贴图，保留原名)
       - raw_normal.png / raw_orm.png (手动补全的Raw数据)
    2. mesh.obj: 纯净网格 (位于根目录)
    3. albedo.png, normal.png ...: 烘焙好的 GT 贴图 (位于根目录)
"""

import trimesh
import os
import sys
import argparse
import csv
import numpy as np
import shutil
import traceback
import time

try:
    from PIL import Image
except ImportError:
    print("错误: 找不到 'PIL' 模块。请运行: pip install Pillow")
    sys.exit(1)

# ================= 配置 =================
STRICT_MODE = True 
# =======================================

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log(msg):
    print(f"[Extract] {msg}")

def error_log(oid, reason, log_file="extract_errors.txt"):
    print(f"    !!! ERROR {oid}: {reason}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{oid}\t{reason}\n")

class MissingTextureError(Exception):
    pass

def to_clean_list(value, default_val, target_len=None):
    if value is None:
        result = default_val
    elif isinstance(value, (np.ndarray, np.generic)):
        try:
            result = np.atleast_1d(value).flatten().tolist()
        except:
            result = default_val
    elif isinstance(value, (int, float)):
        result = [value]
    else:
        try:
            result = list(value)
        except:
            result = default_val

    if target_len is not None:
        if len(result) == 1 and target_len > 1:
            result = result * target_len
        if len(result) > target_len:
            result = result[:target_len]
        elif len(result) < target_len:
            result.extend(default_val[len(result):])
            
    return [float(x) for x in result]

def bake_and_save(image_obj, factor_list, output_path, mode='RGB', require_texture=True):
    if require_texture and image_obj is None:
        raise MissingTextureError(f"Missing texture for {os.path.basename(output_path)}")

    factor_arr = np.array(factor_list, dtype=np.float32)
    img_arr = None
    width, height = 1024, 1024 
    
    if image_obj is not None:
        width, height = image_obj.size
        try:
            if mode == 'RGB':
                img = image_obj.convert('RGB')
                img_arr = np.array(img).astype(np.float32) / 255.0
            else: # L (Grayscale)
                img = image_obj.convert('L')
                img_arr = np.array(img).astype(np.float32) / 255.0
        except Exception as e:
            raise MissingTextureError(f"Texture corrupted: {e}")
    
    if img_arr is None:
        if mode == 'RGB':
            img_arr = np.ones((height, width, 3), dtype=np.float32) * factor_arr
        else:
            img_arr = np.ones((height, width), dtype=np.float32) * factor_arr
    else:
        img_arr = img_arr * factor_arr
        
    img_arr = np.clip(img_arr, 0.0, 1.0) * 255.0
    final_img = Image.fromarray(img_arr.astype(np.uint8))
    final_img.save(output_path)

def process_single_model(glb_path, out_dir):
    """ 处理单个 GLB 文件的核心逻辑 """
    safe_mkdir(out_dir)
    
    # 1. 创建隔离区
    raw_dir = os.path.join(out_dir, "raw_assets")
    safe_mkdir(raw_dir)
    
    try:
        scene = trimesh.load(glb_path, force='scene')
    except Exception as e:
        raise Exception(f"Trimesh load failed: {e}")

    target_geom = None
    target_mat = None
    
    for geom in scene.geometry.values():
        if hasattr(geom.visual, 'material'):
            target_geom = geom
            target_mat = geom.visual.material
            break
            
    if target_geom is None:
        raise MissingTextureError("No geometry with material found")

    # ====================================================
    # 步骤 A: 导出 Mesh
    # ====================================================
    # Trimesh 导出时会自动生成关联的 .mtl 和图片文件到 out_dir
    mesh_out_path = os.path.join(out_dir, "mesh.obj")
    target_geom.export(mesh_out_path)
    
    # ====================================================
    # 步骤 B: 智能隔离 (Moving Raw Assets) - [修复版]
    # ====================================================
    
    # 1. 扫描目录下所有的 .mtl 和 .png/.jpg，将它们移入 raw_assets
    #    注意：此时我们还没有开始生成 albedo.png 等最终贴图，
    #    所以文件夹里现有的图片全是 Trimesh 生成的原始素材。
    
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        
        # 跳过目录本身和刚刚生成的 mesh.obj
        if os.path.isdir(file_path): continue
        if filename == "mesh.obj": continue
        
        # 移动 .mtl 文件 (无论叫什么名字，只要是 mtl 就移走)
        if filename.lower().endswith('.mtl'):
            shutil.move(file_path, os.path.join(raw_dir, filename))
            
        # 移动 图片文件 (保留原名)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.move(file_path, os.path.join(raw_dir, filename))

    # ====================================================
    # 步骤 C: 手动补全丢失的 Raw PBR (Normal, ORM)
    # ====================================================
    
    # 1. 补全原始法线 (Raw Normal)
    tex_normal_raw = getattr(target_mat, 'normalTexture', None)
    if tex_normal_raw:
        try:
            tex_normal_raw.save(os.path.join(raw_dir, "raw_normal.png"))
        except: pass

    # 2. 补全原始 ORM (Raw ORM)
    tex_mr_raw = getattr(target_mat, 'metallicRoughnessTexture', None)
    if tex_mr_raw:
        try:
            tex_mr_raw.save(os.path.join(raw_dir, "raw_orm.png"))
        except: pass

    # ====================================================
    # 步骤 D: 生成 GT 数据集 (Baking Process)
    # (生成在根目录的标准文件，用于训练和评测)
    # ====================================================

    # 准备 Factor
    raw_base = getattr(target_mat, 'baseColorFactor', None)
    base_list = to_clean_list(raw_base, default_val=[1.0, 1.0, 1.0, 1.0], target_len=4)
    if max(base_list) > 1.0: base_list = [x/255.0 for x in base_list]
    base_rgb = base_list[:3]

    raw_metal = getattr(target_mat, 'metallicFactor', None)
    metal_list = to_clean_list(raw_metal, default_val=[1.0], target_len=1)

    raw_rough = getattr(target_mat, 'roughnessFactor', None)
    rough_list = to_clean_list(raw_rough, default_val=[1.0], target_len=1)
    
    # 获取贴图对象
    tex_albedo = getattr(target_mat, 'baseColorTexture', None)
    if hasattr(target_mat, 'image') and tex_albedo is None: tex_albedo = target_mat.image
    
    tex_normal = getattr(target_mat, 'normalTexture', None)
    tex_mr = getattr(target_mat, 'metallicRoughnessTexture', None)
    
    tex_metal = None
    tex_rough = None
    
    if tex_mr:
        mr_arr = np.array(tex_mr.convert('RGB'))
        tex_rough = Image.fromarray(mr_arr[..., 1]) # G
        tex_metal = Image.fromarray(mr_arr[..., 2]) # B
    
    # 执行烘焙
    bake_and_save(tex_albedo, base_rgb, os.path.join(out_dir, "albedo.png"), 
                 mode='RGB', require_texture=STRICT_MODE)
    
    if STRICT_MODE and tex_normal is None:
        raise MissingTextureError("Missing Normal Map")
    elif tex_normal:
        tex_normal.save(os.path.join(out_dir, "normal.png"))
    else:
        pass 

    bake_and_save(tex_metal, metal_list, os.path.join(out_dir, "metallic.png"), 
                 mode='L', require_texture=STRICT_MODE)
                 
    bake_and_save(tex_rough, rough_list, os.path.join(out_dir, "roughness.png"), 
                 mode='L', require_texture=STRICT_MODE)

    return True

def main():
    parser = argparse.ArgumentParser(description="GLB Asset Extraction Tool")
    parser.add_argument("--tsv", required=True, help="Input manifest tsv")
    parser.add_argument("--data-root", required=True, help="Root directory containing GLB files")
    parser.add_argument("--out-root", required=True, help="Output directory for extracted assets")
    args = parser.parse_args()

    print(f"========== GLB Extraction Tool ==========")
    print(f"Config Strict Mode: {STRICT_MODE}")
    print(f"Input Manifest:     {args.tsv}")
    print(f"Data Root:          {args.data_root}")
    print(f"Output Root:        {args.out_root}")
    
    safe_mkdir(args.out_root)
    manifest_out_path = os.path.join(args.out_root, "manifest_extracted.tsv")
    error_log_file = os.path.join(args.out_root, "extract_errors.txt")
    
    if os.path.exists(error_log_file): os.remove(error_log_file)

    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    tasks = []
    try:
        with open(args.tsv, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in, delimiter='\t')
            for row in reader: tasks.append(row)
    except Exception as e:
        log(f"Error reading TSV: {e}")
        return

    print(f"Total tasks loaded: {len(tasks)}")
    print(f"Starting extraction...\n")

    start_time = time.time()

    with open(manifest_out_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["obj_id", "caption", "mesh", "albedo", "rough", "metal", "normal", "glb_path"])
        
        for i, row in enumerate(tasks):
            oid = row['obj_id']
            glb_rel = row.get('rel_glb', row.get('path', ''))
            glb_path = os.path.join(args.data_root, glb_rel)
            out_dir = os.path.join(args.out_root, oid)
            
            p_mesh = os.path.join(out_dir, "mesh.obj")
            p_albedo = os.path.join(out_dir, "albedo.png")
            p_rough = os.path.join(out_dir, "roughness.png")
            p_metal = os.path.join(out_dir, "metallic.png")
            p_normal = os.path.join(out_dir, "normal.png")

            if i % 10 == 0:
                print(f"\rProcessing: {i}/{len(tasks)} (Success: {processed_count}, Fail: {failed_count})", end="")

            if os.path.exists(p_mesh) and os.path.exists(p_albedo) and \
               os.path.exists(p_rough) and os.path.exists(p_metal) and os.path.exists(p_normal):
                writer.writerow([oid, row['caption'], 
                                 os.path.abspath(p_mesh), os.path.abspath(p_albedo), 
                                 os.path.abspath(p_rough), os.path.abspath(p_metal), os.path.abspath(p_normal),
                                 os.path.abspath(glb_path)])
                skipped_count += 1
                continue

            if not os.path.exists(glb_path):
                error_log(oid, "GLB file not found", error_log_file)
                failed_count += 1
                continue

            try:
                process_single_model(glb_path, out_dir)
                writer.writerow([
                    oid, row['caption'],
                    os.path.abspath(p_mesh),
                    os.path.abspath(p_albedo),
                    os.path.abspath(p_rough),
                    os.path.abspath(p_metal),
                    os.path.abspath(p_normal),
                    os.path.abspath(glb_path)
                ])
                processed_count += 1
                if processed_count % 10 == 0: f_out.flush()
                
            except MissingTextureError as e:
                failed_count += 1
                error_log(oid, str(e), error_log_file)
                if os.path.exists(out_dir): shutil.rmtree(out_dir)
            except Exception as e:
                failed_count += 1
                error_log(oid, f"Crash: {str(e)}", error_log_file)
                if os.path.exists(out_dir): shutil.rmtree(out_dir)

    print(f"\n\n========== Extraction Summary ==========")
    print(f"Time Taken:     {time.time() - start_time:.2f}s")
    print(f"Total:          {len(tasks)}")
    print(f"Success:        {processed_count}")
    print(f"Skipped:        {skipped_count}")
    print(f"Failed:         {failed_count}")
    print(f"Output Manifest:{manifest_out_path}")
    print(f"========================================\n")
    
    print(f"下一步：请运行数据集划分脚本 (split_dataset.py)。")
    print(f"运行命令示例：python split_dataset.py --manifest {manifest_out_path}")

if __name__ == "__main__":
    main()