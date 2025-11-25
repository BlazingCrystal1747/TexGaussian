import bpy
import os
import sys
import argparse
import csv
import math
import time
from mathutils import Vector

# ================= 脚本元数据 =================
# 脚本名称: render_pbr_eval.py (Fixed Version)
# 修复内容: 增加了 GLB 层级打平与网格合并逻辑，修复 GLB 渲染不可见或位置偏移的问题
# ============================================

# ================= 配置区域 =================
COL_MAP = {
    "id": "obj_id",
    "mesh": "mesh",         
    "glb": "glb_path",      
    "albedo": "albedo",
    "rough": "rough",
    "metal": "metal"
}

def log(msg):
    print(f"[Step 4 Render] {msg}")

# ================= 1. 场景初始化 =================
def init_scene(res, samples=64):
    """初始化 Blender 场景：清空、设置 Cycles 引擎、配置 GPU"""
    bpy.ops.wm.read_homefile(use_empty=True)
    
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.film_transparent = True 
    
    sc.cycles.samples = samples
    sc.cycles.film_exposure = 1.0 
    
    sc.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'CUDA' 
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
    except Exception:
        sc.cycles.device = 'CPU'
        
    try: sc.view_layers[0].cycles.use_denoising = True
    except: pass
    
    return sc

# ================= 2. 材质设置 (仅 PBR 模式用) =================
def set_pbr_material(mesh_obj, albedo_path, rough_path, metal_path):
    """构建标准 PBR 材质节点"""
    if mesh_obj.data.materials:
        mesh_obj.data.materials.clear()
    
    mat = bpy.data.materials.new(name="PBR_Eval")
    mat.use_nodes = True
    mesh_obj.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    def load_tex(path, colorspace, pos_y, input_socket):
        if path and os.path.exists(path):
            node = nodes.new('ShaderNodeTexImage')
            node.location = (-300, pos_y)
            try:
                img = bpy.data.images.load(path)
                img.colorspace_settings.name = colorspace
                node.image = img
                links.new(node.outputs['Color'], bsdf.inputs[input_socket])
                return True
            except: pass
        return False

    load_tex(albedo_path, 'sRGB', 300, 'Base Color')
    load_tex(rough_path, 'Non-Color', 0, 'Roughness')
    load_tex(metal_path, 'Non-Color', -300, 'Metallic')

# ================= 3. HDRI 环境光 =================
def set_hdri(hdri_path):
    if not hdri_path or not os.path.exists(hdri_path):
        log("Warning: HDRI path invalid.")
        return

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(hdri_path)
    bg = nodes.new('ShaderNodeBackground')
    out = nodes.new('ShaderNodeOutputWorld')
    
    links.new(env.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], out.inputs['Surface'])
    bg.inputs['Strength'].default_value = 1.0

# ================= 4. 模型导入与几何处理 (修复版) =================
def import_and_clean(mesh_path, keep_materials=False):
    """
    导入模型并进行几何标准化。
    修复：处理 GLB 的层级结构 (Parenting) 和多 Mesh 合并。
    """
    ext = os.path.splitext(mesh_path)[1].lower()
    
    # 1. 导入文件
    if ext == '.obj':
        try: bpy.ops.import_scene.obj(filepath=mesh_path, use_image_search=False, use_split_objects=False) 
        except: bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == '.glb' or ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else: 
        return None
        
    # 2. 收集所有导入的 MESH 对象
    imported_meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    
    if not imported_meshes:
        # 有时候 gltf 导入后选中的是 Empty 节点，需要遍历全场景找最近添加的 Mesh
        # 这里简单处理：如果没选中 mesh，直接返回 None (通常 import_scene 默认会选中)
        return None

    # 3. [关键修复] 处理层级和多部件
    # 3.1 选中所有 Mesh 并解除父子关系 (保持变换)，确保它们在世界坐标系下
    bpy.ops.object.select_all(action='DESELECT')
    for m in imported_meshes:
        m.select_set(True)
    
    # 解除父级，保留变换 (防止 Mesh 飞走)
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # 3.2 合并所有 Mesh 为一个对象 (处理多部件 GLB)
    bpy.context.view_layer.objects.active = imported_meshes[0]
    if len(imported_meshes) > 1:
        bpy.ops.object.join()
    
    mesh_obj = bpy.context.view_layer.objects.active

    # 4. [分支逻辑] 材质处理
    if not keep_materials:
        if mesh_obj.data.materials:
            mesh_obj.data.materials.clear()
    
    # 5. [通用逻辑] 几何归一化
    # 确保只选中最终的 mesh_obj
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    
    bpy.ops.object.shade_smooth() 
    
    # 居中原点 (现在它是世界坐标系下的独立物体，移动安全)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    mesh_obj.location = (0, 0, 0)
    
    # 缩放到单位球
    bpy.context.view_layer.update() # 强制更新包围盒
    bbox_corners = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    max_dist = max([v.length for v in bbox_corners])
    
    if max_dist > 0:
        scale_factor = 0.9 / max_dist
        mesh_obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)
        
    return mesh_obj

# ================= 5. 主工作流 =================
def render_worker(args):
    # 抑制输出工具类
    class SuppressOutput:
        def __enter__(self):
            self.save_stdout = os.dup(sys.stdout.fileno())
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, sys.stdout.fileno())
        def __exit__(self, exc_type, exc_value, traceback):
            os.dup2(self.save_stdout, sys.stdout.fileno())
            os.close(self.devnull)

    # 1. 读取 Manifest
    if not os.path.exists(args.manifest):
        log(f"Error: Manifest not found at {args.manifest}")
        return

    tasks = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        tasks = list(reader)
        
    log(f"Render Mode: {args.mode.upper()}")
    log(f"Total Tasks: {len(tasks)}")
    
    start_time = time.time()
    success_count = 0

    for i, row in enumerate(tasks):
        oid = row.get(COL_MAP['id'])
        
        if args.mode == 'pbr':
            input_mesh_path = row.get(COL_MAP['mesh'])
        else: 
            input_mesh_path = row.get(COL_MAP['glb'])
            
        if not input_mesh_path or not os.path.exists(input_mesh_path):
            continue
        
        out_dir = os.path.join(args.out_dir, oid)
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= 20:
            continue
        os.makedirs(out_dir, exist_ok=True)
        
        log(f"[{i+1}/{len(tasks)}] Rendering ({args.mode}): {oid}")

        # --- Blender 渲染流程 ---
        init_scene(args.res)
        set_hdri(args.hdri)
        
        keep_mat = (args.mode == 'glb')
        obj = None
        
        # 尝试导入 (捕获可能的几何错误)
        try:
            # 临时关闭 stdout 避免 Blender 刷屏
            with SuppressOutput():
                obj = import_and_clean(input_mesh_path, keep_materials=keep_mat)
        except Exception as e:
            # log(f"Import Error: {e}")
            pass
            
        if not obj: 
            log(f"Failed to load object: {oid}")
            continue
        
        # PBR 模式挂载材质
        if args.mode == 'pbr':
            alb_p = row.get(COL_MAP['albedo'])
            rgh_p = row.get(COL_MAP['rough'])
            met_p = row.get(COL_MAP['metal'])
            set_pbr_material(obj, alb_p, rgh_p, met_p)
        
        # 相机设置
        cam_data = bpy.data.cameras.new("Cam")
        cam = bpy.data.objects.new("Cam", cam_data)
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        
        # 渲染循环
        elevations = [20, 45] 
        azimuths = range(0, 360, 36)
        radius = 2.8
        
        cnt = 0
        for elev in elevations:
            for az in azimuths:
                theta = math.radians(elev)
                phi = math.radians(az)
                x = radius * math.cos(theta) * math.cos(phi)
                y = radius * math.cos(theta) * math.sin(phi)
                z = radius * math.sin(theta)
                cam.location = (x, y, z)
                
                direction = -cam.location
                rot_quat = direction.to_track_quat('-Z', 'Y')
                cam.rotation_euler = rot_quat.to_euler()
                
                bpy.context.scene.render.filepath = os.path.join(out_dir, f"{cnt:03d}.png")
                
                with SuppressOutput():
                    bpy.ops.render.render(write_still=True)
                cnt += 1
        
        success_count += 1
        
        # 内存清理
        for block in bpy.data.meshes: bpy.data.meshes.remove(block)
        for block in bpy.data.materials: bpy.data.materials.remove(block)
        for block in bpy.data.objects: bpy.data.objects.remove(block)
        for block in bpy.data.cameras: bpy.data.cameras.remove(block)
        for block in bpy.data.images: 
            if block.name != "Render Result": bpy.data.images.remove(block)

    total_time = time.time() - start_time
    log("==========================================")
    log(f"Done. Rendered {success_count} objects in {total_time:.2f}s.")
    log("==========================================")

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to TSV file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--hdri", required=True, help="Path to HDRI file")
    parser.add_argument("--res", type=int, default=512, help="Image resolution")
    parser.add_argument("--mode", default="pbr", choices=["pbr", "glb"], 
                        help="Render mode: 'pbr' or 'glb'")
    
    args = parser.parse_args(argv)
    render_worker(args)