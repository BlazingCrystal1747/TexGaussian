#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_gt_dataset.py

用途：
    - train 模式：使用 Emission Shader 渲染 GT 数据（albedo/rough/metal/normal 四个通道），无光照，法线当作 RGB 颜色输出。
    - eval 模式：使用 Principled BSDF + HDRI 环境光渲染真实 PBR 预览，用于 FID/KID 评估（仅输出 Beauty）。

目录结构：
    --mode train -> {out_root}/train_unlit/{obj_id}/
    --mode eval  -> {out_root}/eval_lit/{obj_id}/
"""

import argparse
import csv
import math
import os
import random
import sys
from typing import Any, Dict, List

import bpy
from mathutils import Vector

# ===================== 通用配置 =====================
PASS_CONFIG_TRAIN = [
    ("albedo", "sRGB"),
    ("rough", "Non-Color"),
    ("metal", "Non-Color"),
    ("normal", "Non-Color"),  # 法线贴图仅当作颜色
]

DEFAULT_VIEWS_TRAIN = 64
DEFAULT_VIEWS_EVAL = 20
DEFAULT_RESOLUTION = 512
DEFAULT_BLEND_NAME = "scene.blend"


def resolve_path(path: str, base_dir: str) -> str:
    """Resolve a possibly-relative path using manifest directory as base."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


# ===================== 工具函数 =====================
def log(msg: str) -> None:
    print(f"[Render GT] {msg}")


def set_color_management(scene: bpy.types.Scene) -> None:
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.sequencer_colorspace_settings.name = "sRGB"


def reset_scene(resolution: int, samples: int) -> bpy.types.Scene:
    bpy.ops.wm.read_homefile(use_empty=True)
    sc = bpy.context.scene

    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.device = "GPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception:
        sc.cycles.device = "CPU"

    sc.render.resolution_x = resolution
    sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"

    set_color_management(sc)

    # 清空灯光 / 黑色背景
    for obj in list(sc.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    return sc


def set_hdri(world: bpy.types.World, hdri_path: str, strength: float = 1.0) -> None:
    if not os.path.exists(hdri_path):
        raise FileNotFoundError(f"HDRI not found: {hdri_path}")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def find_normal_path(explicit_normal: str, fallback_from: str) -> str:
    """
    查找可用的法线贴图路径：
        1) 如果显式提供 normal 且文件存在，直接返回。
        2) 否则在 fallback_from 所在目录搜索 normal.png / *_normal.* / *-normal.*。
    """
    if explicit_normal and os.path.exists(explicit_normal):
        return explicit_normal

    candidates = []
    if fallback_from:
        base_dir = os.path.dirname(os.path.abspath(fallback_from))
        for name in os.listdir(base_dir):
            low = name.lower()
            if not low.endswith((".png", ".jpg", ".jpeg", ".exr")):
                continue
            if low == "normal.png" or "_normal" in low or "-normal" in low:
                candidates.append(os.path.join(base_dir, name))
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def import_and_normalize(mesh_path: str) -> bpy.types.Object:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        try:
            bpy.ops.import_scene.obj(filepath=mesh_path, use_image_search=False, use_split_objects=False)
        except Exception:
            bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        return None

    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not imported:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1:
        bpy.ops.object.join()

    mesh_obj = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    mesh_obj.location = (0, 0, 0)
    bpy.context.view_layer.update()

    max_dim = max(mesh_obj.dimensions)
    if max_dim > 0:
        scale = 1.0 / max_dim
        mesh_obj.scale = (scale, scale, scale)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.shade_smooth()
    return mesh_obj


def build_emission_material(name: str, image_path: str, colorspace: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    tex.image.colorspace_settings.name = colorspace
    tex.interpolation = "Smart"

    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Strength"].default_value = 1.0

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex.outputs["Color"], emis.inputs["Color"])
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_pbr_material(albedo: str, rough: str, metal: str, normal: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name="PBR_MATERIAL")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def add_tex(path: str, colorspace: str, loc_y: int):
        node = nodes.new("ShaderNodeTexImage")
        node.location = (-400, loc_y)
        node.image = bpy.data.images.load(path)
        node.image.colorspace_settings.name = colorspace
        node.interpolation = "Smart"
        return node

    if albedo:
        alb = add_tex(albedo, "sRGB", 200)
        links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
    if rough:
        rgh = add_tex(rough, "Non-Color", 0)
        links.new(rgh.outputs["Color"], bsdf.inputs["Roughness"])
    if metal:
        mtl = add_tex(metal, "Non-Color", -200)
        links.new(mtl.outputs["Color"], bsdf.inputs["Metallic"])
    if normal:
        nrm_tex = add_tex(normal, "Non-Color", -400)
        nrm_node = nodes.new("ShaderNodeNormalMap")
        nrm_node.location = (-200, -400)
        links.new(nrm_tex.outputs["Color"], nrm_node.inputs["Color"])
        links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def fibonacci_sphere(samples: int, radius: float, rng: random.Random) -> List[Vector]:
    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))
    rnd = rng.random() * samples
    for i in range(samples):
        y = ((i * offset) - 1.0) + (offset * 0.5)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = (i + rnd) * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append(Vector((x, z, y)) * radius)  # Z up
    return points


def create_camera(scene: bpy.types.Scene, focal_length: float) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def look_at_origin(cam_obj: bpy.types.Object, target: Vector) -> None:
    direction = (target - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def compute_intrinsics(cam_obj: bpy.types.Object, scene: bpy.types.Scene) -> Dict[str, float]:
    cam = cam_obj.data
    render = scene.render
    scale = render.resolution_percentage / 100.0
    res_x = render.resolution_x * scale
    res_y = render.resolution_y * scale
    aspect_ratio = render.pixel_aspect_x / render.pixel_aspect_y

    sensor_fit = cam.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if res_x >= res_y else "VERTICAL"

    if sensor_fit == "VERTICAL":
        s_u = res_x / (cam.sensor_height * aspect_ratio)
        s_v = res_y / cam.sensor_height
    else:
        s_u = res_x / cam.sensor_width
        s_v = res_y / (cam.sensor_width / aspect_ratio)

    fx = cam.lens * s_u
    fy = cam.lens * s_v
    w = int(round(res_x))
    h = int(round(res_y))
    cx = w * 0.5
    cy = h * 0.5
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h}


def clear_data_blocks() -> None:
    # Detach world references to avoid dangling image users
    for sc in list(bpy.data.scenes):
        sc.world = None
    for world in list(bpy.data.worlds):
        if world.use_nodes and world.node_tree:
            world.node_tree.links.clear()
            world.node_tree.nodes.clear()
        bpy.data.worlds.remove(world, do_unlink=True)

    # Remove scene objects first to clear material/mesh users
    for block in list(bpy.data.objects):
        bpy.data.objects.remove(block, do_unlink=True)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)

    # Finally remove images; skip Render Result to avoid Blender internal asserts
    for img in list(bpy.data.images):
        if img.name == "Render Result":
            continue
        img.user_clear()
        try:
            bpy.data.images.remove(img)
        except RuntimeError:
            pass


# ===================== 渲染流程 =====================
def render_train(row: Dict[str, str], args: argparse.Namespace, rng: random.Random) -> bool:
    oid = row.get("obj_id", "unknown")
    mesh_path = row.get("mesh")
    albedo = row.get("albedo")
    rough = row.get("rough") or row.get("roughness")
    metal = row.get("metal") or row.get("metallic")
    normal = row.get("normal")

    required = [mesh_path, albedo, rough, metal, normal]
    if not all(required):
        log(f"{oid}: missing texture paths, skip.")
        return False
    if not all(os.path.exists(p) for p in required):
        log(f"{oid}: file not found, skip.")
        return False

    out_dir = os.path.join(args.out_root, "train_unlit", oid)
    os.makedirs(out_dir, exist_ok=True)
    completion_flag = os.path.join(out_dir, f"{args.views - 1:03d}_normal.png")
    if os.path.exists(completion_flag):
        log(f"{oid}: found existing train renders, skip.")
        return True

    scene = reset_scene(args.resolution, samples=1)
    mesh_obj = import_and_normalize(mesh_path)
    if not mesh_obj:
        log(f"{oid}: import failed.")
        return False

    mats = {
        name: build_emission_material(f"{name.upper()}_EMIT", path, colorspace)
        for name, colorspace, path in [
            ("albedo", "sRGB", albedo),
            ("rough", "Non-Color", rough),
            ("metal", "Non-Color", metal),
            ("normal", "Non-Color", normal),
        ]
    }

    cam = create_camera(scene, args.focal_length)
    points = fibonacci_sphere(args.views, args.radius, rng)
    intrinsics = compute_intrinsics(cam, scene)
    frames: List[Dict[str, Any]] = []
    zero = Vector((0.0, 0.0, 0.0))

    for idx, pos in enumerate(points):
        cam.location = pos
        look_at_origin(cam, zero)
        bpy.context.view_layer.update()

        w2c = cam.matrix_world.inverted()
        frame_prefix = f"{idx:03d}"
        frames.append(
            {
                "frame_id": idx,
                "file_prefix": frame_prefix,
                "images": {
                    "albedo": f"{frame_prefix}_albedo.png",
                    "rough": f"{frame_prefix}_rough.png",
                    "metal": f"{frame_prefix}_metal.png",
                    "normal": f"{frame_prefix}_normal.png",
                },
                "world_to_camera": [[float(v) for v in row_vec] for row_vec in w2c],
            }
        )

        for pass_name, _ in PASS_CONFIG_TRAIN:
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(mats[pass_name])
            scene.render.filepath = os.path.join(out_dir, f"{frame_prefix}_{pass_name}.png")
            bpy.ops.render.render(write_still=True)

    meta = {
        "obj_id": oid,
        "mode": "train",
        "intrinsics": intrinsics,
        "frames": frames,
        "meta": {
            "views": args.views,
            "radius": args.radius,
            "focal_length_mm": args.focal_length,
            "resolution": args.resolution,
        },
    }
    with open(os.path.join(out_dir, "transforms.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    if args.save_blend:
        blend_path = os.path.join(out_dir, DEFAULT_BLEND_NAME)
        try:
            bpy.ops.wm.save_mainfile(filepath=blend_path)
            log(f"{oid}: saved blend to {blend_path}")
        except Exception as e:
            log(f"{oid}: failed to save blend file ({e})")

    log(f"{oid}: train renders saved to {out_dir}")
    return True


def render_eval(row: Dict[str, str], args: argparse.Namespace, rng: random.Random) -> bool:
    oid = row.get("obj_id", "unknown")
    mesh_path = row.get("mesh")
    albedo = row.get("albedo")
    rough = row.get("rough") or row.get("roughness")
    metal = row.get("metal") or row.get("metallic")
    normal = find_normal_path(row.get("normal"), albedo or mesh_path)
    if not normal:
        log(f"{oid}: normal map not found, will render with geometry normals.")

    required = [mesh_path, albedo, rough, metal]
    if not all(required):
        log(f"{oid}: missing required PBR textures, skip.")
        return False
    if not all(os.path.exists(p) for p in required):
        log(f"{oid}: required PBR file not found, skip.")
        return False
    if not args.hdri or not os.path.exists(args.hdri):
        log(f"{oid}: hdri missing for eval.")
        return False

    out_dir = os.path.join(args.out_root, "eval_lit", oid)
    os.makedirs(out_dir, exist_ok=True)
    completion_flag = os.path.join(out_dir, f"{args.views - 1:03d}_beauty.png")
    if os.path.exists(completion_flag):
        log(f"{oid}: found existing eval renders, skip.")
        return True

    scene = reset_scene(args.resolution, samples=args.samples)
    set_hdri(scene.world, args.hdri, strength=args.hdri_strength)
    mesh_obj = import_and_normalize(mesh_path)
    if not mesh_obj:
        log(f"{oid}: import failed.")
        return False

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(build_pbr_material(albedo, rough, metal, normal))

    cam = create_camera(scene, args.focal_length)
    points = fibonacci_sphere(args.views, args.radius, rng)
    intrinsics = compute_intrinsics(cam, scene)
    frames: List[Dict[str, Any]] = []
    zero = Vector((0.0, 0.0, 0.0))

    for idx, pos in enumerate(points):
        cam.location = pos
        look_at_origin(cam, zero)
        bpy.context.view_layer.update()

        w2c = cam.matrix_world.inverted()
        frame_prefix = f"{idx:03d}"
        frames.append(
            {
                "frame_id": idx,
                "file_name": f"{frame_prefix}_beauty.png",
                "world_to_camera": [[float(v) for v in row_vec] for row_vec in w2c],
            }
        )

        scene.render.filepath = os.path.join(out_dir, f"{frame_prefix}_beauty.png")
        bpy.ops.render.render(write_still=True)

    meta = {
        "obj_id": oid,
        "mode": "eval",
        "intrinsics": intrinsics,
        "frames": frames,
        "meta": {
            "views": args.views,
            "radius": args.radius,
            "focal_length_mm": args.focal_length,
            "resolution": args.resolution,
            "hdri": os.path.abspath(args.hdri),
            "normal_map_used": bool(normal),
        },
    }
    with open(os.path.join(out_dir, "transforms.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    if args.save_blend:
        blend_path = os.path.join(out_dir, DEFAULT_BLEND_NAME)
        try:
            bpy.ops.wm.save_mainfile(filepath=blend_path)
            log(f"{oid}: saved blend to {blend_path}")
        except Exception as e:
            log(f"{oid}: failed to save blend file ({e})")

    log(f"{oid}: eval renders saved to {out_dir}")
    return True


# ===================== 主入口 =====================
def pick_field(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT dataset for TexGaussian (train/eval modes).")
    parser.add_argument("--manifest", required=True, help="Path to manifest_extracted.tsv (Step 2).")
    parser.add_argument("--out-root", required=True, help="Dataset root. Will create train_unlit/ and eval_lit/ inside.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Render mode.")
    parser.add_argument("--hdri", default="", help="HDRI path (required for eval mode).")
    parser.add_argument("--hdri-strength", type=float, default=1.0, help="HDRI intensity for eval mode.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Render resolution (square).")
    parser.add_argument("--views", type=int, default=None, help="Number of views. Defaults to 64 (train) / 20 (eval).")
    parser.add_argument("--radius", type=float, default=2.0, help="Camera orbit radius.")
    parser.add_argument("--focal-length", type=float, default=50.0, help="Camera focal length in mm.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples for eval mode (train uses 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-blend", action="store_true", help="Save a .blend file per object for inspection.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    args.out_root = os.path.abspath(args.out_root)
    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)

    if args.views is None:
        args.views = DEFAULT_VIEWS_TRAIN if args.mode == "train" else DEFAULT_VIEWS_EVAL

    if not os.path.exists(manifest_path):
        log(f"Manifest not found: {manifest_path}")
        return
    if args.mode == "eval" and (not args.hdri):
        log("Eval mode requires --hdri.")
        return

    os.makedirs(args.out_root, exist_ok=True)
    rng = random.Random(args.seed)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        tasks_raw = list(reader)

    tasks: List[Dict[str, str]] = []
    for row in tasks_raw:
        new_row = dict(row)
        new_row["mesh"] = resolve_path(row.get("mesh") or row.get("mesh_path"), manifest_dir)
        new_row["albedo"] = resolve_path(row.get("albedo"), manifest_dir)
        new_row["rough"] = resolve_path(row.get("rough") or row.get("roughness"), manifest_dir)
        new_row["metal"] = resolve_path(row.get("metal") or row.get("metallic"), manifest_dir)
        new_row["normal"] = resolve_path(row.get("normal"), manifest_dir)
        tasks.append(new_row)

    log(f"Loaded {len(tasks)} entries from manifest.")
    success = 0
    for idx, row in enumerate(tasks):
        oid = row.get("obj_id", f"idx_{idx}")
        log(f"[{idx + 1}/{len(tasks)}] Rendering {oid} ({args.mode})")
        try:
            if args.mode == "train":
                ok = render_train(row, args, rng)
            else:
                ok = render_eval(row, args, rng)
            success += int(ok)
        except Exception as e:
            log(f"{oid}: render failed with error {e}")
        finally:
            clear_data_blocks()

    log(f"Completed. Success: {success}/{len(tasks)}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    main(argv)
