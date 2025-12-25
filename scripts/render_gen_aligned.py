#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_gen_aligned.py

Render generated meshes/textures using the exact camera poses saved from GT renders
(`transforms.json`). This reproduces the same viewpoints for visual inspection or
metric computation.
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import bpy
from mathutils import Matrix

PASS_CONFIG = [
    ("albedo", "sRGB"),
    ("rough", "Non-Color"),
    ("metal", "Non-Color"),
    ("normal", "Non-Color"),
]


# ===================== 基础工具 =====================
def log(msg: str) -> None:
    print(f"[Render Gen] {msg}")


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def set_color_management(scene: bpy.types.Scene) -> None:
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.sequencer_colorspace_settings.name = "sRGB"


def reset_scene(width: int, height: int, samples: int) -> bpy.types.Scene:
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

    sc.render.resolution_x = int(width)
    sc.render.resolution_y = int(height)
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


def setup_background(scene: bpy.types.Scene, mode: str, hdri_path: str = "", strength: float = 1.0) -> None:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputWorld")

    has_hdri = hdri_path and os.path.exists(hdri_path)

    if has_hdri:
        # 1. Create the HDRI lighting node (always needed for lighting)
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path)
        bg_light = nodes.new("ShaderNodeBackground")
        bg_light.inputs["Strength"].default_value = strength
        links.new(env.outputs["Color"], bg_light.inputs["Color"])

        if mode == "hdri":
            scene.render.film_transparent = False
            links.new(bg_light.outputs["Background"], out.inputs["Surface"])
        elif mode == "transparent":
            scene.render.film_transparent = True
            links.new(bg_light.outputs["Background"], out.inputs["Surface"])
        else:
            # black or white
            scene.render.film_transparent = False
            
            mix = nodes.new("ShaderNodeMixShader")
            light_path = nodes.new("ShaderNodeLightPath")
            bg_cam = nodes.new("ShaderNodeBackground")
            
            if mode == "white":
                bg_cam.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                bg_cam.inputs["Strength"].default_value = 1.0
            else: # black
                bg_cam.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                bg_cam.inputs["Strength"].default_value = 0.0
            
            links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
            links.new(bg_light.outputs["Background"], mix.inputs[1])
            links.new(bg_cam.outputs["Background"], mix.inputs[2])
            links.new(mix.outputs["Shader"], out.inputs["Surface"])
            
    else:
        # No HDRI provided (e.g. unlit mode, or user didn't provide one)
        bg = nodes.new("ShaderNodeBackground")
        if mode == "white":
            scene.render.film_transparent = False
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 1.0
        elif mode == "black":
            scene.render.film_transparent = False
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
        else: # transparent or fallback
            scene.render.film_transparent = True
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
            
        links.new(bg.outputs["Background"], out.inputs["Surface"])


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


def build_geometry_normal_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    # Fallback for missing normal map:
    # Output Tangent Space Identity Normal (0.5, 0.5, 1.0)
    # This ensures compatibility with GT normal maps (which are in tangent space)
    # for metric computation (PSNR/SSIM).
    
    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (0.5, 0.5, 1.0, 1.0)
    emis.inputs["Strength"].default_value = 1.0
    
    out = nodes.new("ShaderNodeOutputMaterial")
    
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


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


def clear_data_blocks() -> None:
    for sc in list(bpy.data.scenes):
        sc.world = None
    for world in list(bpy.data.worlds):
        if world.use_nodes and world.node_tree:
            world.node_tree.links.clear()
            world.node_tree.nodes.clear()
        bpy.data.worlds.remove(world, do_unlink=True)

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

    for img in list(bpy.data.images):
        if img.name == "Render Result":
            continue
        img.user_clear()
        try:
            bpy.data.images.remove(img)
        except RuntimeError:
            pass


# ===================== 渲染逻辑 =====================
def load_transforms(transform_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    with open(transform_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intrinsics = data.get("intrinsics") or {}
    frames = data.get("frames") or []
    meta = data.get("meta") or {}
    return intrinsics, frames, meta


def frame_basename(frame: Dict[str, Any], idx: int) -> str:
    if frame.get("file_prefix"):
        return str(frame["file_prefix"])
    if frame.get("file_name"):
        return os.path.splitext(os.path.basename(frame["file_name"]))[0]
    return f"{idx:03d}"


def to_matrix(w2c_raw: Any) -> Matrix:
    mat = Matrix(w2c_raw)
    if len(mat) != 4 or len(mat[0]) != 4:
        raise ValueError("world_to_camera must be 4x4")
    return mat


def validate_intrinsics(cam: bpy.types.Object, scene: bpy.types.Scene, target: Dict[str, Any], oid: str) -> None:
    if not target:
        return
    actual = compute_intrinsics(cam, scene)
    keys = ["fx", "fy", "cx", "cy", "w", "h"]
    diffs = []
    for k in keys:
        if k in target:
            diffs.append(abs(actual[k] - float(target[k])))
    if diffs and max(diffs) > 1e-2:
        log(f"{oid}: intrinsics drift detected (max diff {max(diffs):.4f})")


def render_object(row: Dict[str, str], args: argparse.Namespace) -> bool:
    oid = row.get("obj_id", "unknown")
    mesh_path = row.get("mesh")
    albedo = row.get("albedo")
    rough = row.get("rough") or row.get("roughness")
    metal = row.get("metal") or row.get("metallic")
    normal = row.get("normal")

    transform_path = row.get("transforms")
    if not transform_path:
        transform_path = os.path.join(args.gt_root, args.transforms_subdir, oid, "transforms.json")

    if not mesh_path or not os.path.exists(mesh_path):
        log(f"{oid}: mesh missing or not found.")
        return False
    if not os.path.exists(transform_path):
        log(f"{oid}: transforms.json not found at {transform_path}")
        return False

    intrinsics, frames, meta = load_transforms(transform_path)
    if not frames:
        log(f"{oid}: no frames in transforms.json, skip.")
        return False
    if "w" not in intrinsics or "h" not in intrinsics:
        log(f"{oid}: transforms missing resolution.")
        return False

    width = int(intrinsics["w"])
    height = int(intrinsics["h"])
    lens_mm = meta.get("focal_length_mm")
    if lens_mm is None and intrinsics.get("fx"):
        sensor_width = 36.0
        lens_mm = float(intrinsics["fx"]) * sensor_width / float(width)
    lens_mm = float(lens_mm) if lens_mm is not None else args.fallback_focal

    samples = 1 if args.mode == "unlit" else args.samples
    scene = reset_scene(width, height, samples)
    
    try:
        setup_background(scene, args.background, args.hdri, args.hdri_strength)
    except Exception as e:
        log(f"{oid}: background setup failed ({e}), skip.")
        return False

    mesh_obj = import_and_normalize(mesh_path)
    if not mesh_obj:
        log(f"{oid}: import/normalize failed.")
        return False

    def usable(path: str) -> str:
        return path if path and os.path.exists(path) else ""

    albedo = usable(albedo)
    rough = usable(rough)
    metal = usable(metal)
    normal = usable(normal)
    if args.mode == "beauty" and not albedo:
        log(f"{oid}: beauty mode requires an albedo texture.")
        return False

    if args.mode == "beauty":
        mat_beauty = build_pbr_material(albedo, rough, metal, normal)
    else:
        tex_map = {
            "albedo": albedo,
            "rough": rough,
            "metal": metal,
            "normal": normal,
        }
        emission_mats = {}
        for name, cs in PASS_CONFIG:
            path = tex_map.get(name)
            if path and os.path.exists(path):
                emission_mats[name] = build_emission_material(f"{name.upper()}_EMIT", path, cs)
            elif name == "normal":
                emission_mats[name] = build_geometry_normal_material(f"{name.upper()}_GEO")
            else:
                emission_mats[name] = None

    cam = create_camera(scene, lens_mm)
    validate_intrinsics(cam, scene, intrinsics, oid)

    sub_dir = "eval_lit" if args.mode == "beauty" else "train_unlit"
    out_dir = os.path.join(args.out_root, sub_dir, oid)
    os.makedirs(out_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        try:
            w2c = to_matrix(frame["world_to_camera"])
        except Exception as e:
            log(f"{oid}: invalid world_to_camera at frame {idx} ({e})")
            continue

        try:
            cam.matrix_world = w2c.inverted()
        except Exception as e:
            log(f"{oid}: failed to invert w2c at frame {idx} ({e})")
            continue

        bpy.context.view_layer.update()

        restored_w2c = cam.matrix_world.inverted()
        diff = max(abs(restored_w2c[i][j] - w2c[i][j]) for i in range(4) for j in range(4))
        if diff > 1e-4:
            log(f"{oid}: w2c mismatch at frame {idx}, max diff {diff:.6f}")

        base = frame_basename(frame, idx)
        if args.mode == "beauty":
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(mat_beauty)
            out_name = frame.get("file_name") or f"{base}_beauty.png"
            scene.render.filepath = os.path.join(out_dir, out_name)
            bpy.ops.render.render(write_still=True)
        else:
            for pass_name, _ in PASS_CONFIG:
                mat = emission_mats.get(pass_name)
                if mat is None:
                    continue
                mesh_obj.data.materials.clear()
                mesh_obj.data.materials.append(mat)
                out_name = f"{base}_{pass_name}.png"
                scene.render.filepath = os.path.join(out_dir, out_name)
                bpy.ops.render.render(write_still=True)

    if args.save_blend:
        blend_path = os.path.join(out_dir, "scene.blend")
        try:
            bpy.ops.wm.save_mainfile(filepath=blend_path)
            log(f"{oid}: saved blend to {blend_path}")
        except Exception as e:
            log(f"{oid}: failed to save blend file ({e})")

    log(f"{oid}: rendered to {out_dir}")
    return True


# ===================== 入口 =====================
def pick_field(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated assets with GT camera poses (transforms.json).")
    parser.add_argument("--manifest", required=True, help="TSV manifest with columns: obj_id, mesh, albedo[, rough, metal, normal, transforms].")
    parser.add_argument("--gt-root", required=True, help="GT render root containing transforms (e.g., datasets/texverse_rendered).")
    parser.add_argument("--transforms-subdir", default="eval_lit", help="Subfolder under gt-root that stores transforms.json (eval_lit or train_unlit).")
    parser.add_argument("--out-root", required=True, help="Where rendered images will be saved (per obj_id subfolder).")
    parser.add_argument("--mode", choices=["beauty", "unlit"], default="beauty", help="beauty=Principled + HDRI; unlit=per-map emission.")
    parser.add_argument("--hdri", default="", help="HDRI path (required for beauty mode).")
    parser.add_argument("--hdri-strength", type=float, default=1.0, help="HDRI intensity for beauty mode.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples for beauty mode (unlit always uses 1).")
    parser.add_argument("--background", choices=["black", "white", "hdri", "transparent"], default=None, help="Background mode.")
    parser.add_argument("--fallback-focal", type=float, default=50.0, help="Fallback focal length (mm) if transforms.json lacks focal metadata.")
    parser.add_argument("--save-blend", action="store_true", help="Save a .blend per object for debugging.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    args.out_root = os.path.abspath(args.out_root)
    args.gt_root = os.path.abspath(args.gt_root)
    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)

    if args.background is None:
        args.background = "hdri" if args.mode == "beauty" else "transparent"

    if not os.path.exists(manifest_path):
        log(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows_raw = list(reader)

    tasks: List[Dict[str, str]] = []
    for row in rows_raw:
        new_row = dict(row)
        new_row["mesh"] = resolve_path(pick_field(row, ["mesh", "mesh_path"]), manifest_dir)
        new_row["albedo"] = resolve_path(row.get("albedo"), manifest_dir)
        new_row["rough"] = resolve_path(row.get("rough") or row.get("roughness"), manifest_dir)
        new_row["metal"] = resolve_path(row.get("metal") or row.get("metallic"), manifest_dir)
        new_row["normal"] = resolve_path(row.get("normal"), manifest_dir)
        new_row["transforms"] = resolve_path(row.get("transforms"), manifest_dir)
        tasks.append(new_row)

    os.makedirs(args.out_root, exist_ok=True)
    log(f"Loaded {len(tasks)} entries from manifest.")

    success = 0
    for idx, row in enumerate(tasks):
        oid = row.get("obj_id", f"idx_{idx}")
        log(f"[{idx + 1}/{len(tasks)}] Rendering {oid}")
        try:
            ok = render_object(row, args)
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
