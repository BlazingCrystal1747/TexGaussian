import os
import csv
import json
from dataclasses import asdict, is_dataclass
import tyro
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import trimesh
from core.regression_models import TexGaussian
from core.options import AllConfigs, Options
from core.gs import GaussianRenderer
from external.clip import tokenize

from ocnn.octree import Octree, Points
import ocnn

import nvdiffrast.torch as dr

import kiui
from kiui.mesh import Mesh
from kiui.op import uv_padding
from kiui.cam import orbit_camera, get_perspective

from safetensors.torch import load_file

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Converter(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.device = torch.device("cuda")

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

        self.gs_renderer = GaussianRenderer(opt)
        
        if self.opt.force_cuda_rast:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
    
        self.proj = torch.from_numpy(get_perspective(self.opt.fovy)).float().to(self.device)
        self.v = self.f = None
        self.vt = self.ft = None
        self.deform = None

        self.model = TexGaussian(opt, self.device)

        self.pointcloud_dir = self.opt.pointcloud_dir

        self.text_embedding = None
        if self.opt.use_text and self.opt.text_prompt:
            self.set_text_prompt(self.opt.text_prompt)
    
    def normalize_mesh(self):
        self.mesh.vertices = self.mesh.vertices - self.mesh.bounding_box.centroid
        distances = np.linalg.norm(self.mesh.vertices, axis=1)
        self.mesh.vertices /= np.max(distances)

    def set_text_prompt(self, text_prompt: str):
        """Update text prompt and re-encode embedding."""
        self.opt.text_prompt = text_prompt
        if not self.opt.use_text:
            self.text_embedding = None
            return

        token = tokenize(text_prompt)
        token = token.to(self.device)
        self.text_embedding = self.model.text_encoder.encode(token).float() # [bs, 77, 768]

    def load_mesh(self, path, num_samples = 200000):
        self.mesh = trimesh.load(path, force = 'mesh')
        self.normalize_mesh()

        point, idx = trimesh.sample.sample_surface(self.mesh, num_samples)
        normals = self.mesh.face_normals[idx]
        
        points_gt = Points(points = torch.from_numpy(point).float(), normals = torch.from_numpy(normals).float())
        points_gt.clip(min=-1, max=1)

        points = [points_gt]
        points = [pts.cuda(non_blocking = True) for pts in points]
        
        octrees = [self.points2octree(pts) for pts in points]
        octree_in = ocnn.octree.merge_octrees(octrees)

        octree_in.construct_all_neigh()

        xyzb = octree_in.xyzb(depth = octree_in.depth, nempty = True)
        x, y, z, b = xyzb
        xyz = torch.stack([x,y,z], dim = 1)
        octree_in.position = 2 * xyz / (2 ** octree_in.depth) - 1

        self.octree_in = octree_in

        self.input_data = self.octree_in.get_input_feature(feature = self.opt.input_feature, nempty = True) 

    def points2octree(self, points):
        octree = ocnn.octree.Octree(depth = self.opt.input_depth, full_depth = self.opt.full_depth)
        octree.build_octree(points)
        return octree
    
    def load_ckpt(self, ckpt_path):

        print('Start loading checkpoint')

        if ckpt_path.endswith('safetensors'):
            ckpt = load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = self.model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')
       
    @torch.no_grad()
    def render_gs(self, pose, use_material = False):
    
        cam_poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        batch_id = self.octree_in.batch_id(self.opt.input_depth, nempty = True)
        
        if use_material:
            out = self.gs_renderer.render(self.mr_gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        else:
            out = self.gs_renderer.render(self.gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        
        image = out['image'].squeeze(1).squeeze(0) # [C, H, W]
        alpha = out['alpha'].squeeze(2).squeeze(1).squeeze(0) # [H, W]

        return image, alpha
    
    def render_mesh(self, pose, use_material = False):

        h = w = self.opt.output_size

        v = self.v
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
        
        texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
        if use_material:
            image = torch.sigmoid(dr.texture(self.mr_albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]
        else:
            image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]

        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        image = alpha * image + (1 - alpha)

        return image, alpha
    
    # uv mesh refine
    def fit_mesh_uv(self, iters=1024, resolution=512, texture_resolution=1024, padding=2):

        if self.opt.use_material:
            _, self.gaussians, self.mr_gaussians = self.model.forward_gaussians(self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True)
        else:
            _, self.gaussians = self.model.forward_gaussians(self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True)

        self.opt.output_size = resolution

        v = self.mesh.vertices.astype(np.float32)
        f = self.mesh.faces.astype(np.int32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        self.f = torch.from_numpy(f).contiguous().int().to(self.device)

        # unwrap uv
        print(f"[INFO] uv unwrapping...")
        mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh.auto_uv()

        self.vt = mesh.vt
        self.ft = mesh.ft

        # render uv maps
        h = w = texture_resolution
        uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]
        xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(mesh.v[:, :1]).unsqueeze(0), rast, mesh.f) # [1, h, w, 1]

        # masked query 
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        
        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
        albedo = albedo.view(h, w, -1)
        mask = mask.view(h, w)
        albedo = uv_padding(albedo, mask, padding)

        if self.opt.use_material:
            mr_albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
            mr_albedo = mr_albedo.view(h, w, -1)
            mask = mask.view(h, w)
            mr_albedo = uv_padding(mr_albedo, mask, padding)

        # optimize texture
        self.albedo = nn.Parameter(albedo).to(self.device)

        if self.opt.use_material:
            self.mr_albedo = nn.Parameter(mr_albedo).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.albedo, 'lr': 1e-1},
        ])

        if self.opt.use_material:
            mr_optimizer = torch.optim.Adam([
                {'params': self.mr_albedo, 'lr': 1e-3},
            ])

        vers = [-89, 89, 0, 0, 0, 0]
        hors = [0, 0, -90, 0, 90, 180]

        rad = self.opt.texture_cam_radius # np.random.uniform(1, 2)

        for (ver, hor) in zip(vers, hors):

            print(f"[INFO] fitting mesh albedo...")
            pbar = tqdm.trange(iters)

            for i in pbar:
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose)
                image_pred, alpha_pred = self.render_mesh(pose)

                if self.opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
        pbar = tqdm.trange(iters * 2)

        for i in pbar:

            # shrink to front view as we care more about it...
            ver = np.random.randint(-89, 89)
            hor = np.random.randint(-180, 180)
            
            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            image_pred, alpha_pred = self.render_mesh(pose)

            if self.opt.save_image:
                image_gt_save = image_gt.detach().cpu().numpy()
                image_gt_save = image_gt_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                image_pred_save = image_pred.detach().cpu().numpy()
                image_pred_save = image_pred_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

            loss_mse = F.mse_loss(image_pred, image_gt)
            loss = loss_mse

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
        print(f"[INFO] finished fitting mesh albedo!")
    
        if self.opt.use_material:

            for (ver, hor) in zip(vers, hors):

                print(f"[INFO] fitting mesh material...")
                pbar = tqdm.trange(iters)

                for i in pbar:
                    
                    pose = orbit_camera(ver, hor, rad)
                    
                    image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                    image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                    if self.opt.save_image:
                        image_gt_save = image_gt.detach().cpu().numpy()
                        image_gt_save = image_gt_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                        image_pred_save = image_pred.detach().cpu().numpy()
                        image_pred_save = image_pred_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                    loss_mse = F.mse_loss(image_pred, image_gt)
                    loss = loss_mse

                    loss.backward()

                    mr_optimizer.step()
                    mr_optimizer.zero_grad()

                    pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
            pbar = tqdm.trange(iters * 2)

            for i in pbar:

                # shrink to front view as we care more about it...
                ver = np.random.randint(-89, 89)
                hor = np.random.randint(-180, 180)
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                if self.opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                mr_optimizer.step()
                mr_optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            
            print(f"[INFO] finished fitting mesh material!")


    @torch.no_grad()
    def export_mesh(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        v = self.mesh.vertices.astype(np.float32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        
        # Export a single mesh (geometry + UVs only) and write individual PBR textures.
        mesh = Mesh(v=self.v, f=self.f, vt=self.vt, ft=self.ft, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh_path = os.path.join(save_dir, 'mesh.obj')
        mesh.write(mesh_path)
        # Remove auxiliary files emitted by the writer (mtl / baked albedo).
        mtl_path = os.path.splitext(mesh_path)[0] + '.mtl'
        aux_albedo = os.path.join(save_dir, 'mesh_albedo.png')
        for p in (mtl_path, aux_albedo):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        albedo_img = torch.sigmoid(self.albedo).detach().clamp(0, 1).cpu().numpy()
        kiui.write_image(os.path.join(save_dir, 'albedo.png'), albedo_img)

        if self.opt.use_material and hasattr(self, "mr_albedo"):
            mr_img = torch.sigmoid(self.mr_albedo).detach().clamp(0, 1).cpu().numpy()
            roughness = mr_img[..., 1:2]  # G channel
            metallic = mr_img[..., 2:3]   # B channel
            kiui.write_image(os.path.join(save_dir, 'metallic.png'), metallic)
            kiui.write_image(os.path.join(save_dir, 'roughness.png'), roughness)

def load_batch_from_tsv(tsv_path: str, caption_field: str):
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required_captions = ("caption_short", "caption_long")
        missing = [name for name in required_captions if name not in fieldnames]
        if missing:
            raise ValueError(
                f"TSV missing required caption columns: {', '.join(missing)} "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        if caption_field not in fieldnames:
            raise ValueError(
                f"TSV missing caption_field '{caption_field}' "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        rows = [row for row in reader]

    return rows

def to_jsonable(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # fallback for types like numpy scalars etc.
    try:
        return obj.item()
    except Exception:
        return str(obj)

def save_experiment_config(exp_dir, opt, processed_samples, skipped_samples=None, manifest_path=None):
    cfg = {
        "options": to_jsonable(opt),
        "ckpt_path": opt.ckpt_path,
        "tsv_path": os.path.abspath(opt.tsv_path) if opt.tsv_path else None,
        "save_image": opt.save_image,
        "processed_samples": processed_samples,
    }
    if skipped_samples:
        cfg["skipped_samples"] = skipped_samples
    if manifest_path:
        cfg["result_tsv"] = os.path.abspath(manifest_path)

    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def build_result_row(obj_id: str, sample_dir: str, caption: str = None):
    """Collect generated asset paths for a single sample into a TSV-ready dict."""
    sample_dir = os.path.abspath(sample_dir)

    def path_if_exists(name: str) -> str:
        p = os.path.join(sample_dir, name)
        return os.path.abspath(p) if os.path.exists(p) else ""

    row = {
        "obj_id": obj_id,
        "mesh": path_if_exists("mesh.obj"),
        "albedo": path_if_exists("albedo.png"),
        "rough": path_if_exists("roughness.png"),
        "metal": path_if_exists("metallic.png"),
        "normal": path_if_exists("normal.png"),
    }
    if caption is not None:
        row["caption"] = caption
    return row


def write_result_manifest(tsv_path: str, rows):
    """Write generated asset info to a TSV following split_dataset-style columns."""
    if not rows:
        print(f"[WARN] No rows to write for manifest {tsv_path}")
        return

    fieldnames = ["obj_id", "mesh", "albedo", "rough", "metal", "normal"]
    if any("caption" in r for r in rows):
        fieldnames.append("caption")

    tsv_dir = os.path.dirname(tsv_path) or "."
    os.makedirs(tsv_dir, exist_ok=True)

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[INFO] Saved generated manifest to {tsv_path}")


if __name__ == "__main__":

    opt = tyro.cli(AllConfigs)

    opt.use_checkpoint = str2bool(opt.use_checkpoint)
    opt.use_material = str2bool(opt.use_material)
    opt.use_text = str2bool(opt.use_text)
    opt.save_image = str2bool(opt.save_image)
    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)

    if opt.tsv_path is None and opt.batch_path is not None:
        opt.tsv_path = opt.batch_path

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    textures_dir = os.path.join(output_dir, "textures") if opt.tsv_path else output_dir

    result_tsv_path = opt.result_tsv
    if result_tsv_path:
        if not os.path.isabs(result_tsv_path):
            result_tsv_path = os.path.abspath(os.path.join(output_dir, result_tsv_path))
    else:
        result_tsv_path = os.path.join(output_dir, "generated_manifest.tsv")
    result_tsv_path = os.path.abspath(result_tsv_path)
    opt.result_tsv = result_tsv_path

    converter = Converter(opt).cuda()
    converter.load_ckpt(opt.ckpt_path)

    if opt.tsv_path:
        tsv_dir = os.path.dirname(os.path.abspath(opt.tsv_path))
        batch_rows = load_batch_from_tsv(opt.tsv_path, opt.caption_field)
        print(f"[INFO] Loaded {len(batch_rows)} rows from {opt.tsv_path}")
        print(f"[INFO] Using caption field: {opt.caption_field}")
        os.makedirs(textures_dir, exist_ok=True)

        processed_samples = []
        skipped_samples = []

        for idx, row in enumerate(batch_rows):
            mesh_path = (row.get("mesh") or "").strip()
            caption = (row.get(opt.caption_field) or "").strip()
            obj_id = (row.get("obj_id") or "").strip() or f"sample_{idx}"

            if not mesh_path or not caption:
                print(f"[WARN] Skip row {idx}: missing mesh or caption (obj_id={obj_id})")
                skipped_samples.append({"obj_id": obj_id, "reason": "missing mesh or caption"})
                continue

            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(tsv_dir, mesh_path)

            converter.opt.texture_name = obj_id
            converter.opt.mesh_path = mesh_path
            converter.set_text_prompt(caption)

            sample_output_dir = os.path.join(textures_dir, converter.opt.texture_name)
            os.makedirs(sample_output_dir, exist_ok=True)

            print(f"[INFO] Processing {converter.opt.texture_name} ({idx + 1}/{len(batch_rows)})")
            converter.load_mesh(mesh_path)
            converter.fit_mesh_uv(iters = 1000)
            converter.export_mesh(sample_output_dir)

            processed_samples.append(build_result_row(
                converter.opt.texture_name,
                sample_output_dir,
                caption if caption else None,
            ))

        if processed_samples:
            write_result_manifest(result_tsv_path, processed_samples)
        save_experiment_config(output_dir, opt, processed_samples, skipped_samples, manifest_path=result_tsv_path if processed_samples else None)
    else:
        if opt.use_text and opt.text_prompt:
            converter.set_text_prompt(opt.text_prompt)
        converter.load_mesh(opt.mesh_path)
        converter.fit_mesh_uv(iters = 1000)
        sample_output_dir = os.path.join(textures_dir, opt.texture_name)
        converter.export_mesh(sample_output_dir)

        processed_samples = [build_result_row(opt.texture_name, sample_output_dir, opt.text_prompt if opt.text_prompt else None)]
        write_result_manifest(result_tsv_path, processed_samples)
        save_experiment_config(output_dir, opt, processed_samples, manifest_path=result_tsv_path)

    # converter.export_mesh(os.path.join(output_dir, opt.texture_name))
