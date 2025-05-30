import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import copy

import ocnn

from kiui.lpips import LPIPS
import open3d as o3d

# from core.unet import UNet
from core.octree_unet import OctreeUNet
from core.options import Options
from core.gs import GaussianRenderer
from ocnn.octree import Octree, Points
from core.utils import *
from clip_networks.network import CLIPTextEncoder

class TexGaussian(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        self.opt = opt
        self.device = device

        if self.opt.gaussian_loss:
            self.gaussian_mean = torch.load(self.opt.mean_path).to(self.device)
            self.gaussian_std = torch.load(self.opt.std_path).to(self.device)

        if self.opt.use_text:
            if self.opt.use_local_pretrained_ckpt:
                self.text_encoder = CLIPTextEncoder(model="./ViT-L-14.pt")
            else:
                self.text_encoder = CLIPTextEncoder(model="ViT-L/14")
            self.text_encoder.to(self.device)

            self.text_encoder.requires_grad_(False)

            self.text_encoder.eval()

        if self.opt.use_material:
            self.opt.out_channels += 3

        if self.opt.input_feature == 'L':
            self.opt.in_channels = 3

        elif self.opt.input_feature == 'ND':
            self.opt.in_channels = 4

        self.model = OctreeUNet(
            in_channels = self.opt.in_channels,
            out_channels = self.opt.out_channels,
            model_channels = self.opt.model_channels,
            channel_mult=self.opt.channel_mult,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_attention=self.opt.up_attention,
            num_heads = self.opt.num_heads,
            context_dim = self.opt.context_dim,
            use_checkpoint = self.opt.use_checkpoint,
        )

        # ema
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(self.device)

        self.ema_rate = self.opt.ema_rate
        self.ema_updater = EMA(self.ema_rate)
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.01 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)   # [0, 1]
        if self.opt.use_material:
            self.mr_act = lambda x: torch.sigmoid(x)

        self.input_depth = self.opt.input_depth
        self.full_depth = self.opt.full_depth

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            if self.opt.use_local_pretrained_ckpt:
                self.lpips_loss = LPIPS(net='vgg', use_url=False, checkpoint_dir = './lpips_checkpoints')
            else:
                self.lpips_loss = LPIPS(net ='vgg')
            self.lpips_loss.requires_grad_(False)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def forward_gaussians(self, x, octree, condition = None, data = None, ema = False):
        # x: [N, 4]
        # return: Gaussians: [N, dim_t]

        if ema:
            x = self.ema_model(x, octree, condition)
        else:
            x = self.model(x, octree, condition) # [N, out_channels]

        if self.opt.gaussian_loss:

            gaussian_loss = F.mse_loss(x, data['gaussian'])

            zeros = x.new_zeros([x.shape[0],4])
            x = torch.cat([x[:,:4], zeros, x[:,4:]], dim = 1)

            x = x * self.gaussian_std + self.gaussian_mean

        else:
            zeros = x.new_zeros([x.shape[0],4])
            x = torch.cat([x[:,:4], zeros, x[:,4:]], dim = 1)
            gaussian_loss = torch.zeros(1, device = self.device)

        pos = self.pos_act(octree.position) # [N, 3]
        opacity = self.opacity_act(x[:, :1]) # [N, 1]
        scale = self.scale_act(x[:, 1:4]) # [N, 3]
        rotation = self.rot_act(x[:, 4:8]) # [N, 4]
        rgbs = self.rgb_act(x[:, 8:11]) # [N, 3]
        if self.opt.use_material:
            mr = self.mr_act(x[:, 11:14]) # [N, 3]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [N, 14]
        if self.opt.use_material:
            mr_gaussians = torch.cat([pos, opacity, scale, rotation, mr], dim=-1) # [N, 14]
            return gaussian_loss, gaussians, mr_gaussians

        else:
            return gaussian_loss, gaussians


    def set_input(self, input=None):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        points = []

        for pts, norms in zip(input['points'], input['normals']):
            points.append(Points(points = pts.float(),normals = norms.float()))

        points = [pts.cuda(non_blocking=True) for pts in points]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()

        xyzb = octree.xyzb(depth = octree.depth, nempty = True)
        x, y, z, b = xyzb
        xyz = torch.stack([x,y,z], dim = 1)
        octree.position = 2 * xyz / (2 ** octree.depth) - 1

        input['octree'] = octree

        if self.opt.gaussian_loss:
            input['gaussian'] = (input['gaussian'] - self.gaussian_mean) / self.gaussian_std
            input['gaussian'] = torch.cat([input['gaussian'][:,:4], input['gaussian'][:,8:]], dim = 1)

        if self.opt.use_text:
            text_embeds = self.text_encoder.encode(input['token']).float()
            input['text_embedding'] = text_embeds  # [bs, 77, 768]

    def forward(self, data, ema = False):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        self.set_input(data)

        octree = data['octree']

        condition = None

        if self.opt.use_text:
            condition = data['text_embedding']  # [bs, 77, 768]

        input_feature = octree.get_input_feature(feature = self.opt.input_feature, nempty = True)
        if self.opt.use_material:
            gaussian_loss, gaussians, mr_gaussians = self.forward_gaussians(input_feature, octree, condition, data, ema = ema) # [N, 14]
        else:
            gaussian_loss, gaussians = self.forward_gaussians(input_feature, octree, condition, data, ema = ema) # [N, 14]
        batch_id = octree.batch_id(self.opt.input_depth, nempty = True)

        loss += gaussian_loss

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        if self.opt.use_material:
            mr_results = self.gs.render(mr_gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
            mr_pred_images = mr_results['image']

        results['images_pred'] = pred_images
        if self.opt.use_material:
            results['mr_images_pred'] = mr_pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        if self.opt.use_material:
            mr_gt_images = data['mr_images_output']
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)  # [B, V, 3, output_size, output_size]
        if self.opt.use_material:
            mr_gt_images = mr_gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        albedo_loss = F.mse_loss(pred_images, gt_images)
        if self.opt.use_material:
            mr_loss = F.mse_loss(mr_pred_images, mr_gt_images)

        mask_loss = F.mse_loss(pred_alphas, gt_masks)

        if self.opt.use_material:
            loss_mse = albedo_loss + mr_loss + mask_loss

        else:
            loss_mse = albedo_loss + mask_loss

        results['albedo_loss'] = albedo_loss.item()
        if self.opt.use_material:
            results['mr_loss'] = mr_loss.item()
        results['mask_loss'] = mask_loss.item()
        results['gaussian_loss'] = gaussian_loss.item()

        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            loss = loss + self.opt.lambda_lpips * loss_lpips
            if self.opt.use_material:
                mr_loss_lpips = self.lpips_loss(
                    F.interpolate(mr_gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    F.interpolate(mr_pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                ).mean()
                loss = loss + self.opt.lambda_lpips * mr_loss_lpips

        results['lpips_loss'] = self.opt.lambda_lpips * loss_lpips.item()

        if self.opt.use_material:
            results['mr_lpips_loss'] = self.opt.lambda_lpips * mr_loss_lpips.item()

        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

            if self.opt.use_material:
                mr_psnr = -10 * torch.log10(torch.mean((mr_pred_images.detach() - mr_gt_images) ** 2))
                results['mr_psnr'] = mr_psnr

        del octree

        return results
