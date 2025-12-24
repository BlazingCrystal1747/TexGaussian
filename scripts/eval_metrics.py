#!/usr/bin/env python3
"""
Compute evaluation metrics between ground-truth (GT) renders and generated (Gen) renders.

Metrics:
- Pixel/structural: PSNR, SSIM, LPIPS (VGG backbone)
- Distribution: FID, KID
- Semantic: CLIP image similarity (cosine between embeddings)

The script reads images directly from their source directories without copying them
elsewhere. GT and Gen filenames are assumed to align (e.g., 000.png matches 000.png).
"""

import argparse
import csv
import glob
import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import clip
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image

# Silence torchvision deprecated pretrained warnings triggered inside lpips
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated renders against GT renders.")
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Name of the experiment; used to resolve the generated render directory.",
    )
    parser.add_argument(
        "--base_gt_dir",
        default="../datasets/texverse_rendered/eval_lit",
        help="Root directory containing GT renders, organized by object id.",
    )
    parser.add_argument(
        "--base_gen_dir",
        default=None,
        help="Root directory containing generated renders. Defaults to "
        "'../experiments/{experiment_name}/texverse_gen_renders'.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric computation.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (e.g., 'cuda', 'cuda:1', or 'cpu'). Falls back to CPU if unavailable.",
    )
    parser.add_argument("--kid_subset_size", type=int, default=50, help="Subset size for KID computation.")
    parser.add_argument("--clip_model", default="ViT-B/32", help="CLIP model variant to load.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for metrics (JSON or CSV). Defaults to 'metrics_{experiment_name}.json' in CWD.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Which metrics to compute. Presets: all | pixel (psnr,ssim,lpips) | "
            "dist (fid,kid) | semantic (clip). "
            "You can also pass a comma list, e.g., 'psnr,ssim,clip'."
        ),
    )
    return parser.parse_args()


def resolve_gen_dir(args: argparse.Namespace) -> str:
    if args.base_gen_dir:
        return args.base_gen_dir
    return os.path.join("../experiments", args.experiment_name, "texverse_gen_renders")


def list_object_ids(base_gen_dir: str) -> List[str]:
    obj_ids = [
        d
        for d in os.listdir(base_gen_dir)
        if os.path.isdir(os.path.join(base_gen_dir, d))
    ]
    return sorted(obj_ids)


def collect_aligned_paths(base_gt_dir: str, base_gen_dir: str) -> Tuple[List[str], List[str], List[str]]:
    obj_ids = list_object_ids(base_gen_dir)
    if not obj_ids:
        raise FileNotFoundError(f"No object directories found under {base_gen_dir}")

    all_gt_paths: List[str] = []
    all_gen_paths: List[str] = []

    for obj_id in obj_ids:
        gen_paths = sorted(glob.glob(os.path.join(base_gen_dir, obj_id, "*.png")))
        gt_paths = sorted(glob.glob(os.path.join(base_gt_dir, obj_id, "*.png")))

        if not gen_paths:
            raise FileNotFoundError(f"No generated images found for {obj_id} in {base_gen_dir}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT images found for {obj_id} in {base_gt_dir}")
        if len(gen_paths) != len(gt_paths):
            raise ValueError(f"Image count mismatch for {obj_id}: {len(gen_paths)} gen vs {len(gt_paths)} gt")

        all_gen_paths.extend(gen_paths)
        all_gt_paths.extend(gt_paths)

    return obj_ids, all_gt_paths, all_gen_paths


def parse_metrics_arg(metrics_arg: str) -> Dict[str, bool]:
    metrics_arg = metrics_arg.lower().replace(" ", "")
    presets = {
        "all": {"psnr", "ssim", "lpips", "fid", "kid", "clip"},
        "pixel": {"psnr", "ssim", "lpips"},
        "structural": {"psnr", "ssim", "lpips"},
        "dist": {"fid", "kid"},
        "distribution": {"fid", "kid"},
        "semantic": {"clip"},
    }
    supported: Set[str] = {"psnr", "ssim", "lpips", "fid", "kid", "clip"}
    if metrics_arg in presets:
        selected = presets[metrics_arg]
    else:
        selected = {m for m in metrics_arg.split(",") if m}
        unknown = selected - supported
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("No metrics selected.")
    return {name: (name in selected) for name in supported}


def load_batch(
    paths: Sequence[str],
    clip_preprocess,
    device: torch.device,
    include_clip: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load images as float [0,1], uint8 [0,255], and optionally CLIP-preprocessed batches."""
    float_batch: List[torch.Tensor] = []
    uint8_batch: List[torch.Tensor] = []
    clip_batch: List[torch.Tensor] = []

    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            np_img = np.array(img, dtype=np.uint8)
            if include_clip:
                clip_tensor = clip_preprocess(img).unsqueeze(0)  # type: ignore[arg-type]
                clip_batch.append(clip_tensor)

        uint8_tensor = torch.from_numpy(np_img).permute(2, 0, 1)
        float_tensor = uint8_tensor.float().div(255.0)

        uint8_batch.append(uint8_tensor)
        float_batch.append(float_tensor)

    float_tensor_batch = torch.stack(float_batch, dim=0).to(device)
    uint8_tensor_batch = torch.stack(uint8_batch, dim=0).to(device)
    clip_tensor_batch = torch.cat(clip_batch, dim=0).to(device) if include_clip else None
    return float_tensor_batch, uint8_tensor_batch, clip_tensor_batch


def compute_clip_similarity(model, gen_batch: torch.Tensor, gt_batch: torch.Tensor) -> float:
    gen_feat = model.encode_image(gen_batch)
    gt_feat = model.encode_image(gt_batch)
    gen_feat = F.normalize(gen_feat, dim=-1)
    gt_feat = F.normalize(gt_feat, dim=-1)
    sim = (gen_feat * gt_feat).sum(dim=-1)
    return sim.mean().item()


def save_metrics(final_metrics: Dict[str, float], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in final_metrics.items():
                writer.writerow([k, v])
    else:
        with open(output_path, "w") as f:
            json.dump(final_metrics, f, indent=4)


def main() -> None:
    args = parse_args()
    metric_flags = parse_metrics_arg(args.metrics)
    base_gen_dir = resolve_gen_dir(args)
    base_gt_dir = args.base_gt_dir

    if not os.path.isdir(base_gen_dir):
        raise FileNotFoundError(f"Generated render directory not found: {base_gen_dir}")
    if not os.path.isdir(base_gt_dir):
        raise FileNotFoundError(f"GT render directory not found: {base_gt_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("CUDA requested but not available; falling back to CPU.")

    obj_ids, all_gt_paths, all_gen_paths = collect_aligned_paths(base_gt_dir, base_gen_dir)
    if not all_gt_paths:
        raise RuntimeError("No image pairs found.")

    print(f"Found {len(obj_ids)} objects and {len(all_gt_paths)} aligned image pairs.")

    # Guard against too-small datasets for KID subset sampling
    kid_subset_size = min(args.kid_subset_size, len(all_gt_paths)) if metric_flags["kid"] else None
    if kid_subset_size is not None and kid_subset_size < args.kid_subset_size:
        print(f"KID subset_size clipped to {kid_subset_size} due to limited samples.")

    psnr_metric = (
        torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
        if metric_flags["psnr"]
        else None
    )
    ssim_metric = (
        torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if metric_flags["ssim"]
        else None
    )
    fid_metric = (
        torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        if metric_flags["fid"]
        else None
    )
    kid_metric = (
        torchmetrics.image.kid.KernelInceptionDistance(subset_size=kid_subset_size, normalize=False).to(device)
        if metric_flags["kid"]
        else None
    )

    lpips_model = lpips.LPIPS(net="vgg").to(device) if metric_flags["lpips"] else None
    if lpips_model:
        lpips_model.eval()

    clip_model = None
    clip_preprocess = None
    if metric_flags["clip"]:
        clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
        clip_model.eval()

    lpips_scores: List[torch.Tensor] = []
    clip_scores: List[float] = []

    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(all_gen_paths), batch_size):
            batch_gen_paths = all_gen_paths[start : start + batch_size]
            batch_gt_paths = all_gt_paths[start : start + batch_size]

            gen_float, gen_uint8, gen_clip = load_batch(
                batch_gen_paths, clip_preprocess, device, include_clip=metric_flags["clip"]
            )
            gt_float, gt_uint8, gt_clip = load_batch(
                batch_gt_paths, clip_preprocess, device, include_clip=metric_flags["clip"]
            )

            if psnr_metric:
                psnr_metric.update(gen_float, gt_float)
            if ssim_metric:
                ssim_metric.update(gen_float, gt_float)

            if lpips_model:
                lpips_val = lpips_model(gen_float * 2.0 - 1.0, gt_float * 2.0 - 1.0)
                lpips_scores.append(lpips_val.mean().detach().cpu())

            if metric_flags["clip"] and clip_model and gen_clip is not None and gt_clip is not None:
                clip_scores.append(compute_clip_similarity(clip_model, gen_clip, gt_clip))

            if fid_metric:
                fid_metric.update(gt_uint8, real=True)
                fid_metric.update(gen_uint8, real=False)
            if kid_metric:
                kid_metric.update(gt_uint8, real=True)
                kid_metric.update(gen_uint8, real=False)

    final_metrics: Dict[str, float] = {}
    if psnr_metric:
        final_metrics["PSNR"] = psnr_metric.compute().item()
    if ssim_metric:
        final_metrics["SSIM"] = ssim_metric.compute().item()
    if lpips_model:
        final_metrics["LPIPS"] = torch.stack(lpips_scores).mean().item() if lpips_scores else float("nan")
    if metric_flags["clip"]:
        final_metrics["CLIP_Sim"] = float(np.mean(clip_scores)) if clip_scores else float("nan")
    if fid_metric:
        final_metrics["FID"] = fid_metric.compute().item()
    if kid_metric:
        try:
            final_metrics["KID"] = kid_metric.compute()[0].item()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: KID computation failed ({exc}); skipping KID.")
            final_metrics["KID"] = float("nan")

    output_path = args.output or f"metrics_{args.experiment_name}.json"
    save_metrics(final_metrics, output_path)
    print(f"Saved metrics to {output_path}:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
