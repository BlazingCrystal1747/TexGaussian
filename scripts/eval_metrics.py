#!/usr/bin/env python3
"""
Compute evaluation metrics between ground-truth (GT) renders and generated (Gen) renders.

Metrics:
- Masked (unlit, per-channel): Albedo PSNR/SSIM (sRGB), Roughness/Metallic L1, Normal MeanAngularError
- Distribution (lit only): FID, KID
- Semantic (lit only): CLIP image similarity, CLIP text similarity (caption_short), LongCLIP text similarity (caption_long)
- Optional: LPIPS on masked albedo when enabled

The script reads images directly from their source directories without copying them
elsewhere. GT and Gen filenames are assumed to align for each channel.
"""

import argparse
import csv
import gc
import glob
import json
import math
import os
import sys
import traceback
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Set environment variables BEFORE importing torch to prevent CUDA issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# Disable tokenizers parallelism to prevent deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Lazy imports for heavy modules to improve stability
clip = None
lpips = None
torchmetrics = None


def _lazy_import_clip():
    """Lazy import clip to avoid import-time crashes."""
    global clip
    if clip is None:
        import clip as _clip
        clip = _clip
    return clip


def _lazy_import_lpips():
    """Lazy import lpips to avoid import-time crashes."""
    global lpips
    if lpips is None:
        import lpips as _lpips
        lpips = _lpips
    return lpips


def _lazy_import_torchmetrics():
    """Lazy import torchmetrics to avoid import-time crashes."""
    global torchmetrics
    if torchmetrics is None:
        import torchmetrics as _torchmetrics
        torchmetrics = _torchmetrics
    return torchmetrics

# Silence torchvision deprecated pretrained warnings triggered inside lpips
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LONGCLIP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "third_party", "Long-CLIP"))
DEFAULT_LONGCLIP_MODEL = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
DEFAULT_LONGCLIP_CONTEXT_LENGTH = 248


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated renders against GT renders.")
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Name of the experiment; used to resolve the generated render directory.",
    )
    parser.add_argument(
        "--base_gt_dir",
        default="../datasets/texverse_rendered",
        help="Root directory containing GT renders, organized by object id.",
    )
    parser.add_argument(
        "--base_gen_dir",
        default=None,
        help="Root directory containing generated renders. Defaults to "
        "'../experiments/{experiment_name}/texverse_gen_renders'.",
    )
    parser.add_argument(
        "--lit_subdir",
        default="lit",
        help="Subdirectory under each obj_id that stores lit images.",
    )
    parser.add_argument(
        "--unlit_subdir",
        default="unlit",
        help="Subdirectory under each obj_id that stores unlit images.",
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
        "--longclip_model",
        default=DEFAULT_LONGCLIP_MODEL,
        help="Path to LongCLIP checkpoint (.pt) used with caption_long text similarity.",
    )
    parser.add_argument(
        "--longclip_root",
        default=DEFAULT_LONGCLIP_ROOT,
        help="Path to Long-CLIP repo for importing longclip when not installed.",
    )
    parser.add_argument(
        "--longclip_context_length",
        type=int,
        default=DEFAULT_LONGCLIP_CONTEXT_LENGTH,
        help="Context length for LongCLIP tokenization.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints and diff map dumps for masked metrics.",
    )
    parser.add_argument(
        "--prompts_file",
        default=None,
        help=(
            "Optional JSON mapping (obj_id -> {caption_short, caption_long}) or TSV manifest path "
            "(expects obj_id + caption_short + caption_long) for CLIP/LongCLIP text-image similarity."
        ),
    )
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
            "dist (fid,kid) | semantic (clip). Pixel metrics are computed per unlit channel. "
            "CLIP text similarity uses caption_short and LongCLIP uses caption_long when --prompts_file is provided. "
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

UNLIT_CHANNELS = ["albedo", "rough", "metal", "normal"]
CHANNEL_LABELS = {
    "albedo": "Albedo",
    "rough": "Roughness",
    "metal": "Metallic",
    "normal": "Normal",
}


def collect_lit_paths(
    base_gt_dir: str,
    base_gen_dir: str,
    lit_subdir: str,
    obj_ids: List[str],
) -> Tuple[List[str], List[str]]:
    all_gt_paths: List[str] = []
    all_gen_paths: List[str] = []

    for obj_id in obj_ids:
        gt_dir = os.path.join(base_gt_dir, obj_id, lit_subdir)
        gen_dir = os.path.join(base_gen_dir, obj_id, lit_subdir)
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*_beauty.png")))
        gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*_beauty.png")))

        if not gen_paths:
            raise FileNotFoundError(f"No generated lit images found for {obj_id} in {gen_dir}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT lit images found for {obj_id} in {gt_dir}")
        if len(gen_paths) != len(gt_paths):
            raise ValueError(f"Lit image count mismatch for {obj_id}: {len(gen_paths)} gen vs {len(gt_paths)} gt")

        gt_names = [os.path.basename(p) for p in gt_paths]
        gen_names = [os.path.basename(p) for p in gen_paths]
        if gt_names != gen_names:
            raise ValueError(f"Lit filename mismatch for {obj_id}")

        all_gt_paths.extend(gt_paths)
        all_gen_paths.extend(gen_paths)

    return all_gt_paths, all_gen_paths


def collect_unlit_channel_paths(
    base_gt_dir: str,
    base_gen_dir: str,
    unlit_subdir: str,
    channel: str,
    obj_ids: List[str],
) -> Tuple[List[str], List[str]]:
    all_gt_paths: List[str] = []
    all_gen_paths: List[str] = []
    pattern = f"*_{channel}.png"

    for obj_id in obj_ids:
        gt_dir = os.path.join(base_gt_dir, obj_id, unlit_subdir)
        gen_dir = os.path.join(base_gen_dir, obj_id, unlit_subdir)
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, pattern)))
        gen_paths = sorted(glob.glob(os.path.join(gen_dir, pattern)))

        if not gen_paths:
            raise FileNotFoundError(f"No generated {channel} images found for {obj_id} in {gen_dir}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT {channel} images found for {obj_id} in {gt_dir}")
        if len(gen_paths) != len(gt_paths):
            raise ValueError(
                f"{channel} image count mismatch for {obj_id}: {len(gen_paths)} gen vs {len(gt_paths)} gt"
            )

        gt_names = [os.path.basename(p) for p in gt_paths]
        gen_names = [os.path.basename(p) for p in gen_paths]
        if gt_names != gen_names:
            raise ValueError(f"{channel} filename mismatch for {obj_id}")

        all_gt_paths.extend(gt_paths)
        all_gen_paths.extend(gen_paths)

    return all_gt_paths, all_gen_paths


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


def resolve_longclip_module(longclip_root: Optional[str]) -> Tuple[object, str]:
    last_exc = None
    try:
        import longclip as longclip_module
        return longclip_module, "longclip"
    except Exception as exc:
        last_exc = exc

    candidates = []
    if longclip_root:
        candidates.append(longclip_root)
    if os.path.isdir(DEFAULT_LONGCLIP_ROOT) and DEFAULT_LONGCLIP_ROOT not in candidates:
        candidates.append(DEFAULT_LONGCLIP_ROOT)

    for root in candidates:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from model import longclip as longclip_module
            return longclip_module, f"model.longclip ({root})"
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "longclip is not available; install longclip or set --longclip_root "
        f"(default: {DEFAULT_LONGCLIP_ROOT})"
    ) from last_exc


def load_longclip_model(longclip_model_path: str, device: torch.device, longclip_root: Optional[str]):
    longclip_module, import_source = resolve_longclip_module(longclip_root)
    if not longclip_model_path:
        raise ValueError("LongCLIP model path is empty; provide --longclip_model.")
    
    # Handle relative paths - resolve from script directory if not absolute
    if not os.path.isabs(longclip_model_path):
        # Try relative to current working directory first
        if not os.path.isfile(longclip_model_path):
            # Try relative to script directory
            script_relative = os.path.join(SCRIPT_DIR, "..", longclip_model_path.lstrip("../"))
            script_relative = os.path.abspath(script_relative)
            if os.path.isfile(script_relative):
                longclip_model_path = script_relative
            else:
                # Try default location
                default_path = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
                if os.path.isfile(default_path):
                    longclip_model_path = default_path
    
    if not os.path.isfile(longclip_model_path):
        raise FileNotFoundError(f"LongCLIP model not found: {longclip_model_path}")
    
    print(f"Loading LongCLIP from: {longclip_model_path}")
    # Load to CPU first, then move to device to avoid GPU memory fragmentation
    model, preprocess = longclip_module.load(longclip_model_path, device="cpu")
    model = model.to(device)
    model.eval()
    return model, preprocess, longclip_module, import_source


def load_prompt_pairs(prompts_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    ext = os.path.splitext(prompts_path)[1].lower()
    if ext in {".tsv", ".csv"}:
        with open(prompts_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("Prompts TSV must include a header row.")
            required = {"obj_id", "caption_short", "caption_long"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Prompts TSV missing columns: {', '.join(sorted(missing))}")
            short_map: Dict[str, str] = {}
            long_map: Dict[str, str] = {}
            for row in reader:
                obj_id = (row.get("obj_id") or "").strip()
                if not obj_id:
                    continue
                caption_short = (row.get("caption_short") or "").strip()
                caption_long = (row.get("caption_long") or "").strip()
                if caption_short:
                    short_map[obj_id] = caption_short
                if caption_long:
                    long_map[obj_id] = caption_long
        return short_map, long_map

    with open(prompts_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompts file must be a JSON object mapping obj_id to prompt text.")

    short_map = {}
    long_map = {}
    fallback_used = False
    for obj_id, value in data.items():
        obj_id = str(obj_id).strip()
        if not obj_id:
            continue
        if isinstance(value, dict):
            short_val = value.get("caption_short", value.get("caption", ""))
            long_val = value.get("caption_long", value.get("caption", ""))
        else:
            short_val = value
            long_val = value
            fallback_used = True
        caption_short = str(short_val).strip() if short_val is not None else ""
        caption_long = str(long_val).strip() if long_val is not None else ""
        if caption_short:
            short_map[obj_id] = caption_short
        if caption_long:
            long_map[obj_id] = caption_long

    if fallback_used:
        print("[WARN] Prompts JSON uses obj_id -> string; using the same text for caption_short and caption_long.")

    return short_map, long_map


def obj_id_from_path(path: str, base_dir: str) -> str:
    rel = os.path.relpath(path, base_dir)
    return rel.split(os.sep)[0]


def extract_alpha_mask(gen_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
    if gen_tensor.shape[1] < 4 or gt_tensor.shape[1] < 4:
        raise ValueError("Alpha channel required for masked metrics. Ensure keep_alpha=True when loading.")
    gen_alpha = gen_tensor[:, 3:4, ...]
    gt_alpha = gt_tensor[:, 3:4, ...]
    return (gen_alpha > 0.5) & (gt_alpha > 0.5)


MIN_MASK_PIXELS = 10


def compute_masked_metrics(
    gen_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    metric_type: str = "color",
    debug: bool = False,
    obj_ids: Optional[Sequence[str]] = None,
    debug_names: Optional[Sequence[str]] = None,
    debug_dir: Optional[str] = None,
    debug_tag: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Input: gen_tensor, gt_tensor (B, 4, H, W) in [0, 1].
    Logic:
    1. Extract alpha and compute mask intersection.
    2. Compute metrics only over masked pixels.
    3. If mask pixels are extremely few, return NaN.
    4. Branch:
       - metric_type == 'normal_world': return (MeanAngularError, mask_count, aux).
       - metric_type == 'color': return (PSNR, mask_count, MSE) in sRGB space (albedo).
       - metric_type == 'scalar': return (L1, mask_count, aux) for roughness/metallic.
    """
    if gen_tensor.shape != gt_tensor.shape:
        raise ValueError("Gen/GT tensor shapes must match.")

    gen_rgb = gen_tensor[:, :3, ...]
    gt_rgb = gt_tensor[:, :3, ...]
    mask = extract_alpha_mask(gen_tensor, gt_tensor)
    mask_f = mask.float()
    mask_count = mask_f.sum().item()
    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        diff = (gen_tensor - gt_tensor).abs()
        diff_gray = diff.mean(dim=1).clamp(0.0, 1.0)
        for idx in range(diff_gray.shape[0]):
            name = None
            if debug_names and idx < len(debug_names):
                name = debug_names[idx]
            else:
                name = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
            safe_name = name.replace(os.sep, "_")
            tag = debug_tag or metric_type
            diff_path = os.path.join(debug_dir, f"{safe_name}_{tag}_diff.png")
            diff_img = (diff_gray[idx].detach().cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(diff_img, mode="L").save(diff_path)

    if mask_count < MIN_MASK_PIXELS:
        return float("nan"), 0.0, float("nan")

    if metric_type in {"normal_world", "normal"}:
        if debug:
            for idx in range(gen_rgb.shape[0]):
                oid = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
                mask_i = mask[idx].squeeze(0)
                if mask_i.any():
                    values = gen_rgb[idx][:, mask_i]
                    var = values.var(unbiased=False).item()
                else:
                    var = float("nan")
                print(f"[Debug {oid}] Normal Variance: {var:.6f}")

        gen_vec = gen_rgb * 2.0 - 1.0
        gt_vec = gt_rgb * 2.0 - 1.0
        gen_vec = F.normalize(gen_vec, dim=1, eps=1e-6)
        gt_vec = F.normalize(gt_vec, dim=1, eps=1e-6)
        dot = (gen_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
        ang_err = torch.acos(dot) * (180.0 / math.pi)
        mean_angular_error = (ang_err * mask.squeeze(1).float()).sum() / mask_count
        return mean_angular_error.item(), mask_count, float("nan")

    if metric_type == "color":
        if debug:
            for idx in range(gen_rgb.shape[0]):
                oid = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
                mask_i = mask[idx].squeeze(0)
                denom = mask_i.sum().item() * 3.0
                if denom > 0:
                    gen_mean = (gen_rgb[idx] * mask_i).sum().item() / denom
                    gt_mean = (gt_rgb[idx] * mask_i).sum().item() / denom
                else:
                    gen_mean = float("nan")
                    gt_mean = float("nan")
                print(
                    f"[Debug {oid}] Channel: {metric_type} | "
                    f"Gen Mean: {gen_mean:.4f} | GT Mean: {gt_mean:.4f}"
                )
        # Albedo PSNR/SSIM are computed in sRGB space (renderer outputs sRGB PNGs).
        diff = (gen_rgb - gt_rgb) * mask_f
        mse = diff.pow(2).sum() / (mask_count * 3.0)
        mse_val = mse.item()
        psnr = float("inf") if mse_val == 0 else -10.0 * math.log10(mse_val)
        return psnr, mask_count, mse_val

    if metric_type == "scalar":
        diff = (gen_rgb - gt_rgb) * mask_f
        l1 = diff.abs().sum() / (mask_count * 3.0)
        return l1.item(), mask_count, float("nan")

    raise ValueError(f"Unknown metric_type: {metric_type}")


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = gaussian[:, None] @ gaussian[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def compute_masked_ssim(
    gen_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tuple[float, float]:
    gen_rgb = gen_tensor[:, :3, ...]
    gt_rgb = gt_tensor[:, :3, ...]
    mask = extract_alpha_mask(gen_tensor, gt_tensor).float()
    mask_count = mask.sum().item()
    if mask_count < MIN_MASK_PIXELS:
        return float("nan"), 0.0

    device = gen_rgb.device
    dtype = gen_rgb.dtype
    kernel_2d = _gaussian_kernel(window_size, sigma, device, dtype)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(gen_rgb.shape[1], 1, 1, 1)
    pad = window_size // 2

    masked_gen = gen_rgb * mask
    masked_gt = gt_rgb * mask
    mask_sum = F.conv2d(mask, kernel[:1], padding=pad).clamp_min(1e-6)

    mu_gen = F.conv2d(masked_gen, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum
    mu_gt = F.conv2d(masked_gt, kernel, padding=pad, groups=gt_rgb.shape[1]) / mask_sum

    sigma_gen = F.conv2d(masked_gen * masked_gen, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum - mu_gen.pow(2)
    sigma_gt = F.conv2d(masked_gt * masked_gt, kernel, padding=pad, groups=gt_rgb.shape[1]) / mask_sum - mu_gt.pow(2)
    sigma_gen_gt = (
        F.conv2d(masked_gen * masked_gt, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum - mu_gen * mu_gt
    )

    sigma_gen = torch.clamp(sigma_gen, min=0.0)
    sigma_gt = torch.clamp(sigma_gt, min=0.0)

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim_map = ((2 * mu_gen * mu_gt + c1) * (2 * sigma_gen_gt + c2)) / (
        (mu_gen.pow(2) + mu_gt.pow(2) + c1) * (sigma_gen + sigma_gt + c2)
    )
    ssim_map = ssim_map.mean(dim=1, keepdim=True)
    ssim_val = (ssim_map * mask).sum() / mask_count
    return ssim_val.item(), mask_count


def _clear_cuda_cache():
    """Clear CUDA cache and run garbage collection to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass  # Ignore errors during cleanup
    gc.collect()


def encode_clip_texts(
    model,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    clip_module = _lazy_import_clip()
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            try:
                tokens = clip_module.tokenize(batch_prompts, truncate=True).to(device)
                text_feat = model.encode_text(tokens)
                text_feat = F.normalize(text_feat, dim=-1)
                # Move to CPU to reduce GPU memory pressure
                features.append(text_feat.cpu())
                del tokens, text_feat
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] CUDA OOM during text encoding at batch {start}, clearing cache and retrying...")
                    _clear_cuda_cache()
                    tokens = clip_module.tokenize(batch_prompts, truncate=True).to(device)
                    text_feat = model.encode_text(tokens)
                    text_feat = F.normalize(text_feat, dim=-1)
                    features.append(text_feat.cpu())
                    del tokens, text_feat
                else:
                    raise
            # Periodically clear cache
            if (start // batch_size) % 10 == 0:
                _clear_cuda_cache()
    result = torch.cat(features, dim=0).to(device)
    _clear_cuda_cache()
    return result


def encode_longclip_texts(
    model,
    longclip_module,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
    context_length: int,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            try:
                tokens = longclip_module.tokenize(
                    batch_prompts,
                    truncate=True,
                    context_length=context_length,
                ).to(device)
                text_feat = model.encode_text(tokens)
                text_feat = F.normalize(text_feat, dim=-1)
                # Move to CPU to reduce GPU memory pressure
                features.append(text_feat.cpu())
                del tokens, text_feat
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] CUDA OOM during LongCLIP text encoding at batch {start}, clearing cache and retrying...")
                    _clear_cuda_cache()
                    tokens = longclip_module.tokenize(
                        batch_prompts,
                        truncate=True,
                        context_length=context_length,
                    ).to(device)
                    text_feat = model.encode_text(tokens)
                    text_feat = F.normalize(text_feat, dim=-1)
                    features.append(text_feat.cpu())
                    del tokens, text_feat
                else:
                    raise
            # Periodically clear cache
            if (start // batch_size) % 10 == 0:
                _clear_cuda_cache()
    result = torch.cat(features, dim=0).to(device)
    _clear_cuda_cache()
    return result


def load_batch(
    paths: Sequence[str],
    clip_preprocess,
    device: torch.device,
    include_clip: bool,
    keep_alpha: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load images as float [0,1], uint8 [0,255], and optionally CLIP-preprocessed batches."""
    float_batch: List[torch.Tensor] = []
    uint8_batch: List[torch.Tensor] = []
    clip_batch: List[torch.Tensor] = []

    for path in paths:
        try:
            with Image.open(path) as img:
                if include_clip:
                    clip_img = img.convert("RGB")
                    if clip_preprocess is None:
                        raise ValueError("CLIP preprocess is required when include_clip=True.")
                    clip_tensor = clip_preprocess(clip_img).unsqueeze(0)  # type: ignore[arg-type]
                    clip_batch.append(clip_tensor)

                img = img.convert("RGBA" if keep_alpha else "RGB")
                np_img = np.array(img, dtype=np.uint8)

            uint8_tensor = torch.from_numpy(np_img).permute(2, 0, 1)
            float_tensor = uint8_tensor.float().div(255.0)

            uint8_batch.append(uint8_tensor)
            float_batch.append(float_tensor)
        except Exception as e:
            print(f"[WARN] Failed to load image {path}: {e}")
            raise

    # Use non_blocking=True for async transfer to reduce blocking
    float_tensor_batch = torch.stack(float_batch, dim=0).to(device, non_blocking=True)
    uint8_tensor_batch = torch.stack(uint8_batch, dim=0).to(device, non_blocking=True)
    clip_tensor_batch = torch.cat(clip_batch, dim=0).to(device, non_blocking=True) if include_clip and clip_batch else None
    return float_tensor_batch, uint8_tensor_batch, clip_tensor_batch


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

    batch_size = max(1, args.batch_size)
    final_metrics: Dict[str, float] = {}
    debug_dir = os.path.abspath("debug_output") if args.debug else None

    obj_ids = list_object_ids(base_gen_dir)
    if not obj_ids:
        raise FileNotFoundError(f"No object directories found under {base_gen_dir}")

    do_clip_image = metric_flags["clip"]
    do_clip_text = metric_flags["clip"] and args.prompts_file is not None
    do_longclip_text = metric_flags["clip"] and args.prompts_file is not None
    do_lit_metrics = metric_flags["fid"] or metric_flags["kid"] or do_clip_image or do_clip_text or do_longclip_text
    do_unlit_metrics = metric_flags["psnr"] or metric_flags["ssim"] or metric_flags["lpips"]

    if do_lit_metrics:
        lit_gt_paths, lit_gen_paths = collect_lit_paths(
            base_gt_dir,
            base_gen_dir,
            args.lit_subdir,
            obj_ids,
        )
        if not lit_gt_paths:
            raise RuntimeError("No lit image pairs found.")

        print(f"Lit pairs: {len(lit_gt_paths)} images across {len(obj_ids)} objects.")

        kid_subset_size = min(args.kid_subset_size, len(lit_gt_paths)) if metric_flags["kid"] else None
        if kid_subset_size is not None and kid_subset_size < args.kid_subset_size:
            print(f"KID subset_size clipped to {kid_subset_size} due to limited samples.")

        # Lazy import torchmetrics for FID/KID
        fid_metric = None
        kid_metric = None
        if metric_flags["fid"] or metric_flags["kid"]:
            torchmetrics_mod = _lazy_import_torchmetrics()
            if metric_flags["fid"]:
                fid_metric = torchmetrics_mod.image.fid.FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            if metric_flags["kid"]:
                kid_metric = torchmetrics_mod.image.kid.KernelInceptionDistance(subset_size=kid_subset_size, normalize=False).to(device)
            _clear_cuda_cache()

        clip_model = None
        clip_preprocess = None
        if do_clip_image or do_clip_text:
            print("Loading CLIP model...")
            try:
                _clear_cuda_cache()  # Clear before loading
                clip_module = _lazy_import_clip()
                # Load to CPU first, then move to device to avoid fragmentation
                clip_model, clip_preprocess = clip_module.load(args.clip_model, device="cpu")
                clip_model = clip_model.to(device)
                clip_model.eval()
                # Disable gradient computation for inference
                for param in clip_model.parameters():
                    param.requires_grad = False
                print(f"CLIP model loaded: {args.clip_model}")
                _clear_cuda_cache()
            except Exception as e:
                print(f"[ERROR] Failed to load CLIP model: {e}")
                traceback.print_exc()
                raise RuntimeError(f"CLIP model loading failed: {e}") from e

        longclip_model = None
        longclip_preprocess = None
        longclip_module = None
        if do_longclip_text:
            print("Loading LongCLIP model...")
            try:
                _clear_cuda_cache()  # Clear before loading
                longclip_model, longclip_preprocess, longclip_module, import_source = load_longclip_model(
                    args.longclip_model,
                    device,
                    args.longclip_root,
                )
                # Disable gradient computation for inference
                for param in longclip_model.parameters():
                    param.requires_grad = False
                print(f"LongCLIP model loaded from: {import_source}")
                _clear_cuda_cache()
            except Exception as e:
                print(f"[ERROR] Failed to load LongCLIP model: {e}")
                traceback.print_exc()
                raise RuntimeError(f"LongCLIP model loading failed: {e}") from e

        prompt_short_map = None
        prompt_long_map = None
        clip_text_feature_map = None
        longclip_text_feature_map = None
        if do_clip_text or do_longclip_text:
            if not os.path.isfile(args.prompts_file):
                raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
            prompt_short_map, prompt_long_map = load_prompt_pairs(args.prompts_file)

            if do_clip_text:
                if clip_model is None:
                    raise RuntimeError("CLIP model must be loaded to compute text-image similarity.")
                if not prompt_short_map:
                    raise ValueError("Prompts file contains no usable (obj_id, caption_short) pairs.")
                missing_short = [obj_id for obj_id in obj_ids if obj_id not in prompt_short_map]
                if missing_short:
                    preview = ", ".join(missing_short[:5])
                    raise ValueError(f"Missing caption_short for {len(missing_short)} obj_ids (e.g., {preview}).")
                prompt_texts = [prompt_short_map[obj_id] for obj_id in obj_ids]
                text_features = encode_clip_texts(clip_model, prompt_texts, device, batch_size)
                clip_text_feature_map = {obj_id: text_features[i] for i, obj_id in enumerate(obj_ids)}

            if do_longclip_text:
                if longclip_model is None or longclip_module is None:
                    raise RuntimeError("LongCLIP model must be loaded to compute text-image similarity.")
                if not prompt_long_map:
                    raise ValueError("Prompts file contains no usable (obj_id, caption_long) pairs.")
                missing_long = [obj_id for obj_id in obj_ids if obj_id not in prompt_long_map]
                if missing_long:
                    preview = ", ".join(missing_long[:5])
                    raise ValueError(f"Missing caption_long for {len(missing_long)} obj_ids (e.g., {preview}).")
                long_prompt_texts = [prompt_long_map[obj_id] for obj_id in obj_ids]
                long_text_features = encode_longclip_texts(
                    longclip_model,
                    longclip_module,
                    long_prompt_texts,
                    device,
                    batch_size,
                    args.longclip_context_length,
                )
                longclip_text_feature_map = {
                    obj_id: long_text_features[i] for i, obj_id in enumerate(obj_ids)
                }

        clip_image_scores: List[float] = []
        clip_text_scores: List[float] = []
        longclip_text_scores: List[float] = []
        total_batches = (len(lit_gen_paths) + batch_size - 1) // batch_size
        print(f"Processing {total_batches} batches for lit metrics...")
        
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(lit_gen_paths), batch_size)):
                batch_gen_paths = lit_gen_paths[start : start + batch_size]
                batch_gt_paths = lit_gt_paths[start : start + batch_size]

                try:
                    _, gen_uint8, gen_clip = load_batch(
                        batch_gen_paths,
                        clip_preprocess,
                        device,
                        include_clip=do_clip_image or do_clip_text,
                    )
                    _, gt_uint8, gt_clip = load_batch(
                        batch_gt_paths,
                        clip_preprocess,
                        device,
                        include_clip=do_clip_image,
                    )

                    gen_feat = None
                    if (do_clip_image or do_clip_text) and clip_model and gen_clip is not None:
                        gen_feat = clip_model.encode_image(gen_clip)
                        gen_feat = F.normalize(gen_feat, dim=-1)

                    if do_clip_image and clip_model and gen_feat is not None and gt_clip is not None:
                        gt_feat = clip_model.encode_image(gt_clip)
                        gt_feat = F.normalize(gt_feat, dim=-1)
                        sim = (gen_feat * gt_feat).sum(dim=-1)
                        clip_image_scores.append(sim.mean().item())
                        del gt_feat

                    batch_obj_ids = None
                    if do_clip_text or do_longclip_text:
                        batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]

                    if do_clip_text and gen_feat is not None and clip_text_feature_map is not None:
                        text_feat = torch.stack([clip_text_feature_map[obj_id] for obj_id in batch_obj_ids], dim=0)
                        sim = (gen_feat * text_feat).sum(dim=-1)
                        clip_text_scores.append(sim.mean().item())
                        del text_feat

                    # Clean up CLIP tensors before LongCLIP (safe deletion)
                    if gen_clip is not None:
                        del gen_clip
                    if gt_clip is not None:
                        del gt_clip
                    if gen_feat is not None:
                        del gen_feat
                    gen_clip = gt_clip = gen_feat = None  # Reset to None
                    if batch_idx % 5 == 0:
                        _clear_cuda_cache()

                    longclip_gen_feat = None
                    if do_longclip_text and longclip_model and longclip_preprocess is not None:
                        _, _, gen_longclip = load_batch(
                            batch_gen_paths,
                            longclip_preprocess,
                            device,
                            include_clip=True,
                        )
                        if gen_longclip is not None:
                            longclip_gen_feat = longclip_model.encode_image(gen_longclip)
                            longclip_gen_feat = F.normalize(longclip_gen_feat, dim=-1)
                            del gen_longclip

                    if do_longclip_text and longclip_gen_feat is not None and longclip_text_feature_map is not None:
                        if batch_obj_ids is None:
                            batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]
                        long_text_feat = torch.stack(
                            [longclip_text_feature_map[obj_id] for obj_id in batch_obj_ids],
                            dim=0,
                        )
                        sim = (longclip_gen_feat * long_text_feat).sum(dim=-1)
                        longclip_text_scores.append(sim.mean().item())
                        del long_text_feat
                    
                    # Safe cleanup of longclip_gen_feat
                    if longclip_gen_feat is not None:
                        del longclip_gen_feat
                    longclip_gen_feat = None

                    if fid_metric:
                        fid_metric.update(gt_uint8, real=True)
                        fid_metric.update(gen_uint8, real=False)
                    if kid_metric:
                        kid_metric.update(gt_uint8, real=True)
                        kid_metric.update(gen_uint8, real=False)

                    # Safe cleanup of batch tensors
                    del gen_uint8, gt_uint8

                except Exception as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "cuda" in error_msg or "memory" in error_msg:
                        print(f"[WARN] CUDA/Memory error at batch {batch_idx}/{total_batches}: {e}")
                        print("[WARN] Clearing cache and skipping this batch...")
                        _clear_cuda_cache()
                        # Skip this batch on OOM - better than crashing
                        continue
                    else:
                        print(f"[ERROR] Error at batch {batch_idx}: {e}")
                        traceback.print_exc()
                        raise

                # Progress and periodic cleanup
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{total_batches} batches")
                    _clear_cuda_cache()

        # Clean up models after lit metrics are done (safe deletion)
        print("Cleaning up CLIP/LongCLIP models...")
        if clip_model is not None:
            del clip_model
        if clip_preprocess is not None:
            del clip_preprocess
        if longclip_model is not None:
            del longclip_model
        if longclip_preprocess is not None:
            del longclip_preprocess
        if longclip_module is not None:
            del longclip_module
        if clip_text_feature_map is not None:
            del clip_text_feature_map
        if longclip_text_feature_map is not None:
            del longclip_text_feature_map
        clip_model = clip_preprocess = longclip_model = longclip_preprocess = None
        longclip_module = clip_text_feature_map = longclip_text_feature_map = None
        _clear_cuda_cache()

        if do_clip_image:
            final_metrics["CLIP_Image_Score"] = (
                float(np.mean(clip_image_scores)) if clip_image_scores else float("nan")
            )
        if do_clip_text:
            final_metrics["CLIP_Text_Score"] = (
                float(np.mean(clip_text_scores)) if clip_text_scores else float("nan")
            )
        if do_longclip_text:
            final_metrics["LongCLIP_Text_Score"] = (
                float(np.mean(longclip_text_scores)) if longclip_text_scores else float("nan")
            )
        if fid_metric:
            final_metrics["FID"] = fid_metric.compute().item()
            del fid_metric
        if kid_metric:
            try:
                final_metrics["KID"] = kid_metric.compute()[0].item()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Warning: KID computation failed ({exc}); skipping KID.")
                final_metrics["KID"] = float("nan")
            del kid_metric
        _clear_cuda_cache()

    if do_unlit_metrics:
        lpips_mod = _lazy_import_lpips()
        lpips_model = lpips_mod.LPIPS(net="vgg").to(device) if metric_flags["lpips"] else None
        if lpips_model:
            lpips_model.eval()
            _clear_cuda_cache()

        for channel in UNLIT_CHANNELS:
            gt_paths, gen_paths = collect_unlit_channel_paths(
                base_gt_dir,
                base_gen_dir,
                args.unlit_subdir,
                channel,
                obj_ids,
            )
            if not gt_paths:
                raise RuntimeError(f"No {channel} image pairs found.")

            label = CHANNEL_LABELS.get(channel, channel)
            print(f"Unlit {label}: {len(gt_paths)} image pairs.")

            total_mask = 0.0
            total_mse_weighted = 0.0
            total_l1_weighted = 0.0
            total_mean_angular_error_weighted = 0.0
            total_ssim_weighted = 0.0
            total_ssim_mask = 0.0
            lpips_scores: List[torch.Tensor] = []

            with torch.no_grad():
                for start in range(0, len(gen_paths), batch_size):
                    batch_gen_paths = gen_paths[start : start + batch_size]
                    batch_gt_paths = gt_paths[start : start + batch_size]
                    batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]
                    batch_debug_names = [
                        f"{obj_id}_{os.path.splitext(os.path.basename(path))[0]}"
                        for obj_id, path in zip(batch_obj_ids, batch_gen_paths)
                    ]

                    gen_float, _, _ = load_batch(
                        batch_gen_paths,
                        None,
                        device,
                        include_clip=False,
                        keep_alpha=True,
                    )
                    gt_float, _, _ = load_batch(
                        batch_gt_paths,
                        None,
                        device,
                        include_clip=False,
                        keep_alpha=True,
                    )

                    if channel == "normal":
                        mean_angular_error, mask_count, _ = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="normal_world",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mean_angular_error_weighted += mean_angular_error * mask_count
                            total_mask += mask_count
                    elif channel == "albedo":
                        _, mask_count, mse = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="color",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mask += mask_count
                            total_mse_weighted += mse * mask_count

                        if metric_flags["ssim"]:
                            ssim_val, ssim_mask = compute_masked_ssim(gen_float, gt_float)
                            if ssim_mask > 0:
                                total_ssim_weighted += ssim_val * ssim_mask
                                total_ssim_mask += ssim_mask
                    else:
                        # L1 reflects physical coefficient deviation for roughness/metallic.
                        l1, mask_count, _ = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="scalar",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mask += mask_count
                            total_l1_weighted += l1 * mask_count

                    if lpips_model and channel == "albedo":
                        mask = extract_alpha_mask(gen_float, gt_float).float()
                        gen_rgb = gen_float[:, :3, ...] * mask
                        gt_rgb = gt_float[:, :3, ...] * mask
                        lpips_val = lpips_model(gen_rgb * 2.0 - 1.0, gt_rgb * 2.0 - 1.0)
                        lpips_scores.append(lpips_val.mean().detach().cpu())

            if channel == "normal":
                # Normal_MeanAngularError is computed over the alpha-intersection mask.
                final_metrics["Normal_MeanAngularError"] = (
                    total_mean_angular_error_weighted / total_mask if total_mask > 0 else float("nan")
                )
                continue

            if channel == "albedo" and metric_flags["psnr"]:
                if total_mask > 0:
                    mse_val = total_mse_weighted / total_mask
                    final_metrics["Albedo_PSNR_Masked"] = (
                        float("inf") if mse_val == 0 else -10.0 * math.log10(mse_val)
                    )
                else:
                    final_metrics["Albedo_PSNR_Masked"] = float("nan")

            if channel == "albedo" and metric_flags["ssim"]:
                final_metrics["Albedo_SSIM_Masked"] = (
                    total_ssim_weighted / total_ssim_mask if total_ssim_mask > 0 else float("nan")
                )

            if channel == "albedo" and lpips_model:
                final_metrics["Albedo_LPIPS_Masked"] = (
                    torch.stack(lpips_scores).mean().item() if lpips_scores else float("nan")
                )

            if channel == "rough":
                final_metrics["Roughness_L1_Masked"] = (
                    total_l1_weighted / total_mask if total_mask > 0 else float("nan")
                )

            if channel == "metal":
                final_metrics["Metallic_L1_Masked"] = (
                    total_l1_weighted / total_mask if total_mask > 0 else float("nan")
                )

    output_path = args.output or f"metrics_{args.experiment_name}.json"
    save_metrics(final_metrics, output_path)
    print(f"Saved metrics to {output_path}:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
