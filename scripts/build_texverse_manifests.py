#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build TexVerse manifests (train/val/test/display) — minimal version.

Design:
- Only use caption.json; if missing -> drop (no templating, no length checks)
- Strict filters (always ON):
  * ID must be in TexVerse_pbr_id_list.txt (PBR whitelist)
  * pbrType exists (after normalization), max_texture == 2048
  * has a 2K glb path
- Recommended filters (configurable):
  * faceCount in [face-min, face-max]
  * drop rigged / animated by default
- Balanced sampling across material buckets; ensure SG ratio >= 0.20
- Optional:
  * --verify-pbr : GLB JSON check to confirm PBR & rectify MR/SG
  * --download [--also-1k] : precise per-file download (no repo traversal)

Output TSV columns:
  rel_glb<TAB>caption<TAB>workflow<TAB>obj_id<TAB>license
"""

import os
import sys
import json
import argparse
import random
import struct
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# --------- Buckets & license whitelist (kept minimal) ---------

BUCKET_KWS = {
    "wood":     ["wood", "oak", "walnut", "timber", "plywood", "maple", "pine", "birch"],
    "metal":    ["metal", "steel", "iron", "aluminum", "aluminium", "copper", "brass", "chrome", "chromium", "brushed"],
    "fabric":   ["fabric", "cloth", "textile", "linen", "cotton", "wool", "denim", "velvet", "silk", "canvas"],
    "leather":  ["leather", "suede"],
    "plastic":  ["plastic", "polymer", "acrylic", "abs", "petg", "pmma", "polycarbonate"],
    "ceramic":  ["ceramic", "porcelain", "clay", "earthenware", "terracotta", "glaze", "glazed"],
    "glass":    ["glass", "transparent", "translucent"],
    "stone":    ["stone", "marble", "granite", "slate", "concrete", "cement", "plaster"],
    "paint":    ["paint", "painted", "coated", "lacquer", "varnish", "matte", "glossy", "satin"],
    "composite":["carbon fiber", "composite", "rubber", "foam", "resin"],
}

DISPLAY_LICENSE_ALLOW = {"CC Attribution","Creative Commons Attribution","CC BY",
                         "CC Attribution-ShareAlike","CC BY-SA","Public Domain","CC0"}


# ---------------- Utils ----------------

def normalize_pbrtype(v):
    """normalize pbrType: empty/NULL-like -> None; else lower string"""
    if v is None: return None
    s = str(v).strip().lower()
    if s in {"", "null", "none", "n/a", "na", "unknown"}:
        return None
    return s

def is_2k_path(p: str) -> bool:
    return ("glbs/glbs_2k/" in p) and p.endswith("_2048.glb")

def is_1k_path(p: str) -> bool:
    return ("glbs/glbs_1k/" in p) and p.endswith("_1024.glb")

def infer_workflow(pbr_type_norm: Optional[str]) -> str:
    if not pbr_type_norm:
        return "MR"
    if pbr_type_norm.startswith("metal"):  # "metalness", "metallic-roughness"
        return "MR"
    if pbr_type_norm.startswith("spec"):   # "specular_glossiness"
        return "SG"
    return "MR"

def pick_caption_only(obj_id: str, caps: Dict[str, Any]) -> Optional[str]:
    c = caps.get(obj_id)
    if isinstance(c, dict):
        return c.get("material_desc") or c.get("caption")
    if isinstance(c, str):
        return c
    return None

def get_bucket(text: str) -> str:
    t = (text or "").lower()
    for b, kws in BUCKET_KWS.items():
        for kw in kws:
            if kw in t:
                return b
    return "misc"

def passes_geometry(meta_item: Dict[str,Any], fmin:int, fmax:int, allow_rigged:bool, allow_anim:bool) -> bool:
    faces = int(meta_item.get("faceCount", 0) or 0)
    if faces and (faces < fmin or faces > fmax):
        return False
    if not allow_rigged and bool(meta_item.get("isRigged", False)):
        return False
    anim = meta_item.get("animation", 0)
    if isinstance(anim, bool): anim = int(anim)
    if not allow_anim and int(anim or 0) != 0:
        return False
    return True

def bucket_balance_sample(items: List[Dict[str,Any]], n_target:int, mr_sg_min:float=0.20, seed:int=13) -> List[Dict[str,Any]]:
    random.seed(seed)
    by_bucket = defaultdict(list)
    for it in items:
        by_bucket[it["bucket"]].append(it)
    for b in by_bucket:
        random.shuffle(by_bucket[b])

    buckets = list(by_bucket.keys())
    if not buckets: return []

    sampled: List[Dict[str,Any]] = []
    idx = {b:0 for b in buckets}
    while len(sampled) < n_target:
        progress=False
        for b in buckets:
            i = idx[b]
            if i < len(by_bucket[b]):
                sampled.append(by_bucket[b][i]); idx[b]+=1; progress=True
                if len(sampled) >= n_target: break
        if not progress: break

    def sg_ratio(lst): return 0.0 if not lst else sum(1 for x in lst if x["workflow"]=="SG")/len(lst)

    if sg_ratio(sampled) < mr_sg_min:
        remaining_sg = [it for b in buckets for it in by_bucket[b][idx[b]:] if it["workflow"]=="SG"]
        rep_idx = [i for i,x in reversed(list(enumerate(sampled))) if x["workflow"]=="MR"]
        j=0
        for pos in rep_idx:
            if j>=len(remaining_sg): break
            sampled[pos] = remaining_sg[j]; j+=1
            if sg_ratio(sampled) >= mr_sg_min: break
    return sampled[:n_target]

def write_tsv(path:str, rows:List[Dict[str,Any]])->None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['rel_glb']}\t{r['caption']}\t{r['workflow']}\t{r['obj_id']}\t{r.get('license','')}\n")

def maybe_download(rows:List[Dict[str,Any]], repo_id:str, local_dir:str, also_1k:bool=False)->None:
    if hf_hub_download is None:
        print("[WARN] huggingface_hub not installed; skip download"); return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","1")
    seen=set(); total=len(rows)
    for i,r in enumerate(rows,1):
        rel=r["rel_glb"]
        if rel not in seen:
            hf_hub_download(repo_id=repo_id, repo_type="dataset",
                            filename=rel, local_dir=local_dir,
                            local_dir_use_symlinks=False)
            seen.add(rel)
        if also_1k and r.get("rel_glb_1k"):
            rel1=r["rel_glb_1k"]
            if rel1 not in seen:
                hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                filename=rel1, local_dir=local_dir,
                                local_dir_use_symlinks=False)
                seen.add(rel1)
        if i%200==0 or i==total:
            print(f"[info] downloaded {i}/{total} (2k + optional 1k)")

# ---- optional: GLB-level verify (JSON chunk) ----
def _glb_read_json(glb_path: str) -> Optional[Dict[str,Any]]:
    try:
        with open(glb_path,"rb") as f: data=f.read()
        if data[0:4] != b"glTF": return None
        json_len = struct.unpack_from("<I", data, 12)[0]
        json_type = data[16:20]
        if json_type != b"JSON": return None
        json_str = data[20:20+json_len].decode("utf-8", errors="ignore")
        return json.loads(json_str)
    except Exception:
        return None

def verify_pbr_and_workflow(glb_path:str)->(bool, Optional[str]):
    js=_glb_read_json(glb_path)
    if not js: return False, None
    mats=js.get("materials",[])
    has_mr=False; has_sg=False; has_tex=False
    for m in mats:
        if "pbrMetallicRoughness" in m:
            has_mr=True
            pmr=m.get("pbrMetallicRoughness",{})
            if "metallicRoughnessTexture" in pmr or "baseColorTexture" in pmr:
                has_tex=True
        ex=(m.get("extensions") or {})
        if "KHR_materials_pbrSpecularGlossiness" in ex:
            has_sg=True
            sg=ex["KHR_materials_pbrSpecularGlossiness"]
            if "specularGlossinessTexture" in sg or "diffuseTexture" in sg:
                has_tex=True
    wf="SG" if has_sg else ("MR" if has_mr else None)
    ok=(wf is not None) and has_tex
    return ok, wf


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="YiboZhang2001/TexVerse")
    ap.add_argument("--data-root", default="../datasets/texverse")
    ap.add_argument("--out-dir", default="../datasets/texverse/splits")

    # split & sampling
    ap.add_argument("--train-num", type=int, default=800)
    ap.add_argument("--val-num",   type=int, default=150)
    ap.add_argument("--test-num",  type=int, default=200)
    ap.add_argument("--seed",      type=int, default=13)
    ap.add_argument("--min-sg-ratio", type=float, default=0.20)
    ap.add_argument("--no-bucket-balance", dest="enforce_bucket_balance", action="store_false")
    ap.set_defaults(enforce_bucket_balance=True)

    # geometry/animation filters
    ap.add_argument("--face-min", type=int, default=10000)
    ap.add_argument("--face-max", type=int, default=50000)
    ap.add_argument("--allow-rigged", action="store_true")
    ap.add_argument("--allow-anim", action="store_true")

    # strict PBR gating (always ON, but paths configurable)
    ap.add_argument("--pbr-id-file", default="TexVerse_pbr_id_list.txt")

    # optional features
    ap.add_argument("--verify-pbr", action="store_true", help="verify .glb PBR & rectify workflow if local file exists")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--also-1k",  action="store_true")

    args = ap.parse_args()

    # required files
    meta_fp = os.path.join(args.data_root, "metadata.json")
    caps_fp = os.path.join(args.data_root, "caption.json")
    pbr_fp  = args.pbr_id_file if os.path.isabs(args.pbr_id_file) else os.path.join(args.data_root, args.pbr_id_file)

    for fp in [meta_fp, caps_fp, pbr_fp]:
        if not os.path.exists(fp):
            print(f"[ERR] Not found: {fp}"); sys.exit(1)

    with open(meta_fp,"r",encoding="utf-8") as f: meta = json.load(f)
    with open(caps_fp,"r",encoding="utf-8") as f: caps = json.load(f)
    with open(pbr_fp,"r",encoding="utf-8") as f:
        pbr_id_set = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    print(f"[info] Loaded PBR whitelist: {len(pbr_id_set)} ids")

    # collect candidates
    candidates: List[Dict[str,Any]] = []
    total=len(meta); kept=0
    for idx,(obj_id,m) in enumerate(meta.items(),1):
        # PBR whitelist (hard gate)
        if obj_id not in pbr_id_set:
            continue

        # pbrType & 2048
        pbr_type_norm = normalize_pbrtype(m.get("pbrType", None))
        if not pbr_type_norm:
            continue
        if int(m.get("max_texture",0) or 0) != 2048:
            continue

        # glb paths
        glb_paths = m.get("glb_paths", [])
        paths_list: List[str] = []
        if isinstance(glb_paths, list):
            paths_list = glb_paths
        elif isinstance(glb_paths, dict):
            for v in glb_paths.values():
                if isinstance(v, list): paths_list.extend(v)
                elif isinstance(v, str): paths_list.append(v)
        elif isinstance(glb_paths, str):
            paths_list = [glb_paths]

        p2k = [p for p in paths_list if is_2k_path(p)]
        if not p2k:
            continue
        rel_glb = p2k[0]

        # geometry/animation
        if not passes_geometry(m, args.face_min, args.face_max, args.allow_rigged, args.allow_anim):
            continue

        # caption: ONLY from caption.json
        cap = pick_caption_only(obj_id, caps)
        if not cap:
            continue

        lic = str(m.get("license",""))
        wf  = infer_workflow(pbr_type_norm)

        item = {
            "obj_id": obj_id,
            "workflow": wf,
            "caption": cap,
            "license": lic,
            "rel_glb": rel_glb,
            "rel_glb_1k": None,
        }
        p1k = [p for p in paths_list if is_1k_path(p)]
        if p1k: item["rel_glb_1k"] = p1k[0]

        # optional verify (only if local 2k exists)
        if args.verify_pbr:
            local2k = os.path.join(args.data_root, rel_glb)
            if os.path.exists(local2k):
                ok, wf_from_glb = verify_pbr_and_workflow(local2k)
                if not ok: continue
                if wf_from_glb: item["workflow"] = wf_from_glb

        # bucket text = caption + tags + name
        tags = [str(t) for t in (m.get("tags") or []) if t]
        name = m.get("name") or ""
        item["bucket"] = get_bucket(f"{cap} {' '.join(tags)} {name}")

        candidates.append(item); kept+=1
        if idx % 100000 == 0:
            print(f"[scan] {idx}/{total} | kept={kept}")

    print(f"[scan] finished. candidates={len(candidates)}")

    # shuffle & sample
    need_total = args.train_num + args.val_num + args.test_num
    if need_total <= 0:
        print("[ERR] split sizes are zero."); sys.exit(1)

    random.seed(args.seed)
    random.shuffle(candidates)
    if args.enforce_bucket_balance:
        sampled = bucket_balance_sample(candidates, need_total, mr_sg_min=args.min_sg_ratio, seed=args.seed)
    else:
        sampled = candidates[:need_total]

    if len(sampled) < need_total:
        print(f"[WARN] sampled {len(sampled)} < requested {need_total}. Consider relaxing filters.")

    train = sampled[:args.train_num]
    val   = sampled[args.train_num: args.train_num+args.val_num]
    test  = sampled[args.train_num+args.val_num: args.train_num+args.val_num+args.test_num]

    os.makedirs(args.out_dir, exist_ok=True)
    write_tsv(os.path.join(args.out_dir,"train.tsv"), train)
    write_tsv(os.path.join(args.out_dir,"val.tsv"),   val)
    write_tsv(os.path.join(args.out_dir,"test.tsv"),  test)

    display = [r for r in sampled if r["license"] in DISPLAY_LICENSE_ALLOW]
    write_tsv(os.path.join(args.out_dir,"display.tsv"), display[:min(200, len(display))])

    # stats
    def stat(name, rows):
        cnt=len(rows)
        b=Counter([r["bucket"] for r in rows])
        wf=Counter([r["workflow"] for r in rows])
        return f"{name}: {cnt} | buckets={dict(b)} | workflow={dict(wf)}"
    print(stat("TRAIN", train))
    print(stat("VAL",   val))
    print(stat("TEST",  test))
    print(f"[OK] manifests written to {args.out_dir}")

    if args.download:
        all_rows = train + val + test
        maybe_download(all_rows, args.repo, args.data_root, also_1k=args.also_1k)
        print("[OK] download finished.")

if __name__ == "__main__":
    main()
