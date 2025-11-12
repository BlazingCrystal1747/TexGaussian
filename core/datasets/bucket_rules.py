# -*- coding: utf-8 -*-
"""
bucket_rules.py

将自由文本 caption 归入“材质桶”(wood/metal/...)，供采样均衡与物理先验使用。
不依赖第三方库，规则为大小写不敏感的子串匹配。
"""

from typing import Dict, List, Optional

# 10 类材质桶的关键词（全部小写，便于与 caption.lower() 比对）
BUCKET_KWS: Dict[str, List[str]] = {
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

# 桶顺序（用于优先级）；最后追加 "misc" 作为兜底
BUCKET_NAMES: List[str] = [
    "wood", "metal", "fabric", "leather", "plastic",
    "ceramic", "glass", "stone", "paint", "composite", "misc"
]

# 未命中时的桶 ID
MISC_ID: int = 10


def caption_to_bucket(text: Optional[str]) -> int:
    """
    将自由文本 caption 归为材质桶，返回桶 ID（0..9）；都不匹配则返回 MISC_ID(10)。
    规则：大小写不敏感，子串匹配；按 BUCKET_NAMES 的顺序优先命中第一类（忽略 'misc'）。
    """
    if not text:
        return MISC_ID
    lowered = text.lower()
    for idx, name in enumerate(BUCKET_NAMES):
        if name == "misc":
            break
        kws = BUCKET_KWS.get(name, [])
        if any(kw in lowered for kw in kws):
            return idx
    return MISC_ID


__all__ = ["BUCKET_KWS", "BUCKET_NAMES", "MISC_ID", "caption_to_bucket"]
