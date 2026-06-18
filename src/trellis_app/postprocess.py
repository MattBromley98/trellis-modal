from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

METAL_KEYWORDS = [
    'metal', 'steel', 'iron', 'gold', 'silver', 'bronze', 'brass',
    'copper', 'aluminum', 'aluminium', 'tin', 'chrome', 'metallic',
    'alloy', 'titanium', 'platinum', 'lead', 'zinc', 'nickel',
]

NON_METAL_KEYWORDS = [
    'wood', 'wooden', 'fabric', 'cloth', 'cotton', 'silk', 'wool',
    'stone', 'rock', 'plastic', 'polymer', 'food', 'tree', 'plant',
    'flower', 'grass', 'dirt', 'sand', 'snow', 'water', 'glass',
    'ceramic', 'porcelain', 'paper', 'cardboard', 'leather', 'rubber',
    'concrete', 'brick', 'skin', 'fur', 'feather', 'bone', 'ivory',
    'clay', 'terracotta', 'foam', 'sponge', 'carpet', 'canvas',
    'linen', 'velvet', 'wax', 'resin', 'cork', 'bamboo', 'rattan',
    'straw', 'hay', 'soil', 'mud', 'ice', 'lava', 'crystal',
    'marble', 'limestone', 'sandstone', 'granite', 'slate', 'basalt',
    'obsidian',
]


def _classify_material(prompt: str) -> tuple[bool, bool]:
    prompt_lower = prompt.lower()
    is_metal = any(kw in prompt_lower for kw in METAL_KEYWORDS)
    is_non_metal = any(kw in prompt_lower for kw in NON_METAL_KEYWORDS)
    return is_metal, is_non_metal


def refine_pbr_materials(
    scene: Any,
    prompt: str = "",
    *,
    metallic_threshold: float = 0.12,
    metallic_median_blur: int = 3,
    metallic_factor: float = 0.3,
    roughness_factor: float = 1.0,
    verbose: bool = False,
) -> Any:
    if verbose:
        logger.info("Refining PBR materials for prompt: %s", prompt)

    try:
        from PIL import Image
        import trimesh
    except ImportError:
        logger.warning("trimesh/PIL not available, skipping PBR refinement")
        return scene

    is_metal, is_non_metal = _classify_material(prompt)

    if is_metal and not is_non_metal:
        metallic_threshold = 0.0
        metallic_factor = max(metallic_factor, 0.8)
        if verbose:
            logger.info("Prompt suggests metal; using lenient PBR settings")
    elif is_non_metal and not is_metal:
        metallic_threshold = max(metallic_threshold, 0.2)
        metallic_factor = min(metallic_factor, 0.05)
        if verbose:
            logger.info("Prompt suggests non-metal; using aggressive PBR settings")
    elif is_metal and is_non_metal:
        metallic_factor = max(metallic_factor, 0.5)
        if verbose:
            logger.info("Prompt ambiguous; using moderate PBR settings")

    if verbose:
        logger.info(
            "PBR params: threshold=%.2f median_blur=%d "
            "metallic_factor=%.2f roughness_factor=%.2f",
            metallic_threshold, metallic_median_blur,
            metallic_factor, roughness_factor,
        )

    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if not hasattr(geom.visual, 'material'):
            continue
        mat = geom.visual.material
        if not isinstance(mat, trimesh.visual.material.PBRMaterial):
            continue
        if mat.metallicRoughnessTexture is None:
            continue

        orm = np.array(mat.metallicRoughnessTexture.convert('RGB'))
        if orm.size == 0:
            continue

        roughness = orm[:, :, 1].astype(np.float32) / 255.0
        metallic = orm[:, :, 2].astype(np.float32) / 255.0

        if metallic_threshold > 0:
            metallic = np.where(metallic < metallic_threshold, 0.0, metallic)

        if metallic_median_blur > 0 and metallic_median_blur % 2 == 1:
            try:
                import cv2
                met_uint8 = (metallic * 255).astype(np.uint8)
                met_uint8 = cv2.medianBlur(met_uint8, metallic_median_blur)
                metallic = met_uint8.astype(np.float32) / 255.0
            except ImportError:
                pass

        orm[:, :, 1] = (roughness * 255).astype(np.uint8)
        orm[:, :, 2] = (metallic * 255).astype(np.uint8)
        mat.metallicRoughnessTexture = Image.fromarray(orm)
        mat.metallicFactor = metallic_factor
        mat.roughnessFactor = roughness_factor

    return scene
