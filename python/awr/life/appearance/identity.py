"""Visible synthetic identity, shared with task generation.

Plates are ordinary textured triangles on the actor, so they participate in RGB,
depth, instance segmentation, occlusion and the same rigid body transform.
"""

import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def texture_key(appearance):
    return hashlib.sha256(json.dumps(appearance, sort_keys=True).encode()).hexdigest()


PALETTE = (
    ("橄榄色", (100, 110, 60)),
    ("藏青色", (30, 40, 90)),
    ("酒红色", (120, 30, 50)),
    ("灰色", (128, 128, 130)),
    ("白色", (235, 235, 230)),
    ("棕色", (110, 75, 45)),
    ("绿色", (40, 110, 60)),
    ("浅蓝色", (130, 170, 220)),
)
TROUSERS = (("卡其色", (195, 180, 140)), ("黑色", (25, 25, 28)), ("深蓝色", (40, 50, 90)), ("灰色", (120, 120, 125)))


def clothing(ordinal):
    top, top_rgb = PALETTE[ordinal % len(PALETTE)]
    bottom, bottom_rgb = TROUSERS[(ordinal // len(PALETTE)) % len(TROUSERS)]
    return dict(top_name=top, top_rgb=list(top_rgb), bottom_name=bottom, bottom_rgb=list(bottom_rgb), age="adult")


def neutral_clothing_atlas(image):
    """Prepare the appearance-independent outfit substrate; source is untouched."""
    # MakeHuman compiler stores skin, outfit, shoes, hair, eyes and brows in
    # separate 1024 px tiles. Unknown layouts are rejected.
    if image.size != (6144, 1024):
        raise ValueError("Clothing recolour requires the verified six-tile human atlas")
    image = image.copy()
    tile = np.array(image.crop((1024, 0, 2048, 1024)))
    light = tile[..., :3].mean(-1) / 255
    # Preserve the authored weave, seams and folds without retaining a competing
    # hue. 235 is a light neutral substrate, multiplied by the garment colour.
    grey = np.clip(190 + 65 * light, 0, 255).astype("uint8")
    tile[..., :3] = grey[..., None]
    image.paste(Image.fromarray(tile), (1024, 0))
    return image


def clothing_vertices(data, appearance):
    """Per-person garment colours; immutable source geometry/materials are shared."""
    result = dict(data)
    c = data["c"].copy()
    tri_uv = data["uv"].reshape(-1, 3, 2).mean(1)[:, 0]
    outfit = (tri_uv >= 1 / 6) & (tri_uv < 2 / 6)
    z = data["p"].reshape(-1, 3, 3).mean(1)[:, 2]
    upper = z > data["p"][:, 2].max() * 0.51
    for mask, key in ((outfit & upper, "top_rgb"), (outfit & ~upper, "bottom_rgb")):
        c[np.repeat(mask, 3)] = np.asarray(appearance[key]) / 255
    result["c"] = c
    return result


def apply_clothing(data, image, appearance):
    """Uncached compatibility path with the same authored atlas and vertex colours."""
    return clothing_vertices(data, appearance), neutral_clothing_atlas(image)


def plate_text(ordinal):
    if not 0 <= ordinal <= 9999:
        raise ValueError("Synthetic plate ordinal outside unique range")
    return f"SIM {ordinal:04d}"


def plate_image(text):
    if not re.fullmatch(r"SIM [0-9]{4}", text):
        raise ValueError("Only explicit synthetic test plates are admitted")
    im = Image.new("RGBA", (1040, 220), (240, 240, 228, 255))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((2, 2, 1037, 217), radius=12, outline=(25, 25, 25), width=9)
    d.rectangle((12, 12, 105, 207), fill=(28, 64, 135))
    # Pillow's packaged font is pinned with the runtime, avoiding host font drift.
    font = ImageFont.load_default(size=154)
    box = d.textbbox((0, 0), text, font=font)
    text_image = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8))
    ImageDraw.Draw(text_image).text((4 - box[0], 4 - box[1]), text, font=font, fill=(12, 12, 12))
    text_image = text_image.resize((866, 150), Image.Resampling.LANCZOS)
    im.alpha_composite(text_image, (130, 35))
    return im


def add_plates(data, image, text):
    """Texture the authored registration-plate mesh, at its real size/mount."""
    p = data["p"]
    if "plate_ids" not in data or not np.any(data["plate_ids"]):
        raise ValueError("Vehicle has no verified registration plate mesh")
    plate = plate_image(text)
    w, h = image.size
    atlas = Image.new("RGBA", (max(w, plate.width), h + plate.height))
    atlas.paste(image, (0, 0))
    atlas.paste(plate, (0, h))
    uv = data["uv"].copy()
    uv[:, 0] *= w / atlas.width
    uv[:, 1] = (uv[:, 1] * h + plate.height) / atlas.height
    colours = data["c"].copy()
    for side in (1, -1):
        ids = data["plate_ids"] == side
        if not ids.any():
            continue
        y, z = p[ids, 1], p[ids, 2]
        if np.ptp(y) < 0.2 or np.ptp(z) < 0.04:
            raise ValueError("Registration geometry is not an upright readable plate")
        u = (y - y.min()) / np.ptp(y) if side > 0 else (y.max() - y) / np.ptp(y)
        v = (z - z.min()) / np.ptp(z)
        uv[ids] = np.column_stack([u * plate.width / atlas.width, v * plate.height / atlas.height])
        colours[ids] = 1.0
    result = dict(data)
    result.update(c=colours, uv=uv)
    return result, atlas


def apply_paint(data, image, appearance):
    if "paint_mask" not in data or not data["paint_mask"].any():
        raise ValueError("Vehicle has no verified paint material mask")
    result = dict(data)
    c = data["c"].copy()
    c[data["paint_mask"]] = np.asarray(appearance["body_rgb"]) / 255
    w, h = image.size
    atlas = Image.new("RGBA", (w, h + 8), (255, 255, 255, 255))
    atlas.paste(image, (0, 0))
    uv = data["uv"].copy()
    uv[:, 1] = (uv[:, 1] * h + 8) / (h + 8)
    uv[data["paint_mask"]] = [0.5, 4 / (h + 8)]
    result.update(c=c, uv=uv)
    return result, atlas
