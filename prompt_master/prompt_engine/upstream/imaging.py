"""
imaging.py — Claude Prompt LD
Tensor plumbing plus a small honest resizer with three fit modes:

  crop     — fill the target, center-crop the overflow (default; no distortion)
  pad      — letterbox onto black, whole image preserved
  stretch  — exact target, aspect be damned

Dimensions snap to multiples of 32 (LTX latent grid).
"""

import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def snap32(v: int) -> int:
    return max(64, int(round(int(v) / 32)) * 32)


def is_image_filename(name: str) -> bool:
    return os.path.splitext(name or "")[1].lower() in IMAGE_EXTS


def open_rgb(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def b64_to_pil(b64: str):
    if not b64:
        return None
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


FIT_MODES = ["crop", "pad", "stretch"]


def resize_fit(pil, w: int, h: int, mode="crop"):
    w, h = snap32(w), snap32(h)
    sw, sh = pil.size
    if isinstance(mode, int) and 0 <= mode < len(FIT_MODES):
        mode = FIT_MODES[mode]  # Comfy handed the combo's index, not its label
    mode = str(mode or "crop").lower()
    if mode not in FIT_MODES:
        mode = "crop"

    if mode == "stretch" or (sw, sh) == (w, h):
        return pil.resize((w, h), Image.Resampling.LANCZOS)

    if mode == "pad":
        scale = min(w / sw, h / sh)
        nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
        scaled = pil.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
        return canvas

    # crop (cover)
    scale = max(w / sw, h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    scaled = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return scaled.crop((left, top, left + w, top + h))


def jpeg_b64(pil, max_side=768, quality=82) -> str:
    """Compact data-URL for previews and vision messages."""
    sw, sh = pil.size
    if max(sw, sh) > max_side:
        if sw >= sh:
            pil = pil.resize((max_side, max(1, round(sh * max_side / sw))), Image.Resampling.LANCZOS)
        else:
            pil = pil.resize((max(1, round(sw * max_side / sh)), max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Filename tokens that almost always mean drawn media (not a live photo).
_STYLE_NAME_CUES = (
    "cartoon", "comic", "anime", "manga", "toon", "cel", "illustration",
    "illustrated", "drawn", "drawing", "paint", "painted", "vector",
    "sticker", "clipart", "chibi", "webtoon", "manhwa", "2d", "artstation",
)


def style_hint_from_name(name: str) -> str:
    """Soft medium cue from the image filename."""
    n = (name or "").lower()
    if not n:
        return ""
    if any(c in n for c in _STYLE_NAME_CUES):
        return "cartoon"
    return ""


def style_hint(pil, name: str = "") -> str:
    """Conservative medium label for I2V: 'cartoon' or ''.

    Prefer under-firing: the vision model + MEDIUM FIRST law still see the
    still. We only inject a hard CARTOON block when confidence is high
    (very flat cel palette, or filename cue + reasonably flat image).
    Aggressive heuristics false-fire on glossy photos.
    """
    from_name = style_hint_from_name(name)
    if pil is None:
        return from_name
    try:
        small = pil.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
        try:
            quant = small.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        except Exception:
            quant = small.quantize(colors=16)
        qarr = np.asarray(quant.convert("RGB"), dtype=np.float32)
        mse = float(((arr - qarr) ** 2).mean())
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        sat = float(((mx - mn) / (mx + 1e-3)).mean())
        # Very flat cel / vector / posterized art only
        if mse < 70 and sat > 0.12:
            return "cartoon"
        # Filename said cartoon and image is not a high-detail photo
        if from_name == "cartoon" and mse < 450:
            return "cartoon"
        return ""
    except Exception:
        return from_name
