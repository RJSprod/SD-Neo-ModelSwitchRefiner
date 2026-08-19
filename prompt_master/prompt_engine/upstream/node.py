"""
node.py — Claude Prompt LD
Two nodes:
  Claude Prompt LD      — panel-driven writer; commits a CPLD_PACK to the graph
  Claude Prompt Unpack  — pack → image · reference · positive · negative · width · height · fps · frames

Pack carries fps and the snapped 8n+1 frame count so the sampler side of the
graph gets real timing, not just text.
"""

from . import backend as be
from .accents import STRENGTHS, ACCENT_KEYS
from .brain import build_negative, frame_count
from .styles import STYLE_KEYS
from .cinematics import CAMERA_KEYS, TRANSITION_KEYS
from .music import MUSIC_KEYS
from .shotscript import FORMATS
from .imaging import b64_to_pil, black, open_rgb, pil_to_tensor, resize_fit, snap32
from .routes import register, resolve_input_image

try:
    register()
except Exception as e:
    print(f"[ClaudePromptLD] route registration skipped: {e}")

CPLD_PACK = "CPLD_PACK"


def make_pack(image, positive, negative, width, height, fps, frames, reference=None):
    return {
        "image": image,
        "reference": reference,  # T2V optional guidance image; None in I2V
        "positive": (positive or "").strip(),
        "negative": (negative or "").strip(),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "frames": int(frames),
    }


def unpack(pack):
    if not isinstance(pack, dict):
        raise ValueError("[Claude Prompt Unpack] Expected CPLD_PACK — wire Claude Prompt LD's pack output here.")
    for k in ("image", "positive", "negative", "width", "height", "fps", "frames"):
        if k not in pack:
            raise ValueError(f"[Claude Prompt Unpack] pack missing '{k}'")
    # reference is optional (older packs won't have it); fall back to the frame image
    reference = pack.get("reference")
    if reference is None:
        reference = pack["image"]
    return (pack["image"], reference, pack["positive"], pack["negative"],
            int(pack["width"]), int(pack["height"]), int(pack["fps"]), int(pack["frames"]))


class ClaudePromptLD:
    """LTX 2.3 shot writer — director's-brief brain, panel UI, pack output."""

    @classmethod
    def INPUT_TYPES(cls):
        gguf, mmproj = be.scan_models()
        gguf = ["None"] + gguf
        mmproj = ["None (text-only)"] + mmproj
        return {
            "required": {
                "model_file": (gguf, {"default": gguf[1] if len(gguf) > 1 else "None"}),
                "mmproj_file": (mmproj, {"default": mmproj[1] if len(mmproj) > 1 else mmproj[0]}),
                "video_mode": (["i2v", "t2v"], {"default": "i2v"}),
                "pov": (["off", "male", "female"], {"default": "off"}),
                "accent": (ACCENT_KEYS, {"default": "off"}),
                "dialogue": ("INT", {"default": 20, "min": 0, "max": 100}),
                "wardrobe": (["auto", "off", "her", "him"], {"default": "auto"}),
                "undress": ("BOOLEAN", {"default": False}),
                "camera": (CAMERA_KEYS, {"default": "off"}),
                "transition": (TRANSITION_KEYS, {"default": "off"}),
                "music": (["auto"] + MUSIC_KEYS, {"default": "off"}),
                "music_bg": ("BOOLEAN", {"default": False}),
                "lexicon": ("STRING", {"default": "", "multiline": True}),
                "fmt": (FORMATS, {"default": "flowing"}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seconds": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                # control_after_generate TRUE so ComfyUI advances the seed on
                # every Queue. It was False here AND force-pinned to "fixed" in
                # the JS, so the feature was suppressed twice over: every
                # re-roll rebuilt an identical brief and the model returned its
                # highest-probability answer to the same question. The panel's
                # ▶ Generate bypasses the queue entirely, so it advances the
                # seed itself — see advanceSeed() in claude_prompt.js.
                "seed": ("INT", {"default": 7, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                "intent": ("STRING", {"multiline": True, "default": ""}),
                "script": ("STRING", {"multiline": True, "default": ""}),
                "negative_extra": ("STRING", {"multiline": True, "default": ""}),
                "image_name": ("STRING", {"default": ""}),
                "image_b64": ("STRING", {"default": ""}),
                "out_w": ("INT", {"default": 704, "min": 64, "max": 8192}),
                "out_h": ("INT", {"default": 1216, "min": 64, "max": 8192}),
                "fit": (["crop", "pad", "stretch"], {"default": "crop"}),
                # APPENDED LAST, DELIBERATELY. ComfyUI serializes
                # widgets_values as a positional ARRAY, so inserting a
                # widget mid-list shifts every later index and an old
                # workflow's intent/script lands in the wrong field.
                # Any future widget goes below this line, never above it.
                "style": (STYLE_KEYS, {"default": "off"}),
                # ↓ appended after style, same rule as above.
                "smart_negative": ("BOOLEAN", {"default": False}),
                "negative_auto": ("STRING", {"default": "", "multiline": True}),
                "accent_strength": (list(STRENGTHS), {"default": "natural"}),
            },
            "optional": {
                "fps_in": ("INT", {"default": 24, "min": 8, "max": 60, "forceInput": True}),
                "seconds_in": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "step": 0.5, "forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CPLD_PACK,)
    RETURN_NAMES = ("pack",)
    FUNCTION = "run"
    CATEGORY = "LD/PromptMaster"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kw):
        return True

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def run(self, model_file, mmproj_file, video_mode, pov, accent, dialogue, wardrobe, undress, camera, transition, music, music_bg, lexicon, fmt, fps, seconds,
            seed, intent, script, negative_extra, image_name, image_b64,
            out_w, out_h, fit, style="off", smart_negative=False,
            negative_auto="", accent_strength="natural", fps_in=None,
            seconds_in=None, unique_id=None):
        # ComfyUI can hand a combo widget its int index instead of the label —
        # coerce every string-typed input before any .lower()/.strip() touches it.
        def _s(v, default="", choices=None):
            if isinstance(v, str):
                return v
            if v is None:
                return default
            if choices and isinstance(v, int) and 0 <= v < len(choices):
                return choices[v]
            return str(v)
        video_mode = _s(video_mode, "i2v", ["i2v", "t2v"])
        camera = _s(camera, "off", CAMERA_KEYS)
        transition = _s(transition, "off", TRANSITION_KEYS)
        music = _s(music, "off", MUSIC_KEYS)
        fmt = _s(fmt, "flowing", FORMATS)
        pov = _s(pov, "off", ["off", "male", "female"])
        accent = _s(accent, "off", ACCENT_KEYS)
        # dialogue is now a 0-100 dial; legacy string values from old
        # workflows are mapped inside brain.talk_pct — pass through as-is.
        wardrobe = _s(wardrobe, "auto", ["auto", "off", "her", "him"])
        fit = _s(fit, "crop", ["crop", "pad", "stretch"])
        style = _s(style, "off", STYLE_KEYS)
        negative_auto = _s(negative_auto)
        intent = _s(intent)
        script = _s(script)
        negative_extra = _s(negative_extra)
        image_name = _s(image_name)
        image_b64 = _s(image_b64)

        # Wired timing overrides the panel — the workflow owns the clock.
        print(f"[ClaudePromptLD] timing inputs: fps_in={fps_in!r} seconds_in={seconds_in!r} "
              f"| panel fps={fps} seconds={seconds}")
        if fps_in is not None:
            fps = int(round(float(fps_in)))
        if seconds_in is not None:
            seconds = float(seconds_in)
        print(f"[ClaudePromptLD] timing resolved: fps={fps} seconds={seconds}")
        # Queue/Run means LTX is about to want the GPU. This is the ONE moment
        # that matters most: every click of Generate/Queue must hand LTX a clean
        # card. A light flush (soft_empty_cache only) was leaving the previous
        # run's models AND cached latents resident, so VRAM crept up every
        # queue until it overflowed. Full free: abort any in-flight LLM stream,
        # kill/evict the model, unload_all_models, purge the node-output cache,
        # empty_cache.
        try:
            be.abort()          # stop a panel generation still streaming
            msg = be.free(flush=True, light=False)
            print(f"[ClaudePromptLD] queue hand-off (full free): {msg}")
        except Exception as e:
            print(f"[ClaudePromptLD] LLM free skipped: {e}")
        finally:
            try:
                be.clear_abort()
            except Exception:
                pass

        w, h = snap32(out_w), snap32(out_h)
        positive = (script or "").strip()
        if not positive:
            print("[ClaudePromptLD] WARNING: script is empty — use ▶ Generate in the panel "
                  "(or type a prompt) before queueing.")

        # The auto terms were written against THIS script by the panel pass and
        # stored in their own widget, so Queue reproduces exactly the negative
        # the user saw. Unticking the box drops them without clearing them.
        auto_terms = negative_auto if bool(smart_negative) else ""
        negative = build_negative(pov=pov, dialogue=dialogue, undress=undress,
                                  transition=transition, intent=intent, extra=negative_extra,
                                  camera=camera, fmt=fmt, style=style,
                                  mode=video_mode, auto=auto_terms)

        ref_image = None
        if str(video_mode).lower() == "i2v":
            pil = open_rgb(resolve_input_image(image_name) or "")
            if pil is None:
                pil = b64_to_pil(image_b64)
            if pil is None:
                raise ValueError("[ClaudePromptLD] I2V needs an image — load one in the panel.")
            image = pil_to_tensor(resize_fit(pil, w, h, fit))
        else:
            # T2V: pack image is a black frame at target size. Any loaded image
            # rides as a SEPARATE reference (not frame one) on the reference slot.
            image = black(w, h)
            rpil = open_rgb(resolve_input_image(image_name) or "")
            if rpil is None:
                rpil = b64_to_pil(image_b64)
            if rpil is not None:
                ref_image = pil_to_tensor(resize_fit(rpil, w, h, fit))

        frames = frame_count(fps, seconds)
        print(f"[ClaudePromptLD] {str(video_mode).upper()} {w}×{h} @ {fps}fps "
              f"×{seconds:g}s → {frames} frames | pov={pov} accent={accent} dlg={dialogue} | "
              f"{len(positive)}c pos / {len(negative)}c neg")
        return (make_pack(image, positive, negative, w, h, int(fps), frames, reference=ref_image),)



# ── ComfyUI interrupt hook ────────────────────────────────────────────────
# Cancelling a queued run mid-sample leaves models + partial latents resident.
# Comfy raises InterruptProcessingException from throw_exception_if_processing_interrupted;
# we wrap it so our flush runs on every cancel, from anywhere in the graph.
def _install_interrupt_hook():
    try:
        import comfy.model_management as mm
    except Exception as e:
        print(f"[ClaudePromptLD] interrupt hook skipped: {e}")
        return
    if getattr(mm, "_cpld_interrupt_hooked", False):
        return
    orig = mm.throw_exception_if_processing_interrupted

    def wrapped(*a, **kw):
        try:
            return orig(*a, **kw)
        except Exception:
            # Interrupted. Stop the writer LLM and give the card back.
            try:
                be.abort()
            except Exception:
                pass
            try:
                from .vram import flush_vram
                flush_vram("ClaudePromptLD/interrupt", light=False)
            except Exception as e:
                print(f"[ClaudePromptLD] interrupt flush failed: {e}")
            try:
                be.clear_abort()
            except Exception:
                pass
            raise

    mm.throw_exception_if_processing_interrupted = wrapped
    mm._cpld_interrupt_hooked = True
    print("[ClaudePromptLD] interrupt hook installed — cancel now frees VRAM.")


try:
    _install_interrupt_hook()
except Exception as e:
    print(f"[ClaudePromptLD] interrupt hook install failed: {e}")


class ClaudePromptUnpack:
    """Split the CPLD pack for the sampler side of the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pack": (CPLD_PACK,)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "reference", "positive", "negative", "width", "height", "fps", "frames")
    FUNCTION = "run"
    CATEGORY = "LD/PromptMaster"

    def run(self, pack):
        return unpack(pack)


NODE_CLASS_MAPPINGS = {
    "ClaudePromptLD": ClaudePromptLD,
    "ClaudePromptUnpack": ClaudePromptUnpack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudePromptLD": "🎬 Prompt Master - LD",
    "ClaudePromptUnpack": "📦 Prompt Unpack - LD",
}
