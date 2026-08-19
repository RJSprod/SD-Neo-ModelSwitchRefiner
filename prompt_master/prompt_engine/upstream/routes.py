"""
routes.py — Claude Prompt LD
/cpld/* endpoints on the ComfyUI PromptServer.

  POST /cpld/generate   — SSE token stream from the local LLM
  POST /cpld/upload     — save a browser image into ComfyUI's input folder
  GET  /cpld/thumb      — jpeg preview of an input-folder image
  GET  /cpld/models     — scan GGUF models dir (managed backend)
  POST /cpld/backend    — write connection settings
  GET  /cpld/health     — is the configured LLM answering
  POST /cpld/free       — kill / evict the LLM (hand VRAM to LTX)
"""

import asyncio
import threading
import json
import os

from . import backend as be
from . import brain
from . import negative as neg
from .imaging import b64_to_pil, is_image_filename, jpeg_b64, open_rgb, style_hint


def _input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def resolve_input_image(rel_name):
    """Safe join under the input folder; images only."""
    if not rel_name or not is_image_filename(rel_name):
        return None
    base = os.path.normpath(_input_dir())
    full = os.path.normpath(os.path.join(base, str(rel_name).replace("/", os.sep)))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full if os.path.isfile(full) else None


def _prompt_estimate(messages) -> int:
    """Rough prompt size in tokens, for the ctx clamp only.

    The first version did `len(str(content)) // 3` over the whole message list.
    In i2v `content` is a LIST containing a base64 JPEG data URI, so str() of it
    was ~100k characters of base64 and the estimate came back around 30k tokens
    — which is how a 32k context clamped the reply to 1481 tokens. The image is
    not text and does not cost text tokens; a vision projector spends a fixed
    few hundred regardless of how long the base64 happens to be.

    So: count text only at ~4 chars per token (English), and add a flat
    allowance per image instead of measuring it.
    """
    chars, images = 0, 0
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, str):
            chars += len(c)
            continue
        for part in (c or []):
            if not isinstance(part, dict):
                chars += len(str(part))
            elif part.get("type") == "text":
                chars += len(str(part.get("text", "")))
            else:
                images += 1
    return chars // 4 + images * 1024


def _build_messages(body):
    mode = body.get("video_mode", "i2v")
    img_path = resolve_input_image(body.get("image_name", ""))
    pil = open_rgb(img_path) if img_path else b64_to_pil(body.get("image_b64", ""))
    # Medium cue for I2V: flat-color / filename → cartoon, so the script anchors
    # LTX in the still's world instead of inventing a photoreal Australian woman.
    sh = ""
    if (mode or "").lower() == "i2v" and pil is not None:
        sh = style_hint(pil, name=body.get("image_name", "") or "")
        if sh:
            print(f"[ClaudePromptLD] I2V style_hint={sh!r} "
                  f"image={body.get('image_name') or '(b64)'}")

    # send_vision is hoisted ABOVE build_system on purpose. The brief's i2v
    # opener changes completely depending on whether the still is actually on
    # the wire; computing it after the fact is how the system prompt ended up
    # insisting "Frame one is the attached image" while the user turn said the
    # opposite, and the system prompt won.
    send_vision = (mode or "").lower() == "i2v" and pil is not None and be.vision_supported()

    system = brain.build_system(
        mode=mode,
        pov=body.get("pov", "off"),
        accent=body.get("accent", "off"),
        accent_strength=body.get("accent_strength", "natural"),
        dialogue=body.get("dialogue", "some"),
        wardrobe=body.get("wardrobe", "off"),
        undress=bool(body.get("undress")),
        seed=int(body.get("seed", 0) or 0),
        intent=body.get("intent", ""),
        camera=body.get("camera", "off"),
        transition=body.get("transition", "off"),
        music=body.get("music", "off"),
        music_bg=bool(body.get("music_bg")),
        lexicon=body.get("lexicon", ""),
        fmt=body.get("fmt", "flowing"),
        fps=int(body.get("fps", 24) or 24),
        seconds=float(body.get("seconds", 12) or 12),
        style=body.get("style", "off"),
        style_hint=sh,
        has_image=send_vision,
    )

    user_text = brain.build_user(
        intent=body.get("intent", ""), mode=mode, has_image=send_vision,
        style_hint=sh,
    )
    if send_vision:
        content = [
            {"type": "image_url", "image_url": {"url": jpeg_b64(pil, max_side=768)}},
            {"type": "text", "text": user_text},
        ]
    else:
        content = user_text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ], send_vision


def register():
    try:
        from aiohttp import web
        from server import PromptServer
        inst = getattr(PromptServer, "instance", None)
        if inst is None:
            return
    except Exception as e:
        print(f"[ClaudePromptLD] routes skipped: {e}")
        return

    @inst.routes.post("/cpld/generate")
    async def generate(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)

        async def send(obj):
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))

        loop = asyncio.get_event_loop()
        # keep_warm only skips free *before* boot so re-rolls can reuse a live
        # server mid-panel. Post-generate free ALWAYS runs (VRAM back for LTX).
        keep_warm = bool(body.get("keep_warm"))
        free_msg = ""
        cleaned = ""
        had_error = False
        # Set when the LLM worker thread has actually exited. Freeing VRAM while
        # llama-server is still streaming just hands the memory straight back.
        worker_done = threading.Event()
        worker_done.set()          # nothing running yet
        worker_fut = None

        async def free_vram(label: str) -> str:
            """Stop the LLM, wait for it to actually stop, then unload + flush."""
            # 1) Tell any in-flight stream to quit and give the worker thread a
            #    moment to unwind. Without this the kill lands while the model
            #    is mid-token and the driver never returns the memory.
            if not worker_done.is_set():
                be.abort()
                try:
                    await loop.run_in_executor(None, worker_done.wait, 8.0)
                except Exception:
                    pass
                if not worker_done.is_set():
                    print(f"[ClaudePromptLD] {label}: worker still running after 8s "
                          "— killing anyway")
            try:
                await send({"type": "status", "msg": "freeing VRAM…"})
            except Exception:
                pass
            try:
                # free() = kill LLM process/model + flush_vram (same as /pfld/kill)
                msg = await loop.run_in_executor(
                    None, lambda: be.free(flush=True, light=False))
                print(f"[ClaudePromptLD] {label}: {msg}")
                try:
                    await send({"type": "status", "msg": f"VRAM free: {msg}"})
                except Exception:
                    pass
                return msg
            except Exception as e:
                print(f"[ClaudePromptLD] {label} failed: {e}")
                try:
                    await send({"type": "status", "msg": f"VRAM free failed: {e}"})
                except Exception:
                    pass
                return f"free failed: {e}"

        try:
            if not keep_warm:
                await free_vram("pre-generate free")

            await send({"type": "status", "msg": "booting LLM…"})
            state = await loop.run_in_executor(
                None, be.ensure, body.get("model_file", ""), body.get("mmproj_file", ""))
            print(f"[ClaudePromptLD] ensure: {state}")
            if state.startswith("ERR"):
                had_error = True
                await send({"type": "error", "msg": state})
                # fall through to finally free — do not return early without free
            else:
                await send({"type": "status", "msg": "writing…"})

                messages, vision = _build_messages(body)
                seed = body.get("seed")
                sec = float(body.get("seconds", 12) or 12)
                fmt = body.get("fmt", "flowing")
                mt = brain.max_tokens(sec, body.get("dialogue", "some"), fmt=fmt)
                # The budgets above scale with duration and now cover the FULL
                # clip, so a long shot asks for more tokens than it used to.
                # max_tokens cannot see the server, and llama-server silently
                # truncates when prompt + completion exceeds -c — which looks
                # exactly like the word-budget truncation this release fixes.
                # Clamp here, where CONN["ctx"] is visible, and say so.
                try:
                    ctx = int(be.CONN.get("ctx") or 8192)
                    est = _prompt_estimate(messages)
                    room = max(768, ctx - est - 256)
                    if mt > room:
                        print(f"[PromptMasterLD] max_tokens {mt} -> {room} "
                              f"(ctx={ctx}, prompt≈{est}) — raise ctx for long clips")
                        await send({"type": "status",
                                    "msg": f"token ceiling clamped to {room} (ctx {ctx})"})
                        mt = room
                except Exception as e:
                    print(f"[PromptMasterLD] ctx clamp skipped: {e}")
                print(f"[ClaudePromptLD] generate max_tokens={mt} seconds={sec} "
                      f"fmt={fmt!r} vision={vision} backend={be.CONN.get('backend')}")

                def run(msgs):
                    return list(be.chat_stream(msgs, seed=seed, max_tokens=mt))

                # Vision can return empty/unreadable SSE or 400 → retry text-only.
                try:
                    worker_done.clear()
                    q = asyncio.Queue()

                    def worker():
                        try:
                            for piece in be.chat_stream(messages, seed=seed, max_tokens=mt):
                                loop.call_soon_threadsafe(q.put_nowait, ("tok", piece))
                            loop.call_soon_threadsafe(q.put_nowait, ("done", None))
                        except Exception as e:
                            loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))
                        finally:
                            worker_done.set()

                    worker_fut = loop.run_in_executor(None, worker)
                    text = ""
                    while True:
                        kind, val = await q.get()
                        if kind == "tok":
                            text += val
                            await send({"type": "token", "text": val})
                        elif kind == "err":
                            raise RuntimeError(val)
                        else:
                            break
                except Exception as e:
                    if vision:
                        await send({
                            "type": "status",
                            "msg": "vision path empty/failed — retrying text-only "
                                   "(no image; scene may be invented from intent)",
                        })
                        print(f"[ClaudePromptLD] vision path failed ({e}); retry text-only "
                              f"(image stripped — not grounded on the still)")
                        messages, _ = _build_messages({**body, "image_b64": "", "image_name": ""})
                        text = "".join(await loop.run_in_executor(None, run, messages))
                        await send({"type": "token", "text": text})
                    else:
                        raise RuntimeError(str(e))

                cleaned = brain.clean_script(text)
                if not cleaned.strip():
                    raise RuntimeError(
                        f"LLM produced empty script. See {be.log_path()}\n{be.tail_log(25)}"
                    )

                # Smart negative: a second, cheap pass over the FINISHED script
                # while the model is still resident. Must happen before the free
                # or it would need a whole extra boot. Never fatal — a failed
                # pass returns "" and the static banks carry the negative.
                auto_neg = ""
                if bool(body.get("smart_negative")):
                    await send({"type": "status", "msg": "negative pass…"})
                    auto_neg = await loop.run_in_executor(
                        None, lambda: neg.run_auto(cleaned, be.chat_stream, seed=seed))
                    print(f"[PromptMasterLD] auto-negative: {auto_neg or '(none)'}")
                    await send({"type": "negative", "text": auto_neg})

                # Free VRAM *before* closing the stream so the UI always sees
                # it, then attach free_msg on the done event.
                #
                # keep_warm ("fast re-roll") used to skip only the free BEFORE
                # boot, while THIS line freed unconditionally after every
                # generate — so the server was always killed on the way out and
                # the next re-roll paid a full reload. The rest of the promise
                # was already built: node.run() does a full free at queue
                # hand-off, which is exactly "stays loaded until you press Run".
                #
                # free_msg must stay TRUTHY here. The finally block below frees
                # again when `not free_msg`, so leaving it empty would undo this
                # one line later and look identical to the original bug.
                if keep_warm:
                    free_msg = "kept warm — frees when you queue the render"
                    print("[ClaudePromptLD] keep_warm: LLM left resident for "
                          "the next re-roll")
                else:
                    free_msg = await free_vram("post-generate free")
                await send({"type": "done", "text": cleaned, "free": free_msg,
                            "negative": auto_neg})
                print(f"[ClaudePromptLD] done — {len(cleaned.split())} words; free={free_msg}")
        except (asyncio.CancelledError, ConnectionResetError):
            # Client hit Cancel / closed the tab mid-generation.
            print("[ClaudePromptLD] stream aborted — stopping LLM")
            be.abort()
            had_error = True
        except Exception as e:
            had_error = True
            try:
                await send({"type": "error", "msg": str(e)})
            except Exception:
                pass
        finally:
            # Always free again on error/abort paths (success already freed above;
            # second kill is cheap/idempotent). Covers keep_warm re-rolls too.
            if had_error or not free_msg or not worker_done.is_set():
                try:
                    free_msg = await free_vram("finally free")
                except Exception as e:
                    print(f"[ClaudePromptLD] finally free failed: {e}")
            be.clear_abort()
            try:
                await resp.write_eof()
            except Exception:
                pass
        return resp

    @inst.routes.post("/cpld/upload")
    async def upload(request):
        import base64
        try:
            body = await request.json()
            name = os.path.basename((body.get("name") or "upload.png").strip()) or "upload.png"
            data = (body.get("b64") or "")
            if "," in data:
                data = data.split(",", 1)[1]
            if not data:
                return web.json_response({"ok": False, "error": "no data"}, status=400)
            if not is_image_filename(name):
                name += ".png"
            base = _input_dir()
            os.makedirs(base, exist_ok=True)
            stem, ext = os.path.splitext(name)
            path, i = os.path.join(base, name), 1
            while os.path.exists(path):
                path = os.path.join(base, f"{stem}_{i}{ext}")
                i += 1
            with open(path, "wb") as f:
                f.write(base64.b64decode(data))
            pil = open_rgb(path)
            return web.json_response({
                "ok": True, "name": os.path.basename(path),
                "width": pil.width if pil else 0, "height": pil.height if pil else 0,
            })
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @inst.routes.get("/cpld/thumb")
    async def thumb(request):
        path = resolve_input_image(request.rel_url.query.get("name", ""))
        pil = open_rgb(path) if path else None
        if pil is None:
            return web.Response(status=404)
        b64 = jpeg_b64(pil, max_side=512)
        import base64 as _b
        raw = _b.b64decode(b64.split(",", 1)[1])
        return web.Response(body=raw, content_type="image/jpeg")

    @inst.routes.get("/cpld/imginfo")
    async def imginfo(request):
        path = resolve_input_image(request.rel_url.query.get("name", ""))
        pil = open_rgb(path) if path else None
        if pil is None:
            return web.json_response({"ok": False}, status=404)
        return web.json_response({"ok": True, "width": pil.width, "height": pil.height})

    @inst.routes.get("/cpld/models")
    async def models(request):
        gguf, mmproj = be.scan_models()
        return web.json_response({
            "ok": True, "models": gguf, "mmproj": mmproj,
            "models_dir": be.CONN["models_dir"],
        })

    @inst.routes.post("/cpld/backend")
    async def set_backend(request):
        try:
            body = await request.json()
            be.set_conn(
                backend=body.get("backend"),
                server_url=body.get("server_url"),
                remote_model=body.get("remote_model"),
                models_dir=body.get("models_dir"),
                llama_exe=body.get("llama_exe"),
                ctx=body.get("ctx"),
            )
            # set_conn already persists to cpld_conn.json
            return web.json_response({
                "ok": True,
                "conn": dict(be.CONN),
                "saved": be.conn_path(),
            })
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @inst.routes.get("/cpld/health")
    async def health(request):
        return web.json_response({
            "ok": True, "healthy": be.healthy(), "conn": dict(be.CONN),
        })

    @inst.routes.post("/cpld/free")
    async def free(request):
        """Kill/evict the LLM + flush CUDA.

        Optional JSON: {"fast": true} → light Comfy flush only (snappy Queue path).
        Default: full free (unload LLM + unload_all_models + empty_cache).
        """
        light = False
        try:
            body = await request.json()
            if isinstance(body, dict) and (
                body.get("fast") is True or body.get("light") is True
            ):
                light = True
        except Exception:
            pass
        msg = be.free(flush=True, light=light)
        return web.json_response({"ok": True, "msg": msg, "fast": light})

    # ── LoRA Loader LD ────────────────────────────────────────────────────
    # Ported from PromptForgeLD so this pack has no cross-node dependency.
    def _is_audio_key(k):
        return "audio" in k.lower()

    @inst.routes.get("/cpld/lora_list")
    async def lora_list(request):
        try:
            import folder_paths
            loras = folder_paths.get_filename_list("loras")
            return web.json_response({"loras": ["None"] + list(loras)})
        except Exception as e:
            return web.json_response({"loras": ["None"], "error": str(e)})

    @inst.routes.get("/cpld/lora_keycounts")
    async def lora_keycounts(request):
        """Video/audio tensor counts for a LoRA — drives the V:/A: readout."""
        try:
            import folder_paths
            import comfy.utils
            lora_name = request.rel_url.query.get("lora", "")
            if not lora_name:
                return web.json_response({"v": 0, "a": 0})
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.isfile(lora_path):
                return web.json_response({"v": 0, "a": 0})
            try:
                import safetensors
                with safetensors.safe_open(lora_path, framework="pt",
                                           device="cpu") as f:
                    keys = list(f.keys())
            except Exception:
                try:
                    weights = comfy.utils.load_torch_file(lora_path,
                                                          safe_load=True)
                    keys = list(weights.keys())
                except Exception:
                    return web.json_response({"v": -1, "a": -1})
            v = sum(1 for k in keys if not _is_audio_key(k))
            a = sum(1 for k in keys if _is_audio_key(k))
            return web.json_response({"v": v, "a": a})
        except Exception as e:
            return web.json_response({"v": -1, "a": -1, "error": str(e)})

    print("[ClaudePromptLD] routes: /cpld/generate /cpld/upload /cpld/thumb "
          "/cpld/imginfo /cpld/models /cpld/backend /cpld/health /cpld/free "
          "/cpld/lora_list /cpld/lora_keycounts")
