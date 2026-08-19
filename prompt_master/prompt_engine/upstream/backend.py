"""
backend.py — Claude Prompt LD
One shared LLM connection for the node. Three backends, one chat path:

  llama.cpp (managed)  — we spawn llama-server.exe with the chosen GGUF
                          (+ optional mmproj for i2v vision) and own its life
  LM Studio            — connect-only, OpenAI-compatible at :1234
  Ollama               — connect-only, OpenAI-compatible at :11434

All three speak /v1/chat/completions with stream=true, so generation code
never branches on backend.

Managed llama logs to llama_server.log next to this file AND to the Comfy
console so empty 0-word runs are debuggable.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

MANAGED = "llama.cpp (managed)"
LMSTUDIO = "LM Studio"
OLLAMA = "Ollama"
BACKENDS = [MANAGED, LMSTUDIO, OLLAMA]

DEFAULT_PORTS = {
    MANAGED: "http://127.0.0.1:8080",
    LMSTUDIO: "http://127.0.0.1:1234",
    OLLAMA: "http://127.0.0.1:11434",
}

CONN = {
    "backend": MANAGED,
    "server_url": DEFAULT_PORTS[MANAGED],
    "remote_model": "local",
    "models_dir": os.environ.get("CPLD_MODELS_DIR") or r"C:\models",
    "llama_exe": os.environ.get("CPLD_LLAMA_EXE") or r"C:\llama\llama-server.exe",
    "ctx": 8192,
}

_proc = None
_loaded = ("", "")
_log_fh = None

# Keys we persist across Comfy restarts (JSON next to this file).
_CONN_SAVE_KEYS = ("backend", "server_url", "remote_model", "models_dir", "llama_exe", "ctx")


def log_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_server.log")


def conn_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpld_conn.json")


def _log(msg: str) -> None:
    line = f"[ClaudePromptLD] {msg}"
    print(line, flush=True)
    try:
        with open(log_path(), "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def tail_log(n: int = 50) -> str:
    path = log_path()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        chunk = "".join(lines[-n:]).strip()
        return chunk or "(log empty)"
    except FileNotFoundError:
        return "(no llama_server.log yet)"
    except Exception as e:
        return f"(could not read log: {e})"


def save_conn() -> str:
    """Write CONN to disk so settings survive Comfy restarts."""
    path = conn_path()
    data = {k: CONN.get(k) for k in _CONN_SAVE_KEYS}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _log(f"saved connection settings → {path}")
        return path
    except Exception as e:
        _log(f"save_conn failed: {e}")
        raise


def load_conn() -> bool:
    """Load CONN from disk if present. Returns True if a file was applied."""
    path = conn_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        for k in _CONN_SAVE_KEYS:
            if k in data and data[k] is not None:
                if k == "ctx":
                    try:
                        CONN[k] = int(data[k])
                    except (TypeError, ValueError):
                        pass
                else:
                    CONN[k] = data[k]
        # env overrides still win for paths if set
        if os.environ.get("CPLD_MODELS_DIR"):
            CONN["models_dir"] = os.environ["CPLD_MODELS_DIR"]
        if os.environ.get("CPLD_LLAMA_EXE"):
            CONN["llama_exe"] = os.environ["CPLD_LLAMA_EXE"]
        _log(f"loaded connection settings ← {path}  backend={CONN.get('backend')!r} "
             f"url={CONN.get('server_url')!r} model={CONN.get('remote_model')!r}")
        return True
    except Exception as e:
        _log(f"load_conn failed: {e}")
        return False


def set_conn(**kw):
    for k, v in kw.items():
        if k in CONN and v is not None:
            if k == "ctx":
                try:
                    CONN[k] = int(v)
                except (TypeError, ValueError):
                    continue
            else:
                CONN[k] = v
    # Always persist when the panel updates connection settings.
    try:
        save_conn()
    except Exception:
        pass


# Restore last saved backend / URL / model id on import (every Comfy start).
load_conn()


def url() -> str:
    return (CONN["server_url"] or DEFAULT_PORTS.get(CONN["backend"], DEFAULT_PORTS[MANAGED])).rstrip("/")


def is_managed() -> bool:
    return CONN["backend"] == MANAGED


def _normalize_mmproj(mmproj_file: str) -> str:
    m = (mmproj_file or "").strip()
    if not m or m.lower().startswith("none"):
        return ""
    return m


def scan_models():
    gguf, mmproj = [], []
    d = CONN["models_dir"]
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            fl = f.lower()
            if fl.endswith(".gguf"):
                (mmproj if ("mmproj" in fl or "clip" in fl) else gguf).append(f)
    return gguf, mmproj


def healthy(timeout=2.0) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(url() + path, timeout=timeout) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def ensure(model_file: str = "", mmproj_file: str = "") -> str:
    global _proc, _loaded, _log_fh
    if not is_managed():
        ok = healthy()
        _log(f"connect-only {CONN['backend']} @ {url()} → {'OK' if ok else 'DOWN'}")
        return "OK connected" if ok else f"ERR no server at {url()} — start {CONN['backend']} first"

    mmproj_file = _normalize_mmproj(mmproj_file)
    want = (model_file or "", mmproj_file)
    if healthy() and _proc and _proc.poll() is None and _loaded == want:
        _log("managed warm reuse model=%r mmproj=%r" % (want[0], want[1] or "none"))
        return "OK warm"

    kill()
    model_path = os.path.join(CONN["models_dir"], model_file or "")
    if not model_file or not os.path.isfile(model_path):
        msg = f"ERR model not found: {model_path}"
        _log(msg)
        return msg

    port = url().rsplit(":", 1)[-1] or "8080"
    # --reasoning off: Gemma-4 thinking templates otherwise fill max_tokens with
    # plans in reasoning_content and leave content empty (bullet outlines as "script").
    cmd = [CONN["llama_exe"], "-m", model_path, "--port", str(port),
           "-ngl", "999", "-c", str(int(CONN.get("ctx") or 8192)),
           "--host", "127.0.0.1",
           "--reasoning", "off", "--reasoning-budget", "0"]
    if mmproj_file:
        mp = os.path.join(CONN["models_dir"], mmproj_file)
        if os.path.isfile(mp):
            cmd += ["--mmproj", mp]
        else:
            _log(f"mmproj not found, starting text-only: {mp}")

    path = log_path()
    try:
        _log_fh = open(path, "a", encoding="utf-8", errors="replace")
        _log_fh.write(f"\n{'=' * 60}\n")
        _log_fh.write(f"spawn {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_fh.write(f"cmd: {subprocess.list2cmdline(cmd) if os.name == 'nt' else ' '.join(cmd)}\n")
        _log_fh.write(f"{'=' * 60}\n")
        _log_fh.flush()
    except Exception as e:
        _log_fh = None
        _log(f"could not open log file {path}: {e}")

    flags = 0x08000000 if os.name == "nt" else 0
    _log(f"spawning managed llama → log: {path}")
    _log(f"cmd: {cmd}")
    try:
        out = _log_fh if _log_fh is not None else subprocess.DEVNULL
        _proc = subprocess.Popen(
            cmd, creationflags=flags,
            stdout=out, stderr=subprocess.STDOUT if _log_fh is not None else subprocess.DEVNULL,
        )
    except Exception as e:
        _close_log_fh()
        msg = f"ERR spawn failed: {e}"
        _log(msg)
        return msg

    deadline = time.time() + 120
    while time.time() < deadline:
        if _proc.poll() is not None:
            tail = tail_log(30)
            msg = (
                f"ERR llama-server exited during load (exit={_proc.returncode}). "
                f"Check VRAM / model path. Last log:\n{tail}"
            )
            _log(msg)
            _close_log_fh()
            _proc = None
            return msg
        if healthy(1.5):
            _loaded = want
            _log("managed booted OK model=%r mmproj=%r port=%s" % (want[0], want[1] or "none", port))
            return "OK booted"
        time.sleep(1.0)

    msg = f"ERR boot timeout (120s). Last log:\n{tail_log(30)}"
    _log(msg)
    return msg


def _close_log_fh():
    global _log_fh
    if _log_fh is not None:
        try:
            _log_fh.flush()
            _log_fh.close()
        except Exception:
            pass
        _log_fh = None


# Connect-only backends: short idle TTL so the model auto-unloads after chat
# (PromptForgeLD pattern). Managed llama-server ignores this.
IDLE_TTL = 30


def _kill_managed() -> str:
    """Hard-kill managed llama-server — same approach as PromptForgeLD.

    taskkill /IM llama-server.exe catches orphans (Comfy restart, lost _proc
    handle). Then clear our Popen handle if any.
    """
    global _proc, _loaded
    notes = []
    try:
        if os.name == "nt":
            # Short timeout — Queue/Run must not stall on taskkill
            r = subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, timeout=2.5, check=False,
            )
            # 0 = killed something, 128 = process not found
            if r.returncode == 0:
                notes.append("taskkill llama-server.exe")
            else:
                notes.append("no llama-server.exe process")
        else:
            subprocess.run(
                ["pkill", "-f", "llama-server"],
                capture_output=True, timeout=2.5, check=False,
            )
            notes.append("pkill llama-server")
    except Exception as e:
        _log(f"managed image kill failed: {e}")
        notes.append(f"image kill failed: {e}")

    if _proc is not None:
        try:
            _proc.kill()
        except Exception:
            pass
        try:
            _proc.wait(timeout=1.0)
        except Exception:
            pass
        _proc = None
        notes.append("cleared managed handle")

    _loaded = ("", "")
    _close_log_fh()
    # Brief pause so the driver can reclaim VRAM before LTX boots
    time.sleep(0.35)
    return "; ".join(notes) if notes else "managed kill ran"


def _evict_remote() -> bool:
    """Unload model from LM Studio / Ollama VRAM — PromptForgeLD path.

    1) LM Studio native POST /api/v1/models/unload {instance_id: model}
    2) ttl=0 chat nudge (version-agnostic)
    3) lms unload --all CLI
    4) Ollama keep_alive=0
    Returns True if any path looked successful.
    """
    ok = False
    base = url()
    model = (CONN.get("remote_model") or "").strip() or "local"

    # ── LM Studio / OpenAI-compatible unload ──────────────────────────────
    if CONN["backend"] == LMSTUDIO:
        # PromptForge primary: instance_id == loaded model id from panel
        for path, body in (
            ("/api/v1/models/unload", {"instance_id": model}),
            ("/api/v0/models/unload", {"instance_id": model}),
            ("/api/v1/models/unload", {"identifier": model}),
            ("/api/v0/models/unload", {"identifier": model}),
        ):
            try:
                raw = json.dumps(body).encode()
                req = urllib.request.Request(
                    base + path, data=raw, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200:
                        ok = True
                        _log(f"LM Studio unload OK via {path} id={model!r}")
                        break
            except Exception as e:
                _log(f"LM Studio unload {path} failed: {e}")

        # Unload every other loaded model (embeddings etc.) so VRAM is clean
        try:
            for list_path in ("/api/v0/models", "/api/v1/models", "/v1/models"):
                try:
                    with urllib.request.urlopen(base + list_path, timeout=5) as r:
                        data = json.loads(r.read().decode("utf-8", "ignore") or "{}")
                except Exception:
                    continue
                rows = data.get("data") if isinstance(data, dict) else data
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    iid = (row.get("instance_id") or row.get("instanceId")
                           or row.get("identifier") or row.get("id") or row.get("model"))
                    if not iid or str(iid) == model:
                        continue
                    for upath in ("/api/v1/models/unload", "/api/v0/models/unload"):
                        try:
                            raw = json.dumps({"instance_id": str(iid)}).encode()
                            req = urllib.request.Request(
                                base + upath, data=raw, method="POST",
                                headers={"Content-Type": "application/json"},
                            )
                            with urllib.request.urlopen(req, timeout=10) as r:
                                if r.status == 200:
                                    ok = True
                                    _log(f"LM Studio also unloaded {iid!r}")
                                    break
                        except Exception:
                            continue
                break
        except Exception as e:
            _log(f"LM Studio list/unload-all failed: {e}")

        # CLI backup (same as PromptForge users who have lms on PATH)
        if not ok:
            try:
                r = subprocess.run(
                    ["lms", "unload", "--all"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if r.returncode == 0:
                    ok = True
                    _log("lms unload --all OK")
            except FileNotFoundError:
                pass
            except Exception as e:
                _log(f"lms unload failed: {e}")

        # ttl=0 nudge — PromptForge fallback; forces unload after empty turn
        if not ok:
            try:
                body = json.dumps({
                    "model": model, "ttl": 0, "max_tokens": 1,
                    "messages": [{"role": "user", "content": "."}],
                }).encode()
                req = urllib.request.Request(
                    base + "/v1/chat/completions", data=body, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    ok = ok or (r.status == 200)
                    _log(f"LM Studio ttl=0 nudge status={r.status}")
            except Exception as e:
                _log(f"LM Studio ttl=0 nudge failed: {e}")

    # ── Ollama ────────────────────────────────────────────────────────────
    if CONN["backend"] == OLLAMA:
        for path, body in (
            ("/api/generate", {"model": model, "keep_alive": 0}),
            ("/api/chat", {"model": model, "keep_alive": 0, "messages": []}),
        ):
            try:
                raw = json.dumps(body).encode()
                req = urllib.request.Request(
                    base + path, data=raw, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=8).read()
                ok = True
                _log(f"Ollama keep_alive=0 via {path}")
                break
            except Exception as e:
                _log(f"Ollama evict {path} failed: {e}")

    return ok


_abort = False


def abort():
    """Ask any in-flight chat_stream to stop at the next SSE line.

    Set by the cancel path. chat_stream checks this between tokens so the
    generation actually ENDS before we free VRAM — otherwise llama-server keeps
    writing into VRAM we just tried to reclaim.
    """
    global _abort
    _abort = True
    _log("abort() requested — in-flight stream will stop")


def clear_abort():
    global _abort
    _abort = False


def aborted() -> bool:
    return _abort


class StreamAborted(RuntimeError):
    """Raised inside chat_stream when abort() was called."""


def kill() -> str:
    """Unload the writer LLM and free its VRAM.

    Mirrors PromptForgeLD free():
      managed  → taskkill /IM llama-server.exe (+ clear handle)
      LM Studio / Ollama → native unload + ttl=0 / keep_alive=0
    Always safe to call twice.
    """
    if is_managed():
        msg = _kill_managed()
        # Confirm port is dead
        try:
            if healthy(1.0):
                msg += "; WARNING: managed port still healthy"
                _log("kill(): managed still healthy after taskkill")
        except Exception:
            pass
        _log(f"kill() → {msg}")
        return msg

    ok = _evict_remote()
    msg = "evicted remote model" if ok else "evict request sent (check LM Studio / Ollama)"
    _log(f"kill() → {msg}")
    return msg


def free(*, flush: bool = True, light: bool = False) -> str:
    """ONE entry for Free LLM / post-Generate / Queue hand-off.

    flush: also run Comfy/CUDA flush (PromptForgeLD does this after free).
    light: soft CUDA clear only (fast Queue path).
    """
    msg = kill()
    clear_abort()  # LLM is down; next run starts clean
    if flush:
        try:
            from .vram import flush_vram
        except ImportError:
            try:
                from vram import flush_vram  # type: ignore
            except ImportError:
                flush_vram = None
        if flush_vram is not None:
            try:
                flush_vram("ClaudePromptLD", light=light)
            except Exception as e:
                _log(f"flush_vram failed: {e}")
                msg = f"{msg}; flush failed: {e}"
    return msg


def _coerce_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for p in val:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)
    return str(val) if val else ""


def _extract_stream_piece(choice0: dict):
    """Return (kind, text): kind is 'content' or 'reasoning'.

    Reasoning models stream their think-phase in reasoning_content/thinking
    fields BEFORE any real content. Yielding those as content used to splice
    the model's plan (bullet lists, self-corrections) in front of the script,
    so reasoning is buffered by the caller and used only if the stream ends
    with zero real content (the silent-empty-run fix, preserved).
    """
    if not isinstance(choice0, dict):
        return "", ""
    delta = choice0.get("delta") or {}
    msg = choice0.get("message") or {}
    for src in (delta, msg, choice0):
        if not isinstance(src, dict):
            continue
        for key in ("content", "text"):
            t = _coerce_text(src.get(key))
            if t:
                return "content", t
    for src in (delta, msg, choice0):
        if not isinstance(src, dict):
            continue
        for key in ("reasoning_content", "reasoning", "thinking"):
            t = _coerce_text(src.get(key))
            if t:
                return "reasoning", t
    return "", ""


def chat_stream(messages, *, temperature=0.85, top_p=0.95, max_tokens=900, seed=None):
    """Stream assistant *content* only.

    Gemma-4 (and similar) templates often enable thinking: the model fills
    max_tokens with reasoning_content (plans, bullet outlines) and leaves
    content empty. We used to fall back to that reasoning as the script —
    that is why Generate showed rule-lists instead of a shot. Fix: disable
    thinking in the request, never yield reasoning as the shot text.
    """
    payload = {
        "model": CONN["remote_model"] if not is_managed() else "local",
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": True,
        # Kill think-mode so the model writes the shot in `content`, not a plan
        # in `reasoning_content`. Harmless if the server ignores these keys.
        "enable_thinking": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
        },
    }
    # PromptForgeLD: connect-only backends auto-evict after idle TTL seconds.
    if not is_managed():
        payload["ttl"] = IDLE_TTL
    if seed is not None:
        try:
            payload["seed"] = int(seed)
        except (TypeError, ValueError):
            pass

    n_msgs = len(messages or [])
    approx = sum(len(str(m.get("content", ""))) for m in (messages or []))
    _log(f"chat_stream → {url()}/v1/chat/completions  "
         f"backend={CONN['backend']!r} model={payload['model']!r} "
         f"max_tokens={payload['max_tokens']} msgs={n_msgs} content_chars≈{approx} "
         f"thinking=off")

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url() + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    pieces = 0
    chars = 0
    raw_lines = 0
    data_lines = 0
    samples = []
    reason_chars = 0
    delta_keys_seen = set()
    try:
        clear_abort()
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if aborted():
                    _log("chat_stream aborted by request — closing connection")
                    try:
                        resp.close()
                    except Exception:
                        pass
                    raise StreamAborted("generation cancelled")
                raw_lines += 1
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                if line.startswith("{") and "choices" not in line and "error" in line.lower():
                    _log(f"chat_stream non-SSE error body: {line[:500]}")
                    raise RuntimeError(f"LLM error: {line[:400]}")
                if not line.startswith("data:"):
                    if raw_lines <= 3 or "error" in line.lower():
                        _log(f"chat_stream skip line: {line[:240]}")
                    continue
                data_lines += 1
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    _log(f"chat_stream SSE error: {msg}")
                    raise RuntimeError(f"LLM error: {msg}")
                try:
                    choice0 = (obj.get("choices") or [{}])[0] or {}
                    delta = choice0.get("delta") or {}
                    if isinstance(delta, dict) and delta and len(delta_keys_seen) < 12:
                        delta_keys_seen.update(delta.keys())
                    if len(samples) < 4:
                        samples.append(body[:400])
                    kind, piece = _extract_stream_piece(choice0)
                except Exception:
                    continue
                if kind == "content" and piece:
                    pieces += 1
                    chars += len(piece)
                    yield piece
                elif kind == "reasoning" and piece:
                    # Log only — never surface thinking as the shot script.
                    reason_chars += len(piece)
    except StreamAborted:
        raise
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "ignore")[:600]
        except Exception:
            detail = str(e)
        _log(f"chat_stream HTTP {e.code}: {detail}")
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        _log(f"chat_stream URL error: {e}")
        raise RuntimeError(f"LLM connection failed: {e}") from e

    _log(f"chat_stream done: content_chunks={pieces} chars={chars} "
         f"reasoning_chars={reason_chars} sse_data_lines={data_lines} "
         f"raw_lines={raw_lines} delta_keys={sorted(delta_keys_seen) or 'none'}")
    if pieces == 0 or chars == 0:
        for i, s in enumerate(samples):
            _log(f"chat_stream empty sample[{i}]: {s}")
        tail = tail_log(25) if is_managed() else "(connect-only — check LM Studio / Ollama logs)"
        hint = ""
        if reason_chars > 0:
            hint = (
                f" Model spent {reason_chars} chars in reasoning/thinking and "
                "wrote 0 content — thinking may still be on in the server "
                "template (Gemma-4). Update llama.cpp chat template or set "
                "enable_thinking=false. "
            )
        raise RuntimeError(
            f"LLM returned 0 content tokens (sse_lines={data_lines}, "
            f"delta_keys={sorted(delta_keys_seen) or 'none'}).{hint}"
            f"Log: {log_path()}\n{tail}"
        )


def vision_supported() -> bool:
    if is_managed():
        return bool(_loaded[1])
    return True
