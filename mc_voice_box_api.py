"""The Voice Box's routes: JSON on the page token, audio as bytes, nothing held open.

Every route here is one the page (``javascript/voice_box.js``) calls with the
token its root carries -- the same token, header and origin check as Voice
Chat's routes, because the page is served by the same WebUI to the same person
and a second gate would only be a second way to lock them out
(:func:`mc_voice_api._checked` says why there is no sign-in). Render progress
is read by polling a server-owned record, never by a stream: an open stream on
Forge's origin is a connection the page cannot spare, and a render is minutes
long (Mini Paint NEO's browser-connection incident, 2026-09-17).

Blocking work -- files, the native folder dialog, a sample's normalisation --
runs off the event loop through :func:`mc_voice_api._offload`. Nothing here
touches a card: :mod:`mc_voice_box` asks for one through its turn client, and
the runtime's status is read, never driven, except by the one *Unload* action.
"""

from __future__ import annotations

import logging
import threading
import urllib.parse

import mc_voice_api as voice_api
import mc_voice_box as box

try:  # pragma: no cover - the host always has FastAPI; the tests import it explicitly
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response
except ImportError:  # pragma: no cover
    Request = None
    JSONResponse = None
    Response = None

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

PREFIX = "/model-chain/voice-box"

STATUS_ROUTE = f"{PREFIX}/status"
INSTALL_ROUTE = f"{PREFIX}/install"
SETTINGS_ROUTE = f"{PREFIX}/settings"
FOLDER_ROUTE = f"{PREFIX}/settings/folder"
SAMPLES_ROUTE = f"{PREFIX}/samples"
SAMPLE_UPLOAD_ROUTE = f"{PREFIX}/samples/upload"
SAMPLE_RENAME_ROUTE = f"{PREFIX}/samples/rename"
SAMPLE_DELETE_ROUTE = f"{PREFIX}/samples/delete"
SAMPLE_AUDIO_ROUTE = f"{PREFIX}/samples/audio"
PROMPTS_ROUTE = f"{PREFIX}/prompts"
PROMPT_ADD_ROUTE = f"{PREFIX}/prompts/add"
PROMPT_FAVOURITE_ROUTE = f"{PREFIX}/prompts/favourite"
PROMPT_DELETE_ROUTE = f"{PREFIX}/prompts/delete"
CONFIGURATIONS_ROUTE = f"{PREFIX}/configurations"
CONFIGURATION_SAVE_ROUTE = f"{PREFIX}/configurations/save"
CONFIGURATION_DELETE_ROUTE = f"{PREFIX}/configurations/delete"
PIPELINES_ROUTE = f"{PREFIX}/pipelines"
PIPELINE_NEW_ROUTE = f"{PREFIX}/pipelines/new"
PIPELINE_SAVE_ROUTE = f"{PREFIX}/pipelines/save"
PIPELINE_DELETE_ROUTE = f"{PREFIX}/pipelines/delete"
RENDER_ROUTE = f"{PREFIX}/render"
JOBS_ROUTE = f"{PREFIX}/jobs"
JOB_CANCEL_ROUTE = f"{PREFIX}/jobs/cancel"
JOBS_CLEAR_ROUTE = f"{PREFIX}/jobs/clear"
OUTPUTS_ROUTE = f"{PREFIX}/outputs"
OUTPUT_RENAME_ROUTE = f"{PREFIX}/outputs/rename"
OUTPUT_LOOP_ROUTE = f"{PREFIX}/outputs/loop"
OUTPUT_DELETE_ROUTE = f"{PREFIX}/outputs/delete"
OUTPUT_SAVE_ROUTE = f"{PREFIX}/outputs/save"
OUTPUT_AUDIO_ROUTE = f"{PREFIX}/outputs/audio"
RUNTIME_ROUTE = f"{PREFIX}/runtime"

ROUTES = (STATUS_ROUTE, INSTALL_ROUTE, SETTINGS_ROUTE, FOLDER_ROUTE, SAMPLES_ROUTE,
          SAMPLE_UPLOAD_ROUTE, SAMPLE_RENAME_ROUTE, SAMPLE_DELETE_ROUTE, SAMPLE_AUDIO_ROUTE,
          PROMPTS_ROUTE, PROMPT_ADD_ROUTE, PROMPT_FAVOURITE_ROUTE, PROMPT_DELETE_ROUTE,
          CONFIGURATIONS_ROUTE, CONFIGURATION_SAVE_ROUTE, CONFIGURATION_DELETE_ROUTE,
          PIPELINES_ROUTE, PIPELINE_NEW_ROUTE, PIPELINE_SAVE_ROUTE, PIPELINE_DELETE_ROUTE,
          RENDER_ROUTE, JOBS_ROUTE, JOB_CANCEL_ROUTE, JOBS_CLEAR_ROUTE, OUTPUTS_ROUTE,
          OUTPUT_RENAME_ROUTE, OUTPUT_LOOP_ROUTE, OUTPUT_DELETE_ROUTE, OUTPUT_SAVE_ROUTE,
          OUTPUT_AUDIO_ROUTE, RUNTIME_ROUTE)

TITLE_HEADER = "x-mc-title"
"""A sample's title on an upload, percent-encoded UTF-8 (headers are Latin-1)."""
SOURCE_HEADER = "x-mc-source"
MAX_UPLOAD_BYTES = box.MAX_SAMPLE_BYTES

Refused = voice_api.Refused
_installed = False


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


def _engine():
    import mc_voice_vibevoice

    return mc_voice_vibevoice


def _runtime():
    import mc_voice_vibevoice_runtime

    return mc_voice_vibevoice_runtime


def _part(name: str, call, fallback):
    """One part of the status, or its failure as a sentence. The page draws the rest."""
    try:
        return call()
    except Exception as exc:
        logger.debug("Model Chain: the Voice Box status could not read %s", name, exc_info=True)
        fallen = fallback() if callable(fallback) else fallback
        if isinstance(fallen, dict):
            fallen = dict(fallen, error=f"{name} could not be read: {exc}")
        return fallen


def status_payload() -> dict:
    """Everything the page draws from, in one answer it polls."""
    client = box.turns()
    return {
        "ok": True,
        "engine": _part("the engine", lambda: _engine().public_status(),
                        {"ready": False, "message": "VibeVoice's status could not be read"}),
        "progress": _part("the install", lambda: _engine().progress(), {}),
        "engine_settings": _part("the engine's settings", lambda: _engine().settings(), {}),
        "settings": _part("the settings", box.settings, dict(box.SETTINGS_DEFAULTS)),
        "cards": _part("the cards", lambda: client.cards() if client else [], []),
        "runtime": _part("the runtime", lambda: _runtime().status(), {"cards": {}}),
        "turns": _part("the cards' turns", lambda: client.snapshot() if client else [], []),
        "jobs": _part("the jobs", box.jobs, []),
    }


def install_payload(values: dict) -> dict:
    """Start an install in the background; the page follows it through /status."""
    engine = _engine()
    part = str(values.get("part") or "").strip()
    folder = str(values.get("folder") or "").strip()
    already = (engine.progress() or {}).get("running")
    if already:
        return {"ok": True, "already": True}

    def run():
        try:
            if folder:
                engine.install_from(part or "runtime", folder)
            else:
                engine.install(part)
        except Exception:
            # Logged with its reason where the page's progress reads it.
            logger.debug("Model Chain: the VibeVoice install thread ended on an error",
                         exc_info=True)

    threading.Thread(target=run, name="mc-vibevoice-install", daemon=True).start()
    return {"ok": True, "already": False}


ENGINE_KEYS = ("steps", "cfg_scale", "seed", "max_new_tokens", "card_uuid", "model_id",
               "keep_warm")
"""What the engine's own settings file holds. The card, the model and the warm
stay are Voice Box's to decide (its render service reads its own file), and are
mirrored into the engine's so that the two files never disagree about them."""


def settings_payload(values: dict) -> dict:
    """Voice Box's own settings and the engine's, saved together, answered together.

    The engine validates first -- a model id it does not know, a step count out
    of range -- so a refused value leaves both files as they were.
    """
    theirs = {key: values[key] for key in ENGINE_KEYS if key in values}
    engine_settings = _engine().set_settings(theirs) if theirs else _engine().settings()
    own = {key: values[key] for key in box.SETTINGS_DEFAULTS if key in values}
    found = box.set_settings(own) if own else box.settings()
    return {"ok": True, "settings": found, "engine_settings": engine_settings}


def folder_payload(_values: dict) -> dict:
    import mc_llm_native

    try:
        chosen = box.choose_save_folder()
    except mc_llm_native.Unavailable as exc:
        raise Refused(409, str(exc)) from None
    return {"ok": True, "save_folder": chosen, "chosen": bool(chosen)}


def samples_payload(_values: dict) -> dict:
    return {"ok": True, "samples": box.samples()}


def upload_sample(body: bytes, title: str, source: str) -> dict:
    return {"ok": True, "sample": box.add_sample(body, title, source or "file")}


def rename_sample_payload(values: dict) -> dict:
    return {"ok": True, "sample": box.rename_sample(values.get("id"), values.get("title"))}


def delete_sample_payload(values: dict) -> dict:
    return {"ok": True, **box.delete_sample(values.get("id"))}


def prompts_payload(_values: dict) -> dict:
    return {"ok": True, **box.prompts()}


def add_prompt_payload(values: dict) -> dict:
    return {"ok": True, "prompt": box.remember_prompt(values.get("text")), **box.prompts()}


def favourite_prompt_payload(values: dict) -> dict:
    found = box.favourite_prompt(values.get("id"), bool(values.get("favourite", True)))
    return {"ok": True, "prompt": found, **box.prompts()}


def delete_prompt_payload(values: dict) -> dict:
    return {"ok": True, **box.delete_prompt(values.get("id")), **box.prompts()}


def configurations_payload(_values: dict) -> dict:
    return {"ok": True, "configurations": box.configurations()}


def save_configuration_payload(values: dict) -> dict:
    found = box.save_configuration(values.get("configuration") or values)
    return {"ok": True, "configuration": found, "configurations": box.configurations()}


def delete_configuration_payload(values: dict) -> dict:
    return {"ok": True, **box.delete_configuration(values.get("id")),
            "configurations": box.configurations()}


def pipelines_payload(_values: dict) -> dict:
    return {"ok": True, "pipelines": box.pipelines()}


def new_pipeline_payload(values: dict) -> dict:
    return {"ok": True, "pipeline": box.new_pipeline(values.get("name")),
            "pipelines": box.pipelines()}


def save_pipeline_payload(values: dict) -> dict:
    found = box.save_pipeline(values.get("pipeline") or values)
    return {"ok": True, "pipeline": found}


def delete_pipeline_payload(values: dict) -> dict:
    return {"ok": True, **box.delete_pipeline(values.get("id")), "pipelines": box.pipelines()}


def render_payload(values: dict) -> dict:
    configuration = values.get("configuration")
    found = box.render(values.get("pipeline_id"), str(values.get("prompt") or ""),
                       str(values.get("configuration_id") or ""), str(values.get("name") or ""),
                       inline=configuration if isinstance(configuration, dict) else None)
    return {"ok": True, "job": found}


def jobs_payload(_values: dict) -> dict:
    return {"ok": True, "jobs": box.jobs()}


def cancel_job_payload(values: dict) -> dict:
    return {"ok": True, "job": box.cancel_job(values.get("id"))}


def clear_jobs_payload(_values: dict) -> dict:
    """Withdraw every queued render, on every card. The running one is Cancel's."""
    cleared = box.clear_queue()
    return {"ok": True, "cleared": cleared, "jobs": box.jobs()}


def outputs_payload(values: dict) -> dict:
    return {"ok": True, "outputs": box.outputs(str(values.get("pipeline_id") or ""))}


def rename_output_payload(values: dict) -> dict:
    return {"ok": True, "output": box.rename_output(values.get("id"), values.get("name"))}


def loop_output_payload(values: dict) -> dict:
    return {"ok": True, "output": box.set_loop(values.get("id"), bool(values.get("loop")))}


def delete_output_payload(values: dict) -> dict:
    return {"ok": True, **box.delete_output(values.get("id"))}


def save_output_payload(values: dict) -> dict:
    try:
        path = box.save_output(values.get("id"))
    except box.VoiceBoxError as exc:
        if "Choose a folder" in str(exc):
            raise Refused(409, str(exc)) from None
        raise
    return {"ok": True, "path": path, "save_folder": box.settings().get("save_folder", "")}


def runtime_payload(values: dict) -> dict:
    """*Unload* frees the warm guest through the turn system; *stop* ends its process."""
    action = str(values.get("action") or "").strip().lower()
    card = str(values.get("card_uuid") or box.settings().get("card_uuid") or "")
    client = box.turns()
    if action == "unload":
        if client is None:
            raise Refused(409, "Voice Box is not connected to the cards on this WebUI.")
        freed = client.unload(card, "unloaded from the Voice Box")
        return {"ok": True, "freed": int(freed or 0), "runtime": _runtime().status()}
    if action == "stop":
        _runtime().stop(card, reason="stopped from the Voice Box")
        return {"ok": True, "runtime": _runtime().status()}
    raise Refused(400, "That runtime action is not one the Voice Box knows.")


def sample_audio(identifier: str) -> tuple[bytes, str, str]:
    """A sample's sound, its title and its format: always a WAV."""
    entry = box.sample(identifier)
    return box.sample_wav(entry["id"]), str(entry.get("title") or "sample"), "wav"


def output_audio(identifier: str) -> tuple[bytes, str, str]:
    """A render's sound, its name and its format: an MP3, or a WAV where MP3 was not made."""
    entry = box.output(identifier)
    audio, kind = box.output_audio(entry["id"])
    return audio, str(entry.get("name") or "render"), kind


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def _decoded(headers, name: str, limit: int = 512) -> str:
    raw = str((headers or {}).get(name) or "")[:limit * 3]
    try:
        return urllib.parse.unquote(raw, errors="replace")[:limit]
    except Exception:
        return raw[:limit]


def install(_demo=None, app=None) -> bool:
    """Register the routes on the WebUI's FastAPI app. Idempotent, never fatal.

    Signature is ``script_callbacks.on_app_started``'s. A UI reload calls it
    again, and a second registration would give FastAPI two matching routes for
    one path, so the existing paths are looked for first.
    """
    global _installed

    if app is None or not hasattr(app, "add_api_route"):
        return False
    existing = {getattr(route, "path", None) for route in getattr(app, "routes", [])}
    if all(path in existing for path in ROUTES):
        _installed = True
        return True
    if Request is None:
        logger.debug("Model Chain: Voice Box has no FastAPI to register routes on")
        return False

    checked = voice_api._checked
    offload = voice_api._offload
    read_json = voice_api._json

    def _refusal(exc):
        return JSONResponse({"ok": False, "error": exc.reason}, status_code=exc.status)

    def _declined(exc: box.VoiceBoxError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=int(exc.status))

    def _failed(what: str, message: str, status: int = 500):
        logger.warning("Model Chain: %s", what, exc_info=True)
        return JSONResponse({"ok": False, "error": message}, status_code=status)

    def _json_route(route: str, call, failure: str):
        async def handler(request: Request):
            try:
                checked(request, route)
                payload = await read_json(request)
                return JSONResponse(await offload(call, payload))
            except Refused as exc:
                return _refusal(exc)
            except box.VoiceBoxError as exc:
                return _declined(exc)
            except Exception:
                return _failed(f"a Voice Box request to {route} failed", failure)

        return handler

    def _audio_route(route: str, call, fallback: str):
        async def handler(request: Request):
            try:
                checked(request, route)
                identifier = str(request.query_params.get("id") or "")
                download = str(request.query_params.get("download") or "") not in ("", "0")
                audio, name, kind = await offload(call, identifier)
            except Refused as exc:
                return _refusal(exc)
            except box.VoiceBoxError as exc:
                return _declined(exc)
            except Exception:
                return _failed(f"a Voice Box request to {route} failed",
                               f"That {fallback} could not be read.")
            headers = {"Cache-Control": "no-store"}
            if download:
                safe = urllib.parse.quote(box._safe_filename(name, fallback))
                headers["Content-Disposition"] = (f"attachment; filename=\"{fallback}.{kind}\"; "
                                                  f"filename*=UTF-8''{safe}.{kind}")
            return Response(content=audio, media_type=box.MEDIA_TYPES[kind], headers=headers)

        return handler

    async def upload_route(request: Request):
        try:
            checked(request, SAMPLE_UPLOAD_ROUTE)
            body = await request.body()
            if len(body) > MAX_UPLOAD_BYTES:
                raise Refused(413, "That recording is too large for a sample.")
            title = _decoded(request.headers, TITLE_HEADER, box.MAX_TITLE_CHARS)
            source = _decoded(request.headers, SOURCE_HEADER, 64)
            return JSONResponse(await offload(upload_sample, body, title, source))
        except Refused as exc:
            return _refusal(exc)
        except box.VoiceBoxError as exc:
            return _declined(exc)
        except Exception:
            return _failed("a Voice Box sample upload failed", "That sample could not be kept.")

    posts = (
        (STATUS_ROUTE, lambda _p: status_payload(), "Voice Box's status could not be read."),
        (INSTALL_ROUTE, install_payload, "VibeVoice could not be installed."),
        (SETTINGS_ROUTE, settings_payload, "The settings could not be saved."),
        (FOLDER_ROUTE, folder_payload, "The folder dialog could not be opened."),
        (SAMPLES_ROUTE, samples_payload, "The samples could not be listed."),
        (SAMPLE_RENAME_ROUTE, rename_sample_payload, "The sample could not be renamed."),
        (SAMPLE_DELETE_ROUTE, delete_sample_payload, "The sample could not be deleted."),
        (PROMPTS_ROUTE, prompts_payload, "The prompts could not be listed."),
        (PROMPT_ADD_ROUTE, add_prompt_payload, "The prompt could not be remembered."),
        (PROMPT_FAVOURITE_ROUTE, favourite_prompt_payload, "The prompt could not be marked."),
        (PROMPT_DELETE_ROUTE, delete_prompt_payload, "The prompt could not be deleted."),
        (CONFIGURATIONS_ROUTE, configurations_payload, "The configurations could not be listed."),
        (CONFIGURATION_SAVE_ROUTE, save_configuration_payload,
         "The configuration could not be saved."),
        (CONFIGURATION_DELETE_ROUTE, delete_configuration_payload,
         "The configuration could not be deleted."),
        (PIPELINES_ROUTE, pipelines_payload, "The pipelines could not be listed."),
        (PIPELINE_NEW_ROUTE, new_pipeline_payload, "The pipeline could not be created."),
        (PIPELINE_SAVE_ROUTE, save_pipeline_payload, "The pipeline could not be saved."),
        (PIPELINE_DELETE_ROUTE, delete_pipeline_payload, "The pipeline could not be deleted."),
        (RENDER_ROUTE, render_payload, "The render could not be queued."),
        (JOBS_ROUTE, jobs_payload, "The jobs could not be listed."),
        (JOB_CANCEL_ROUTE, cancel_job_payload, "The job could not be cancelled."),
        (JOBS_CLEAR_ROUTE, clear_jobs_payload, "The queue could not be cleared."),
        (OUTPUTS_ROUTE, outputs_payload, "The renders could not be listed."),
        (OUTPUT_RENAME_ROUTE, rename_output_payload, "The render could not be renamed."),
        (OUTPUT_LOOP_ROUTE, loop_output_payload, "The render's loop could not be set."),
        (OUTPUT_DELETE_ROUTE, delete_output_payload, "The render could not be deleted."),
        (OUTPUT_SAVE_ROUTE, save_output_payload, "The render could not be saved."),
        (RUNTIME_ROUTE, runtime_payload, "The runtime could not be changed."),
    )
    for route, call, failure in posts:
        if route not in existing:
            app.add_api_route(route, _json_route(route, call, failure), methods=["POST"])
    if SAMPLE_UPLOAD_ROUTE not in existing:
        app.add_api_route(SAMPLE_UPLOAD_ROUTE, upload_route, methods=["POST"])
    if SAMPLE_AUDIO_ROUTE not in existing:
        app.add_api_route(SAMPLE_AUDIO_ROUTE, _audio_route(SAMPLE_AUDIO_ROUTE, sample_audio,
                                                           "sample"), methods=["GET"])
    if OUTPUT_AUDIO_ROUTE not in existing:
        app.add_api_route(OUTPUT_AUDIO_ROUTE, _audio_route(OUTPUT_AUDIO_ROUTE, output_audio,
                                                           "render"), methods=["GET"])
    _installed = True
    logger.info("Model Chain: Voice Box routes registered under %s", PREFIX)
    return True
