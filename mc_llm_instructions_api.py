"""The writers' instructions over HTTP, for the system prompt editor (``mcSheet.systemPrompt``).

One route, three actions, each a request and its answer -- nothing is held
open:

    POST /model-chain/llm/instructions
        {"action": "describe", "variant", "mode", "fallback"}
            the sets and modes the editor offers, the set it opens on with its
            text and where that comes from (:func:`mc_llm_instructions.view`),
            and a line about the LLM side
        {"action": "save", "variant", "mode", "text"}
        {"action": "restore", "variant", "mode"}
            each answered with the set as it now stands

The editor is opened from LLM Studio's MiniMax workspace, from Mini Paint NEO's
Clipboard tab and from its Send to WanGP popup; all three read and write here,
so one edit is the instructions of every request that brings none of its own.

The gate is the one LLM Studio's conversation routes pass
(:func:`mc_llm_conversation_api.checked`): the page's own origin and the key
Python wrote into the page. Nothing here logs the instructions' text.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")

# Module globals, not imported inside :func:`install`: FastAPI reads a
# handler's annotations against the module's globals, and under postponed
# annotations a name imported locally resolves to nothing (the Gallery
# Quadrant's HTTP 422, CLAUDE.md).
try:                                                 # pragma: no cover - host only
    from fastapi import Request  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except Exception:                                    # pragma: no cover - tests
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore

ROUTE = "/model-chain/llm/instructions"


def availability() -> dict:
    """A line about the LLM side, for the top of the editor: who writes, or why
    nobody can right now."""
    try:
        import mc_llm_api

        found = mc_llm_api.capabilities()
    except Exception:
        logger.debug("Model Chain: the writers' availability could not be read", exc_info=True)
        return {"state": "unknown", "text": "Whether LLM Studio can write right now could not be read."}
    if not found.get("enabled"):
        return {"state": "off", "text": str(found.get("reason") or "LLM Studio is switched off.")}
    if not found.get("configured"):
        return {"state": "unconfigured", "text": str(found.get("reason") or "No language model is set up.")}
    model = str(found.get("model") or "").strip()
    text = f"LLM Studio writes these prompts with {model}." if model else "LLM Studio writes these prompts."
    if not found.get("vision"):
        text += " It has no vision projector, so a request with a picture is refused."
        return {"state": "novision", "text": text}
    return {"state": "ready", "text": text}


def answer(body) -> tuple:
    """``(status, payload)`` for one request. Never raises."""
    import mc_llm_instructions as store

    body = body if isinstance(body, dict) else {}
    action = str(body.get("action") or "")
    try:
        if action == "describe":
            variant = body.get("variant") or body.get("fallback") or store.FL2VA
            try:
                chosen = store.set_of(variant)
            except store.Refused:
                chosen = store.FL2VA
            mode = body.get("mode") or store.TEXT
            try:
                mode = store.mode_of(mode)
            except store.Refused:
                mode = store.TEXT
            return 200, {"ok": True, "sets": store.sets(), "modes": store.modes(),
                         "view": store.view(chosen, mode), "availability": availability()}
        if action == "save":
            view = store.save(body.get("variant"), body.get("mode"), body.get("text"))
            status = ("Saved. Every request that brings no instructions of its own uses these "
                      "from now on, after a restart too." if view["source"] == store.OVERRIDE
                      else "That is the default, so no override is kept.")
            return 200, {"ok": True, "view": view, "status": status}
        if action == "restore":
            had = store.restore(body.get("variant"), body.get("mode"))
            view = store.view(body.get("variant"), body.get("mode"))
            return 200, {"ok": True, "view": view,
                         "status": "Back to the default." if had
                         else "There was no override; the default is shown."}
    except store.Refused as error:
        return 400, {"ok": False, "code": error.code, "message": error.reason}
    except Exception:
        logger.warning("Model Chain: a writer instructions request failed", exc_info=True)
        return 500, {"ok": False, "code": "failed", "message": "That did not work; the log says why."}
    return 400, {"ok": False, "code": "unknown_action",
                 "message": f"{action[:40] or 'That'} is not something the instructions route does."}


def _offload(call, *args):
    """Blocking work off the event loop: the instructions are a file."""
    try:
        from starlette.concurrency import run_in_threadpool

        return run_in_threadpool(call, *args)
    except ImportError:                              # pragma: no cover - tests
        async def here():
            return call(*args)

        return here()


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal, and
    idempotent: a UI reload calls it again."""
    if app is None or not hasattr(app, "add_api_route"):
        return False
    for existing in getattr(app, "routes", []):
        if getattr(existing, "path", None) == ROUTE:
            return True
    if Request is None or JSONResponse is None:
        logger.debug("Model Chain: no FastAPI to register the instructions route on")
        return False

    async def writer_instructions(request: Request):  # type: ignore[valid-type]
        import mc_llm_conversation_api as conversation

        try:
            conversation.checked(request)
        except conversation.Refused as refusal:
            return JSONResponse({"ok": False, "code": "forbidden" if refusal.status == 403
                                 else "auth_required", "message": refusal.reason},
                                status_code=refusal.status, headers={"Cache-Control": "no-store"})
        try:
            body = await request.json()
        except Exception:
            body = {}
        status, payload = await _offload(answer, body)
        return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})

    try:
        app.add_api_route(ROUTE, writer_instructions, methods=["POST"])
    except Exception:
        logger.debug("Model Chain: could not register the instructions route", exc_info=True)
        return False
    return True


__all__ = ["ROUTE", "answer", "availability", "install"]
