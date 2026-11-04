"""The LoRA viewer's one route: a picture and trigger words for each LoRA named.

The viewer (``javascript/model_chain_lora_viewer.js``) is a button at the end of
the prompt tool row that lists every LoRA in the prompt and the two Literal
boxes, with a slider for each weight. The browser reads the tags itself, so
names and weights show the moment it opens; this module answers the rest, for
all of them in one request:

* whether the name is a LoRA on disk, looked up exactly as Forge looks it up
  when it loads one (``networks.load_networks``: by alias, or by file name when
  the alias is ambiguous), with one rescan of the LoRA folders for a name it
  does not know yet, as Forge does;
* its picture: the card picture Forge's LoRA browser shows (``create_item``'s
  preview), else the first Civitai example LoRA Details has downloaded, as a
  small ``data:`` URI, else none;
* its trigger words: LoRA Details' Civitai library first, then Forge's own
  activation text for the LoRA split at commas, with no word twice.

It never hashes a file (a viewer of five LoRAs could otherwise read gigabytes
to open) and never writes anything; the answer reaches only the page that
asked, and a failure's log line names no LoRA and no path.

A second route, :data:`NAMES_ROUTE`, lists the LoRAs the viewer's *Add a LoRA*
box suggests from: every LoRA Forge knows, kept to the folders assigned to the
active UI Preset exactly as the extra-network browser keeps its cards
(``mc_extra_networks_filter.active_rules``: nothing assigned shows everything,
unless the strict reading is on), each with the name its tag is written with
(Forge's ``get_alias()``, as a card press writes it). Names only: no file is
read, so a folder of thousands answers at once.

A plain ``def`` handler, not ``async def``: the work reads files (user metadata,
the library's CSVs, a thumbnail), and FastAPI runs a plain handler on its thread
pool, where a coroutine would hold up every other request on the event loop.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("model_chain")

# Module globals, not imports inside install(): FastAPI resolves a handler's
# postponed annotations against these (the Gallery Quadrant's HTTP 422 trap).
try:
    from fastapi import Request  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except Exception:  # pragma: no cover - a host without FastAPI
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore

ROUTE = "/model-chain/lora-viewer/details"
NAMES_ROUTE = "/model-chain/lora-viewer/names"
MAX_LISTED = 20000
"""The most names the add box is given: more than any folder of LoRAs holds."""
MAX_NAMES = 64
MAX_NAME_LENGTH = 256
PICTURE_SIZE = 128
"""The fallback picture's longest side: the row shows it at 44 CSS pixels."""


# --------------------------------------------------------------------------- #
# Forge's side
# --------------------------------------------------------------------------- #


def forge_networks():
    """Forge's LoRA registry module (``extensions-builtin/sd_forge_lora``), or None."""
    module = sys.modules.get("networks")
    if module is not None and hasattr(module, "available_network_aliases"):
        return module
    try:
        import networks as module  # type: ignore
    except Exception:
        return None
    return module if hasattr(module, "available_network_aliases") else None


def lora_page():
    """Forge's LoRA page of the extra-network browser, or None."""
    try:
        from modules import ui_extra_networks  # type: ignore
    except Exception:
        return None
    for page in list(getattr(ui_extra_networks, "extra_pages", []) or []):
        if str(getattr(page, "name", "")).lower() == "lora":
            return page
    return None


def on_disk(networks, name: str):
    """The LoRA a tag names, the way Forge's ``load_networks`` finds it."""
    forbidden = getattr(networks, "forbidden_network_aliases", set()) or set()
    if name.lower() in forbidden:
        return networks.available_networks.get(name)
    return networks.available_network_aliases.get(name)


def resolve(names, networks=None) -> dict:
    """``{name: entry or None}``, after one rescan for the names not known yet."""
    networks = networks if networks is not None else forge_networks()
    if networks is None:
        return {name: None for name in names}
    found = {name: on_disk(networks, name) for name in names}
    missing = [name for name, entry in found.items() if entry is None]
    if missing and hasattr(networks, "update_available_networks_by_names"):
        try:
            networks.update_available_networks_by_names(missing)
        except Exception:
            logger.debug("Model Chain: the LoRA viewer could not rescan the LoRA folders",
                         exc_info=True)
        for name in missing:
            found[name] = on_disk(networks, name)
    return found


# --------------------------------------------------------------------------- #
# One LoRA's answer
# --------------------------------------------------------------------------- #


def split_activation(text) -> list[str]:
    """Forge's activation text as single trigger words: split at commas, semicolons
    and line ends, as LoRA Details splits a library's trigger words."""
    import mc_lora_library

    return mc_lora_library._split_triggers(str(text or ""))


def triggers_of(details: dict) -> list[str]:
    """The library's trigger words, then the activation text's, none twice."""
    words: list[str] = []
    seen: set[str] = set()
    for word in [word for word, _source in details.get("triggers", ())] + \
            split_activation(details.get("activation", "")):
        key = word.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            words.append(word.strip())
    return words


def fallback_picture(details: dict) -> str:
    """The first downloaded Civitai example that is a picture, as a data URI."""
    import mc_lora_images

    for url, image_id in details.get("prompt_images", ()) or ():
        try:
            path = mc_lora_images.find_local(url, image_id)
            if path is None or mc_lora_images.is_video(path):
                continue
            uri = mc_lora_images.thumbnail_data_uri(path, size=PICTURE_SIZE)
        except Exception:
            continue
        if uri:
            return uri
    return ""


def describe(name: str, entry, page) -> dict:
    """One LoRA's answer. Never raises: what cannot be read is left out."""
    answer = {"name": name, "found": entry is not None, "picture": "", "title": "",
              "triggers": [], "weight": None}
    if entry is None:
        return answer
    disk_name = str(getattr(entry, "name", "") or name)
    item = None
    if page is not None and hasattr(page, "create_item"):
        try:
            item = page.create_item(disk_name, enable_filter=False)
        except Exception:
            logger.debug("Model Chain: the LoRA viewer could not read a LoRA's card",
                         exc_info=True)
    if not item:
        item = {"name": disk_name, "filename": str(getattr(entry, "filename", "") or "")}
    answer["picture"] = str(item.get("preview") or "")
    answer["weight"] = preferred_weight(item)
    try:
        import mc_lora_details

        details = mc_lora_details.gather(page, disk_name, item=item, hash_files=False)
    except Exception:
        logger.debug("Model Chain: the LoRA viewer could not read the LoRA library",
                     exc_info=True)
        details = {"activation": str((item.get("user_metadata") or {}).get("activation text", "")
                                     or "")}
    answer["triggers"] = triggers_of(details)
    records = details.get("records") or ()
    answer["title"] = str(getattr(records[0], "title", "") or "") if records else ""
    if not answer["picture"]:
        answer["picture"] = fallback_picture(details)
    return answer


def preferred_weight(item) -> float | None:
    """The card's *Preferred weight* (Forge's user metadata), which a press on
    the card writes into the tag; None when it has none (0 is none to Forge)."""
    try:
        value = float((item.get("user_metadata") or {}).get("preferred weight") or 0)
    except (TypeError, ValueError, AttributeError):
        return None
    return value if value else None


def details(names, *, networks=None, page=None) -> dict:
    """The route's answer for the names asked, in the order asked, each once."""
    wanted: list[str] = []
    for raw in names or ():
        name = str(raw or "").strip()[:MAX_NAME_LENGTH]
        if name and name not in wanted:
            wanted.append(name)
        if len(wanted) >= MAX_NAMES:
            break
    if page is None:
        page = lora_page()
    found = resolve(wanted, networks)
    return {"loras": [describe(name, found.get(name), page) for name in wanted]}


# --------------------------------------------------------------------------- #
# The add box's names
# --------------------------------------------------------------------------- #


def tag_name(entry, name: str) -> str:
    """The name a card press writes into the tag: the alias, or the file's name
    where Forge prefers it or the alias is ambiguous (``get_alias``)."""
    try:
        found = entry.get_alias()
    except Exception:
        found = getattr(entry, "alias", "") or ""
    return str(found or name)


def preset_rules(page):
    """The active preset's folders for the LoRA page, read as the browser reads
    them: None shows everything, an empty list nothing."""
    try:
        import mc_extra_networks_filter

        return mc_extra_networks_filter.active_rules(page if page is not None else "lora")
    except Exception:
        logger.debug("Model Chain: the LoRA viewer could not read the preset's folders",
                     exc_info=True)
        return None


def listing(*, networks=None, page=None, rules=False) -> dict:
    """The add box's names: ``{"loras": [{"name", "tag"}], "preset", "filtered"}``,
    sorted by name, the LoRAs outside the preset's folders left out.

    ``rules`` is the preset's folders when a caller has them (a test); left at
    False they are read as the browser reads them."""
    import mc_extra_networks_folders
    import mc_extra_networks_filter

    networks = networks if networks is not None else forge_networks()
    if rules is False:
        rules = preset_rules(page if page is not None else lora_page())
    preset = ""
    try:
        preset = mc_extra_networks_folders.current_preset()
    except Exception:
        preset = ""
    answer = {"loras": [], "preset": preset, "filtered": rules is not None}
    if networks is None:
        return answer
    found = []
    for name, entry in list((getattr(networks, "available_networks", None) or {}).items()):
        filename = str(getattr(entry, "filename", "") or "")
        if rules is not None and not mc_extra_networks_filter.item_allowed({"filename": filename}, rules):
            continue
        found.append({"name": str(name), "tag": tag_name(entry, str(name))})
    found.sort(key=lambda one: (one["name"].casefold(), one["name"]))
    answer["loras"] = found[:MAX_LISTED]
    return answer


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal, and
    idempotent: a UI reload calls it again."""
    if app is None or not hasattr(app, "add_api_route"):
        return False
    paths = {getattr(existing, "path", None) for existing in getattr(app, "routes", [])}
    if ROUTE in paths and NAMES_ROUTE in paths:
        return True
    if Request is None or JSONResponse is None:
        logger.debug("Model Chain: the LoRA viewer has no FastAPI to register its route on")
        return False

    def lora_viewer_details(request: Request):  # type: ignore[valid-type]
        try:
            body = details(request.query_params.getlist("name"))
            return JSONResponse(body, headers={"Cache-Control": "no-store"})
        except Exception:
            logger.warning("Model Chain: the LoRA viewer's details failed", exc_info=True)
            return JSONResponse({"loras": [], "error": "The details could not be read; "
                                 "the log says why."}, status_code=500)

    def lora_viewer_names():
        try:
            return JSONResponse(listing(), headers={"Cache-Control": "no-store"})
        except Exception:
            logger.warning("Model Chain: the LoRA viewer's list of LoRAs failed", exc_info=True)
            return JSONResponse({"loras": [], "error": "Your LoRAs could not be listed; "
                                 "the log says why."}, status_code=500)

    try:
        if ROUTE not in paths:
            app.add_api_route(ROUTE, lora_viewer_details, methods=["GET"])
        if NAMES_ROUTE not in paths:
            app.add_api_route(NAMES_ROUTE, lora_viewer_names, methods=["GET"])
    except Exception:
        logger.debug("Model Chain: could not register the LoRA viewer's routes", exc_info=True)
        return False
    return True
