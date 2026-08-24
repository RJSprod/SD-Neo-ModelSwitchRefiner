"""What the two Literal Prompt boxes found on the page, written to the log.

These boxes depend on two other extensions' cooperation and cannot see either
of them from Python. Whether Tag Autocomplete claimed them, and whether Forge's
"last prompt box you used" family exists to be joined, are facts that live in
the browser -- and the answer to "it does not work here" has so far been a
console snippet somebody has to open the developer tools to paste.

So the browser says it instead. ``javascript/model_chain_literals.js`` builds a
small fixed-shape report the first time somebody puts the caret in a literal
box -- which is exactly when tag completion is the thing they are expecting to
happen -- and posts it here, where it becomes one line in the same log every
other message from this extension goes to.

What this is not
----------------
A telemetry channel. It leaves this machine no more than any other line of the
WebUI's log does, and it cannot carry a prompt: :func:`describe` reads a fixed
set of keys, coerces every one of them to a bool or to one of a handful of
known words, and ignores everything else in the payload. There is nowhere for
free text to get in, which is deliberate -- a diagnostic that could carry what
somebody typed would be a diagnostic nobody should install.

One line, once per page load: a warning when something is wrong, an ordinary
line when nothing is. It lands in the console and -- see :mod:`mc_logfile` -- in
``<LLM data root>/logs/model_chain.log`` beside the LLM's own.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ROUTE = "/model-chain/literal-prompts/report"
"""Where the browser posts. Under the extension's own name, and a POST because
it is a report rather than a page."""

FLAGS = ("boxesFound", "claimed", "autocompleteInstalled", "listWrapped",
         "inTheirList", "promptFamily", "placed", "liftedThirdParty")
"""The booleans the browser may answer with. Anything else in the payload is
ignored -- see the module docstring."""

CONFIG = {"missing": "not installed", "null": "still loading", "loaded": "loaded"}
"""The three states Tag Autocomplete's ``TAC_CFG`` can be in, in words.

``var TAC_CFG = null`` until it has read its tag files, so "still loading" is a
real answer and not a failure: the boxes are offered again on the next UI
update, and its own setup asks for the list this extension has extended.
"""


def _flag(value) -> bool:
    return value is True


def describe(report) -> tuple[str, bool]:
    """``(one line for the log, is anything wrong)``.

    Never raises and never trusts: every value is coerced here, so a payload
    from anywhere at all produces a line built out of this module's own words.
    """
    if not isinstance(report, dict):
        report = {}

    found = {name: _flag(report.get(name)) for name in FLAGS}
    config = CONFIG.get(str(report.get("config") or ""), "unknown")
    # Three answers and no fourth: the browser may say true, false, or nothing
    # readable, and "nothing readable" is a word chosen here rather than a value
    # from the payload. Interpolating the payload's own value was how a string
    # from the page first reached a log line, which `test_no_value_from_the_
    # payload_reaches_the_line` now stands in front of.
    answer = report.get("thirdPartyBoxes")
    third_party = "yes" if answer is True else "no" if answer is False else "unknown"

    if not found["boxesFound"]:
        return ("Model Chain: the Literal Prompt boxes reported in without "
                "finding themselves on the page", True)

    if found["claimed"]:
        # The switch that used to stop this. Said out loud rather than left for
        # somebody to discover that a setting reading "off" is not off for two
        # boxes: it is lifted for the length of one call, on these two
        # textareas, and every other textbox it covers still answers to it.
        aside = (" (Tag Autocomplete's \"Active in third party textboxes\" switch"
                 " is off; these two were handed over anyway, as the prompt boxes"
                 " they are -- nothing else that switch covers is affected)"
                 if found["liftedThirdParty"] else "")
        return ("Model Chain: the Literal Prompt boxes have tag completion"
                + aside + ", and the prompt family "
                + ("was" if found["promptFamily"] else "was not")
                + " there to join", not found["promptFamily"])

    if not found["autocompleteInstalled"]:
        return ("Model Chain: no Tag Autocomplete on this page, so the Literal "
                "Prompt boxes complete nothing -- everything else about them is "
                "unaffected", False)

    hints = []
    if config != "loaded":
        hints.append(f"its settings are {config}")
    if third_party == "no":
        hints.append("its \"Active in third party textboxes\" setting is off "
                     "(changing it needs a full restart, not a UI reload)")
    if not found["listWrapped"]:
        hints.append("this extension could not extend the list it attaches from")
    elif not found["inTheirList"]:
        hints.append("the boxes are not in the list it attaches from")

    return ("Model Chain: Tag Autocomplete is installed but has not claimed the "
            "Literal Prompt boxes"
            + (" -- " + "; ".join(hints) if hints else "")
            + f" [config {config}, third-party boxes {third_party},"
            f" list extended {found['listWrapped']},"
            f" boxes in the list {found['inTheirList']},"
            f" row placed {found['placed']}]", True)


def note(report) -> str:
    """Log what the browser found, and hand the line back for the tests.

    A warning when something is wrong, and an ordinary line when nothing is:
    "this works here" is the other half of the answer somebody needs, and one
    line per page load is not a log to hide from.
    """
    line, wrong = describe(report)
    if wrong:
        logger.warning(line)
    else:
        logger.info(line)
    return line


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal.

    Signature is ``script_callbacks.on_app_started``'s -- ``(demo, app)`` -- and
    the app is the only half this uses. A build that hands over something
    without ``add_api_route`` is a build where the boxes still work and this
    line never appears in the log.
    """
    if app is None or not hasattr(app, "add_api_route"):
        return False

    for existing in getattr(app, "routes", []):
        if getattr(existing, "path", None) == ROUTE:
            return True

    def literal_prompt_report(report: dict):
        try:
            note(report)
        except Exception:
            logger.debug("Model Chain: could not log a Literal Prompt report",
                         exc_info=True)
        return {"logged": True}

    try:
        app.add_api_route(ROUTE, literal_prompt_report, methods=["POST"])
    except Exception:
        logger.debug("Model Chain: could not register the Literal Prompt report "
                     "route", exc_info=True)
        return False
    return True
