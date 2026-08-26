"""The "Generation Memory & Persistent LLM" section on txt2img.

One place on the generation tab that answers the question every other part of
this extension has an opinion about and none of them shows: *where is the VRAM
going, and why is the language model the size it is?*

It is deliberately not called "Long Chain" and is not tied to Stage 2. The
whole point of :mod:`mc_plan` is that a generation is assembled from optional
pieces and the arithmetic is the same for all of them, so a section that only
appeared when the chain was armed would be describing a policy narrower than
the one in force. With every optional feature switched off it shows

    Active plan: Stage 1 (krea2)

which is a true and useful thing to say.

Reading, not deciding
---------------------
Nothing here changes a placement. Every figure is read from
:func:`mc_plan.budget` and :meth:`mc_llm_runtime.Runtime.status`, both of which
are snapshots, and the refresh button re-reads them. That matters because the
panel is on the same tab as the Generate button: a status display that
negotiated a placement in order to show one would restart llama-server every
time somebody opened an accordion, which is the exact failure this whole
change set exists to remove.

The three numbers the section must never conflate
-------------------------------------------------
Section 22 of the design intent, and it is worth restating because they are
routinely three different figures:

* the **calculated allowance** -- what the plan says is spare;
* the **actual observed residency** -- what llama-server really took, measured
  either side of its start;
* the **configured cap** -- the ceiling the user set, if they set one.

A placement that fitted its allowance can still have landed short of it, and a
user looking at one number cannot tell which.
"""

from __future__ import annotations

import logging

import mc_plan

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3

TITLE = "Generation Memory & Persistent LLM"

INACTIVE = "_inactive_"
"""How a phase that will not run is shown.

Never as ``0.0 GB``. A zero that looks like a measurement is worse than no
number at all -- section 21 -- because it invites the reader to conclude that
Stage 2 is free rather than absent.
"""


def gigabytes(value) -> str:
    try:
        return f"{max(int(value), 0) / _GB:.1f} GB"
    except (TypeError, ValueError):
        return "—"


def _row(label: str, value: str, note: str = "") -> str:
    return f"| {label} | {value} | {note} |"


# --------------------------------------------------------------------------- #
# The image half
# --------------------------------------------------------------------------- #


def plan_view(budget: mc_plan.Budget | None = None) -> str:
    """The active plan and every phase peak that goes into the reserve."""
    budget = budget if budget is not None else mc_plan.budget()
    plan = budget.plan

    if plan is None:
        return (
            "**Active plan:** none yet — press Generate once and this fills in.\n\n"
            "Until then the language model is placed against whatever VRAM happens to "
            "be free, which is the behaviour this section exists to replace."
        )

    limiting = budget.limiting
    size = ""
    if plan.width and plan.height:
        size = f"{plan.width}x{plan.height}"
        if plan.batch > 1:
            size += f", batch of {plan.batch}"
    lines = [
        f"**Active plan:** {plan.describe()}",
        "",
        "| | | |",
        "| --- | --- | --- |",
        _row("Usable VRAM", gigabytes(budget.total_bytes),
             "what the card can really give, not its nameplate"),
    ]
    if size:
        lines.append(_row("Sampling", size,
                          "a batch multiplies the activations, not the weights"
                          if plan.batch > 1 else ""))

    for phase in plan.phases:
        if not phase.holds_image_vram:
            # A preparation phase holds no image residency of its own. Shown so
            # the plan reads as a whole, with a dash rather than a zero.
            lines.append(_row(phase.label, "—", "no image residency"))
            continue
        note = phase.detail
        # Measured or not, said on every image row. A user watching Task Manager
        # disagree with this table is owed the reason, and the reason is almost
        # always that nothing has loaded yet: a file size plus a fixed overhead
        # is a starting heuristic, and on a quantised, mixed-precision
        # checkpoint it over-read by 3.6 GB of a 24 GB card.
        note = (note + "; " if note else "") + (
            "measured" if phase.measured else "estimated, not yet loaded once")
        if limiting is not None and phase.name == limiting.name:
            note += "; **sets the protected peak**"
        lines.append(_row(phase.label, gigabytes(phase.peak_bytes), note))

    lines.append(_row("Image working peak", gigabytes(budget.working_peak_bytes),
                      "the largest phase, not the sum of them"))
    if limiting is not None and not limiting.measured:
        lines.append(_row("", "",
                          "_generate once and this becomes a measurement rather than "
                          "an estimate_"))
    if budget.user_safety_bytes:
        lines.append(_row("Your safety adjustment", gigabytes(budget.user_safety_bytes),
                          "added on top of the automatic reserve"))
    lines.append(_row("Global safety headroom", gigabytes(budget.safety_bytes),
                      "already inside each phase peak above"))
    lines.append(_row("**Image-protected budget**", f"**{gigabytes(budget.protected_bytes)}**",
                      "kept clear whatever else asks for it"))
    return "\n".join(lines)


def residency_view() -> str:
    """Where the card's memory actually is, right now, adding up to the whole card.

    Added because a user watched Task Manager report 20.1 GB on an idle machine
    whose three model files come to 17.3 GB, and had no way to find the rest.
    The rest was real and entirely ordinary — a text encoder that loads larger
    than its file, a CUDA context that never comes back before the process
    exits, and a language model holding what the plan had given it — but none
    of it was visible anywhere.

    The last row is the one worth having. It is the card, minus what is free,
    minus everything this extension can account for, and it is where a stray
    llama-server from a killed WebUI, another program on the same GPU, or the
    CUDA context itself shows up. A number nobody can explain is exactly the
    number somebody needs to see.
    """
    try:
        import mc_broker

        total = int(mc_broker.total_vram_bytes())
        free = int(mc_broker.free_vram_bytes())
        image = int(mc_broker.held_bytes(mc_broker.FAMILY_IMAGE))
        llm = int(mc_broker.held_bytes(mc_broker.FAMILY_LLM))
    except Exception:
        logger.debug("Model Chain: could not read the residency map", exc_info=True)
        return ""

    if total <= 0:
        return ""

    rest = max(total - free - image - llm, 0)
    lines = [
        "**On the card right now**",
        "",
        "| | | |",
        "| --- | --- | --- |",
        _row("Image models", gigabytes(image), "weights as loaded, not as stored on disk"),
        _row("Language model", gigabytes(llm), "0 when it is running from system RAM"),
        _row("Free", gigabytes(free), ""),
        _row("Everything else", gigabytes(rest),
             "CUDA context, the desktop, other programs — "
             "a CUDA context alone is over a gigabyte and never returns "
             "until the WebUI exits"),
        _row("**Card total**", f"**{gigabytes(total)}**", ""),
    ]
    return "\n".join(lines)


def _absent_phases(plan) -> str:
    """The optional phases this configuration is *not* running, named.

    A reserve is easiest to trust when the reader can see what is missing from
    it, and hardest when a feature they thought was on quietly is not.
    """
    known = (
        (mc_plan.CREATIVE_WRITER, "Creative Writer"),
        (mc_plan.SPATIAL_COMPOSER, "Spatial Composer"),
        (mc_plan.STAGE_2, "Stage 2"),
    )
    missing = [label for name, label in known if plan is None or not plan.has(name)]
    if not missing:
        return ""
    return f"Not in this plan: {', '.join(missing)}."


# --------------------------------------------------------------------------- #
# The LLM half
# --------------------------------------------------------------------------- #


def _placement_view(placement, blocks: int = 0) -> list[str]:
    if placement is None:
        return []
    rows = []
    try:
        rows.append(_row("Placement", placement.describe(blocks) if blocks
                         else placement.describe()))
        rows.append(_row("Context", f"{int(placement.context):,} tokens"))
    except Exception:
        logger.debug("Model Chain: could not describe the LLM placement", exc_info=True)
    return rows


def llm_view(budget: mc_plan.Budget | None = None) -> str:
    """Where the language model is, what it took, and whether that matches the plan."""
    budget = budget if budget is not None else mc_plan.budget()

    if budget.llm_cap_mode == mc_plan.CAP_OFF:
        return ("**Persistent LLM:** Off — no residency is kept between requests, and the "
                "whole image arena belongs to the plan above.")

    try:
        import mc_llm_runtime

        status = mc_llm_runtime.runtime.status()
        prompt_rate, reply_rate = mc_llm_runtime.runtime.speed()
    except Exception:
        logger.debug("Model Chain: could not read the LLM runtime status", exc_info=True)
        return "**Persistent LLM:** the runtime could not be read."

    if not status.get("configured"):
        return ("**Persistent LLM:** no local model is configured. Choose a GGUF in LLM "
                "Studio’s Setup mode and this fills in.")

    mode = "Custom" if budget.llm_cap_mode == mc_plan.CAP_CUSTOM else "Auto"
    running = bool(status.get("running"))
    report = status.get("report")
    observed = int(getattr(report, "observed_bytes", 0) or 0)

    lines = [
        f"**Persistent LLM:** {status.get('quantization') or status.get('model') or 'configured'}"
        f" — {'Ready' if running else 'not running'}",
        "",
        "| | | |",
        "| --- | --- | --- |",
        _row("Mode", mode),
        _row("Calculated allowance", gigabytes(budget.llm_allowance_bytes),
             "what the plan above leaves over"),
    ]
    if budget.llm_custom_cap_bytes:
        lines.append(_row("Your cap", gigabytes(budget.llm_custom_cap_bytes),
                          "a lower ceiling than the allowance"))
    if budget.llm_learned_cap_bytes:
        lines.append(_row("Learned ceiling", gigabytes(budget.llm_learned_cap_bytes),
                          "from a previous reserve miss; not promoted back automatically"))
    lines.append(_row("Observed residency", gigabytes(observed) if observed else INACTIVE,
                      "measured either side of the start"))
    lines.extend(_placement_view(status.get("placement")))
    if prompt_rate or reply_rate:
        lines.append(_row("Measured speed",
                          f"{prompt_rate:.0f} / {reply_rate:.0f} tok/s",
                          "prompt evaluation / generation"))

    matches = _placement_matches_plan(running)
    lines.append(_row("Matches the active plan", "yes" if matches else "not yet",
                      "" if matches else "the next request re-places it"))
    return "\n".join(lines)


def _placement_matches_plan(running: bool) -> bool:
    """Whether the running server was placed for the plan that is now in force."""
    if not running:
        return False
    if mc_plan.current() is None:
        return True
    return mc_plan.placed_for() is not None and not mc_plan.boundary_moved()


# --------------------------------------------------------------------------- #
# Reserve misses
# --------------------------------------------------------------------------- #


def miss_view() -> str:
    """The last reserve miss, or a sentence saying there has not been one.

    Never silent about a miss and never alarming about the absence of one. An
    emergency eviction is recovery rather than scheduling, and a panel that
    showed nothing after one would leave a user with a slow language model and
    no idea why it got slower.
    """
    miss = mc_plan.last_miss()
    if miss is None:
        return "No reserve miss this session — every phase has fitted its estimate."
    advice = ""
    if miss.suggested_bytes > 0:
        advice = (f" Set **Custom** to about {gigabytes(miss.suggested_bytes)} in Settings if "
                  "this repeats.")
    return f"⚠ {miss.describe()}{advice}"


# --------------------------------------------------------------------------- #
# The whole section
# --------------------------------------------------------------------------- #


def report() -> str:
    """Everything above, as one Markdown block. Never raises.

    One function because the panel refreshes as a unit: three components that
    could each fail separately would be three ways to show half a memory
    contract, and half a memory contract is worse than none.
    """
    try:
        budget = mc_plan.budget()
        parts = [plan_view(budget)]
        absent = _absent_phases(budget.plan)
        if absent:
            parts.append(f"_{absent}_")
        parts.append(llm_view(budget))
        residency = residency_view()
        if residency:
            parts.append(residency)
        parts.append(miss_view())
        return "\n\n".join(parts)
    except Exception:
        logger.debug("Model Chain: could not build the memory report", exc_info=True)
        return "The generation memory report could not be built — see the console."


def build(elem_id):
    """The accordion, its refresh button and its handler.

    ``elem_id`` is the owning script's own namer, so the ids stay inside the
    script that built them. Returns the refresh button and the markdown block
    so the caller can wire anything else that should refresh them.
    """
    import gradio as gr

    import mc_pipeline_panel
    from modules.ui_components import ToolButton
    from modules.ui_common import refresh_symbol

    with mc_pipeline_panel.drawer(TITLE, elem_id=elem_id("memory")):
        gr.Markdown(
            "The plan for the next generation, the VRAM it is protected for, and where "
            "the language model was placed in what is left. Mutually exclusive phases — "
            "Stage 1 and Stage 2 — share the same VRAM rather than adding up, so the "
            "reserve is the **largest** phase and not the sum."
        )
        view = gr.Markdown(report(), elem_id=elem_id("memory_report"))
        refresh = ToolButton(value=refresh_symbol, elem_id=elem_id("memory_refresh"),
                             tooltip="Generation memory: re-read")
        refresh.click(fn=report, inputs=[], outputs=[view], show_progress=False)
    return refresh, view
