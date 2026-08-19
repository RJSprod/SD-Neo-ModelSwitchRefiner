"""LLM Studio: one native Forge tab, three distinct workspaces (section 4.1).

This module is the shell. It builds the tab, switches between the three modes,
and owns the two things all of them share -- the residency status that makes
memory decisions visible (section 14), and the Models, hardware and memory
panel where the model, the placement and the context budget are chosen
(sections 6, 11 and 12).

What it deliberately does not do is host the modes' logic. Prompt Studio,
Conversation and MiniMax are built by three separate modules and share no state
beyond the preferences file, which is section 4.1's requirement that the modes
"may reuse shared panels" but "must not be collapsed into a single generic chat
workflow" enforced at the level of the source tree.

Failure is a first-class state here. Section 18 requires that a failure to
start or load the LLM must not poison image generation, and the way that is
guaranteed is that nothing below runs at import time and every entry point is
wrapped: if the vendored package will not import, if Pillow is missing, if the
data directory is unwritable, the tab renders an explanation and the rest of
the WebUI never knows.
"""

from __future__ import annotations

import logging

import gradio as gr

import mc_broker
import mc_llm_paths
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3

TAB_LABEL = "LLM Studio"
TAB_ID = "llm_studio"

OPT_ENABLE = "model_chain_llm_studio"

MODES = (
    ("Prompt Studio", "prompt"),
    ("Conversation", "chat"),
    ("MiniMax H3", "minimax"),
)


def enabled() -> bool:
    """Whether the tab should be built at all.

    A setting rather than an assumption: an installation that only wants the
    image half of this extension should not carry an LLM tab it will never
    open, and section 18 asks for ordinary generation to be unaffected when
    LLM Studio is never used.
    """
    return bool(mc_broker.option(OPT_ENABLE, True))


def on_ui_tabs():
    """The ``on_ui_tabs`` callback. Never raises into the host."""
    if not enabled():
        return []
    try:
        return [(_build(), TAB_LABEL, TAB_ID)]
    except Exception:
        logger.warning("Model Chain: LLM Studio could not be built", exc_info=True)
        try:
            return [(_unavailable(), TAB_LABEL, TAB_ID)]
        except Exception:
            # Two failures means the tab cannot be drawn at all. Returning
            # nothing costs the feature; raising would cost the WebUI its UI.
            return []


def _unavailable():
    with gr.Blocks(analytics_enabled=False) as block:
        gr.Markdown(
            "### LLM Studio could not start\n\n"
            "The panel failed to build. The console holds the traceback.\n\n"
            "Image generation and Model Chain are unaffected."
        )
    return block


def _build():
    import mc_llm_chat_panel
    import mc_llm_minimax_panel
    import mc_llm_prompt_panel

    with gr.Blocks(analytics_enabled=False) as block:
        with gr.Column(elem_id=ui.ident("studio"), elem_classes=ui.classes("studio")):

            with gr.Row(elem_classes=ui.classes("topbar")):
                with gr.Column(scale=2, min_width=280):
                    mode = gr.Radio(
                        label=None, show_label=False, choices=list(MODES), value=_initial_mode(),
                        elem_id=ui.ident("mode"), elem_classes=ui.classes("modes"))
                with gr.Column(scale=3):
                    runtime_status = gr.HTML(_runtime_line(),
                                             elem_id=ui.ident("runtime", "status"))

            with gr.Column(visible=True, elem_classes=ui.classes("mode-view")) as prompt_view:
                mc_llm_prompt_panel.build()
            with gr.Column(visible=False, elem_classes=ui.classes("mode-view")) as chat_view:
                mc_llm_chat_panel.build()
            with gr.Column(visible=False, elem_classes=ui.classes("mode-view")) as minimax_view:
                mc_llm_minimax_panel.build()

            with gr.Accordion("Models, hardware and memory", open=False,
                              elem_id=ui.ident("settings")):
                settings = _settings_panel()

        views = [prompt_view, chat_view, minimax_view]
        mode.change(fn=_switch, inputs=[mode], outputs=views + [runtime_status], queue=False)
        block.load(fn=lambda: (_runtime_line(), _residency_html()),
                   outputs=[runtime_status, settings["residency"]], queue=False)

    return block


def _initial_mode() -> str:
    import mc_llm_state

    stored = mc_llm_state.preferences().get("mode", "prompt")
    return stored if stored in [value for _, value in MODES] else "prompt"


def _switch(chosen):
    """Show one workspace. The other two are hidden, not rebuilt.

    Rebuilding would lose whatever was on screen -- a half-read reply, a prompt
    someone is editing -- every time the selector moved, which is the one thing
    a mode switch must not do.
    """
    import mc_llm_state

    mc_llm_state.remember(mode=chosen)
    return [gr.update(visible=(chosen == value)) for _, value in MODES] + [_runtime_line()]


# --------------------------------------------------------------------------- #
# Status (section 14)
# --------------------------------------------------------------------------- #


def _runtime_line() -> str:
    """The concise status. Detail belongs in the collapsible panel below it."""
    try:
        import mc_llm_runtime

        state = mc_llm_runtime.runtime.status()
    except Exception:
        return ui.notice("LLM runtime unavailable — see the console.", "error")

    if not state["configured"]:
        return ui.notice("No model configured yet — open Models, hardware and memory.", "warn")

    parts = [f"Model: {state['quantization'] or state['model'] or 'unknown'}"]
    parts.append(f"Device: {state['device'] or 'unknown'}")
    parts.append("Server: running" if state["running"] else "Server: stopped")
    if state["running"] and state["resident_bytes"]:
        parts.append(f"VRAM: {ui.gigabytes(state['resident_bytes'])}")
    report = state["report"]
    if report.placement is not None:
        parts.append(f"Context: {ui.tokens(report.placement.context)}")
    if not state["sees"]:
        parts.append("No vision projector")
    return ui.notice(" · ".join(parts))


def _residency_html() -> str:
    """The detailed residency view (section 14), kept out of the main UI."""
    try:
        import mc_llm_runtime

        status = mc_broker.status()
        state = mc_llm_runtime.runtime.status()
    except Exception:
        return ui.notice("Residency information unavailable.", "warn")

    rows = []
    for entry in status.residencies:
        rows.append(
            f"<tr><td>{ui.escape(entry.label)}</td>"
            f"<td>{ui.escape(entry.family)}</td>"
            f"<td>{ui.gigabytes(entry.bytes)}</td>"
            f"<td>{ui.escape(mc_broker.RANK_LABELS.get(entry.effective_rank, '?'))}</td></tr>")
    if not rows:
        rows.append('<tr><td colspan="4">Nothing is registered as VRAM-resident.</td></tr>')

    running = status.active
    owners = ", ".join(status.owners) or "nothing"
    summary = [
        f"<li>Mode: <b>{ui.escape(mc_broker.label_for(mc_broker.MODES, status.mode))}</b></li>",
        f"<li>Policy: <b>{ui.escape(mc_broker.label_for(mc_broker.POLICIES, status.policy))}</b></li>",
        f"<li>VRAM: {ui.gigabytes(status.free_vram)} free of "
        f"{ui.gigabytes(status.total_vram)}, {ui.gigabytes(status.reserve)} reserved</li>",
        f"<li>VRAM owners: {ui.escape(owners)}</li>",
        f"<li>Active workload: {ui.escape(running.label) if running else 'none'}</li>",
    ]
    report = state.get("report")
    if report is not None and report.placement is not None:
        summary.append(
            f"<li>LLM placement: {ui.escape(report.placement.describe())}, "
            f"{ui.tokens(report.placement.context)} token context</li>")
    for text in (report.notes if report is not None else ()):
        summary.append(f"<li>Reported change: {ui.escape(text)}</li>")

    decisions = "".join(f"<li>{ui.escape(entry.text)}</li>"
                        for entry in reversed(mc_broker.decisions(8)))
    return (
        f'<div class="{ui.PREFIX}-residency">'
        f'<ul>{"".join(summary)}</ul>'
        f'<table class="{ui.PREFIX}-table"><thead><tr><th>Resident</th><th>Family</th>'
        f'<th>VRAM</th><th>Rank</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<h4>Recent decisions</h4><ul>{decisions or "<li>No decisions yet.</li>"}</ul>'
        f'</div>'
    )


# --------------------------------------------------------------------------- #
# Models, hardware and memory
# --------------------------------------------------------------------------- #


def _settings_panel() -> dict:
    """Everything about what runs and where, in one collapsible place."""
    import mc_llm_context
    import mc_llm_runtime
    import mc_llm_state

    configuration = mc_llm_runtime.config()
    prefs = mc_llm_state.preferences()

    with gr.Row():
        with gr.Column(scale=2, min_width=320):
            gr.Markdown("#### Which model runs")
            model_path = gr.Textbox(
                label="GGUF model", value=str(configuration.model or ""),
                placeholder="Full path to a .gguf file",
                elem_id=ui.ident("settings", "model"))
            mmproj_path = gr.Textbox(
                label="Vision projector (optional)", value=str(configuration.mmproj or ""),
                placeholder="Full path to an mmproj .gguf, or empty for a text-only model",
                elem_id=ui.ident("settings", "mmproj"))
            with gr.Row():
                apply_model = gr.Button("Use this model", variant="primary", size="sm")
                suggest = gr.Button("Find the projector beside it", size="sm")
            model_notice = gr.HTML(_model_line(configuration))

            gr.Markdown("#### Context and VRAM buffer")
            context_mode = gr.Radio(
                label="Context sizing", value=prefs.get("context_mode", "auto"),
                choices=[("Automatic — fill what is free", "auto"),
                         ("Fixed buffer", "fixed")],
                elem_id=ui.ident("settings", "context-mode"))
            buffer_gb = gr.Slider(
                label="Context / VRAM buffer (GB)", minimum=0.5, maximum=48, step=0.5,
                value=float(prefs.get("context_buffer_gb", 4.0)),
                info="Memory budgeted for the key/value cache. Separate from the weights and "
                     "from the runtime reserve.")
            # A number rather than a slider: modern ceilings run to a million
            # tokens, and a slider across that range cannot be aimed at 32,768.
            context_size = gr.Number(
                label="Context size (tokens)", value=int(prefs.get("context_size", 8192)),
                precision=0, minimum=mc_llm_runtime.MINIMUM_CONTEXT,
                info="Used when sizing is Fixed. Never exceeds the model's own ceiling.")
            with gr.Row():
                kv_k = gr.Dropdown(label="K cache type",
                                   choices=[(label, name) for name, label
                                            in mc_llm_context.KV_TYPE_LABELS],
                                   value=prefs.get("kv_type_k", "f16"))
                kv_v = gr.Dropdown(label="V cache type",
                                   choices=[(label, name) for name, label
                                            in mc_llm_context.KV_TYPE_LABELS],
                                   value=prefs.get("kv_type_v", "f16"))
            save_context = gr.Button("Save context settings", variant="primary", size="sm")

        with gr.Column(scale=3, min_width=360):
            gr.Markdown("#### What fits")
            estimator = gr.HTML(_estimator_html(), elem_id=ui.ident("settings", "estimator"))
            estimate_now = gr.Button("Re-estimate", size="sm")

            gr.Markdown("#### Memory policy")
            # Labels rather than values, and the same labels the Settings page
            # offers: these controls write straight into shared.opts, so what
            # they store has to be a string that page can also display.
            memory_mode = gr.Radio(
                label="Residency mode",
                choices=[label for _, label in mc_broker.MODES],
                value=mc_broker.label_for(mc_broker.MODES, mc_broker.mode()),
                elem_id=ui.ident("settings", "memory-mode"))
            hybrid_policy = gr.Radio(
                label="When it does not all fit",
                choices=[label for _, label in mc_broker.POLICIES],
                value=mc_broker.label_for(mc_broker.POLICIES, mc_broker.policy()),
                elem_id=ui.ident("settings", "policy"))
            release_mode = gr.Radio(
                label="When the image side needs the VRAM back",
                choices=[label for _, label in mc_llm_runtime.RELEASE_MODES],
                value=mc_broker.label_for(
                    mc_llm_runtime.RELEASE_MODES,
                    mc_broker.resolve(mc_broker.option(mc_llm_runtime.OPT_RELEASE,
                                                       mc_llm_runtime.RELEASE_STOP),
                                      mc_llm_runtime.RELEASE_MODES,
                                      mc_llm_runtime.RELEASE_STOP)),
                elem_id=ui.ident("settings", "release"))

            gr.Markdown("#### Residency")
            residency = gr.HTML(_residency_html(), elem_id=ui.ident("settings", "residency"))
            with gr.Row():
                refresh_residency = gr.Button("Refresh", size="sm")
                stop_server = gr.Button("Stop llama-server", size="sm", variant="stop")

    # -- wiring ----------------------------------------------------------- #

    apply_model.click(fn=_apply_model, inputs=[model_path, mmproj_path],
                      outputs=[model_notice, estimator], queue=False)
    suggest.click(fn=_suggest_projector, inputs=[model_path],
                  outputs=[mmproj_path, model_notice], queue=False)

    save_context.click(fn=_save_context,
                       inputs=[context_mode, buffer_gb, context_size, kv_k, kv_v],
                       outputs=[estimator], queue=False)
    estimate_now.click(fn=lambda: _estimator_html(), outputs=[estimator], queue=False)

    for control, name in ((memory_mode, mc_broker.OPT_MODE),
                          (hybrid_policy, mc_broker.OPT_POLICY),
                          (release_mode, mc_llm_runtime.OPT_RELEASE)):
        control.change(fn=_setter(name), inputs=[control], outputs=[residency], queue=False)

    refresh_residency.click(fn=_residency_html, outputs=[residency], queue=False)
    stop_server.click(fn=_stop_server, outputs=[residency], queue=False)

    return {"residency": residency, "estimator": estimator}


def _model_line(configuration) -> str:
    if not configuration.configured:
        root = mc_llm_paths.data_root()
        return ui.notice(
            f"No runtime is provisioned yet. LLM Studio looks in {root}. Point "
            f"PROMPT_MASTER_ROOT or the Model Chain LLM data directory setting at an existing "
            f"Prompt Master install to reuse it, or run its console setup to provision one.",
            "warn")
    return ui.notice(
        f"{configuration.model.name if configuration.model else ''} · "
        f"{'vision projector loaded' if configuration.sees else 'text only'}")


def _apply_model(model, mmproj):
    """Point the install at a different GGUF, without re-provisioning anything."""
    from pathlib import Path

    import mc_llm_runtime
    from prompt_master.inference import model_choice

    if not (model or "").strip():
        return ui.notice("Enter the path to a .gguf file.", "warn"), gr.update()
    try:
        model_choice.choose(mc_llm_paths.app_paths(), Path(model.strip()),
                            Path(mmproj.strip()) if (mmproj or "").strip() else None)
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error"), gr.update()

    # The running server holds the weights it was started with, so it is
    # stopped rather than left to answer as the previous model.
    mc_llm_runtime.runtime.stop()
    return _model_line(mc_llm_runtime.config()), _estimator_html()


def _suggest_projector(model):
    from pathlib import Path

    from prompt_master.inference import model_choice

    if not (model or "").strip():
        return gr.update(), ui.notice("Enter the model path first.", "warn")
    found = model_choice.projector_beside(Path(model.strip()))
    if found is None:
        return gr.update(), ui.notice("No projector was found beside that model.", "warn")
    return str(found), ui.notice(f"Suggested {found.name} — check it belongs to this model.")


def _save_context(context_mode, buffer_gb, context_size, kv_k, kv_v):
    import mc_llm_state

    mc_llm_state.remember(context_mode=context_mode, context_buffer_gb=float(buffer_gb),
                          context_size=int(context_size), kv_type_k=kv_k, kv_type_v=kv_v)
    return _estimator_html()


def _setter(name: str):
    """A change handler that writes one Forge setting and repaints residency."""
    def apply(value):
        try:
            from modules import shared

            shared.opts.set(name, value)
            shared.opts.save(shared.config_filename)
        except Exception:
            logger.debug("Model Chain: could not persist %s", name, exc_info=True)
        return _residency_html()

    return apply


def _stop_server():
    import mc_llm_runtime

    mc_llm_runtime.runtime.stop()
    return _residency_html()


# --------------------------------------------------------------------------- #
# The estimator panel (section 12)
# --------------------------------------------------------------------------- #


def _estimator_html() -> str:
    """Section 12's table, plus the two hybrid answers it asks for.

    Everything shown is per model. When the header cannot be read the panel
    says so rather than filling the table with a constant.
    """
    import mc_gguf
    import mc_llm_context
    import mc_llm_runtime

    configuration = mc_llm_runtime.config()
    if not configuration.model:
        return ui.notice("Choose a GGUF to see what context fits.", "warn")

    described = mc_gguf.describe(configuration.model)
    if described is None or not described.usable:
        return ui.notice(
            f"{configuration.model.name} does not describe its attention shape in its GGUF "
            f"header, so context capacity cannot be estimated for it. The context size you set "
            f"is still used.", "warn")

    try:
        # reclaim=False: this panel is drawn when the tab is built and whenever
        # the accordion opens, and drawing a table is not a reason to evict
        # anybody's checkpoint.
        negotiated = mc_llm_runtime.negotiate(configuration, described, reclaim=False)
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error")

    placement = negotiated.placement
    estimate = negotiated.estimate
    per_token = estimate.kv_bytes_per_token
    reserve = mc_broker.safety_margin_bytes()
    free = mc_broker.free_vram_bytes()
    image_resident = mc_broker.resident_bytes(mc_broker.FAMILY_IMAGE)

    rows = []
    for found in mc_llm_context.table(configuration.model, placement):
        capped = " (model ceiling)" if found.limited_by_model else ""
        rows.append(
            f"<tr><td>{ui.gigabytes(found.budget_bytes)}</td>"
            f"<td>{ui.tokens(found.theoretical)}</td>"
            f"<td>{ui.tokens(found.usable)}{capped}</td></tr>")

    # The two questions section 12 asks the same estimator to answer.
    keeping = mc_llm_context.automatic_buffer_bytes(free, estimate.weights_bytes,
                                                    reserve + estimate.compute_bytes)
    moving = mc_llm_context.automatic_buffer_bytes(free + image_resident, estimate.weights_bytes,
                                                   reserve + estimate.compute_bytes)
    with_image = mc_llm_context.capacity(configuration.model, placement, keeping, gguf=described)
    without_image = mc_llm_context.capacity(configuration.model, placement, moving, gguf=described)

    facts = [
        f"<li>Model ceiling: <b>{ui.tokens(described.context_length)}</b> tokens</li>",
        f"<li>Current context: <b>{ui.tokens(placement.context)}</b> tokens "
        f"({ui.gigabytes(estimate.kv_bytes)} of key/value cache)</li>",
        f"<li>Cost per token: {per_token:,.0f} bytes "
        f"({described.block_count} blocks × {described.head_count_kv} KV heads)</li>",
        f"<li>Weights on the GPU: {ui.gigabytes(estimate.weights_bytes)} "
        f"({ui.escape(placement.describe(described.block_count))})</li>",
        f"<li>Runtime reserve: {ui.megabytes(estimate.compute_bytes)} — "
        f"<b>{'calibrated from a real load' if estimate.calibrated else 'estimated'}</b></li>",
        f"<li>Keeping the current image model resident: "
        f"<b>{ui.tokens(with_image.usable)}</b> tokens</li>",
        f"<li>If the image model is demoted to system RAM: "
        f"<b>{ui.tokens(without_image.usable)}</b> tokens</li>",
    ]
    if estimate.capped:
        facts.append("<li>Context is limited by the model's own ceiling, not by VRAM.</li>")
    if estimate.detail:
        facts.append(f"<li>{ui.escape(estimate.detail)}</li>")
    for text in negotiated.notes:
        facts.append(f"<li>Would be changed to fit: {ui.escape(text)}</li>")

    return (
        f'<div class="{ui.PREFIX}-estimator">'
        f'<ul>{"".join(facts)}</ul>'
        f'<table class="{ui.PREFIX}-table"><thead><tr><th>Context buffer</th>'
        f'<th>Theoretical tokens</th><th>Recommended</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'</div>'
    )
