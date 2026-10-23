"""The "LoRA Details" top-level tab.

Two jobs: manage the CSV library, and run Civitai fetches on demand. Scans run on
a background worker so the rest of the WebUI stays usable; this tab only polls a
snapshot of their state.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

import mc_extra_networks_folders
import mc_extra_networks_folders_ui
import mc_lora_images
import mc_lora_jobs
import mc_lora_last_used
import mc_lora_render
from mc_lora_common import lora_directories, opt, report
from mc_lora_library import get_library

POLL_SECONDS = 1.5

TAB_LABEL = "LoRA Details"
TAB_ID = "model_chain_lora_details"


def _library_status() -> str:
    try:
        return mc_lora_render.library_summary_html(get_library().stats())
    except Exception:
        report("could not read the library status")
        return mc_lora_render.notice("The library could not be read — see the console for details.", kind="error")


def _sources_table() -> str:
    try:
        return mc_lora_render.sources_html(get_library().sources)
    except Exception:
        report("could not list the library sources")
        return ""


def _source_names() -> list[str]:
    try:
        return [source.name for source in get_library().sources]
    except Exception:
        return []


def _folder_choices(keep: str = "") -> list[str]:
    """Every LoRA root plus its immediate subfolders — the useful scan targets.

    ``keep`` is a folder that must stay selectable whatever the listing holds —
    the one a previous fetch ran on, say, which can sit deeper than this reaches
    or under a root that is no longer configured.
    """
    choices: list[str] = []
    for root in lora_directories():
        if not os.path.isdir(root):
            continue
        choices.append(root)
        try:
            for entry in sorted(os.scandir(root), key=lambda item: item.name.casefold()):
                if entry.is_dir() and not entry.name.startswith("."):
                    choices.append(entry.path)
        except OSError:
            continue

    wanted = str(keep or "").strip()
    if wanted and mc_extra_networks_folders.normalize_dir(wanted) not in {mc_extra_networks_folders.normalize_dir(path) for path in choices}:
        choices.insert(0, wanted)
    return choices


def _default_folder() -> str | None:
    """The folder the last fetch ran on, else the first LoRA root."""
    remembered = mc_lora_last_used.remembered_scan_folder()
    if remembered:
        return remembered
    return (_folder_choices() or [None])[0]


def _image_status() -> str:
    try:
        return mc_lora_render.image_library_html(mc_lora_images.stats())
    except Exception:
        report("could not read the image folder")
        return mc_lora_render.notice("The image folder could not be read — see the console for details.", kind="error")


def _jobs_html() -> str:
    try:
        return mc_lora_render.job_html(mc_lora_jobs.MANAGER.snapshots())
    except Exception:
        report("could not render the job list")
        return ""


def create_tab():
    """Build the tab; returns the value ``on_ui_tabs`` expects."""
    timer_available = hasattr(gr, "Timer")

    with gr.Blocks(analytics_enabled=False) as interface:
        # One root for the stylesheet to scope on, the way the Voice Box has #mc-voice-box.
        with gr.Column(elem_id="mc-lora-details", elem_classes=["mc-lora-tab"]):
            gr.HTML(
                mc_lora_render.notice(
                    "Rows from the CSV library are shown inside each LoRA card's "
                    "<b>edit metadata</b> dialog, which Model Chain draws the width of the window.",
                    kind="info",
                )
            )

            with gr.Tabs():
                # ------------------------------------------------------- library
                with gr.TabItem("Library"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            library_status = gr.HTML(value=_library_status)
                            sources_table = gr.HTML(value=_sources_table)
                        with gr.Column(scale=1):
                            upload = gr.File(
                                label="Add CSV files to the library",
                                file_count="multiple",
                                file_types=[".csv", "text/csv"],
                                type="filepath",
                            )
                            upload_status = gr.HTML()
                            reload_button = gr.Button("Reload library from disk", variant="primary")
                            with gr.Row():
                                source_picker = gr.Dropdown(
                                    label="Remove a CSV",
                                    choices=_source_names(),
                                    value=None,
                                    interactive=True,
                                )
                                remove_button = gr.Button("Remove", size="sm")

                # ------------------------------------------------- preset folders
                with gr.TabItem("Preset folders"):
                    try:
                        mc_extra_networks_folders_ui.create_ui()
                    except Exception:
                        report("could not build the preset folder editor")
                        gr.HTML(
                            mc_lora_render.notice(
                                "The folder editor could not be built — see the console for details.",
                                kind="error",
                            )
                        )

                # -------------------------------------------------------- images
                with gr.TabItem("Images"):
                    gr.HTML(
                        mc_lora_render.notice(
                            "Gallery images downloaded from a LoRA's details dialog "
                            "all land in one folder, so they can be managed in one place.",
                            kind="info",
                        )
                    )
                    image_status = gr.HTML(value=_image_status)
                    with gr.Row():
                        refresh_images = gr.Button("Refresh")
                        purge_images = gr.Button("Delete every downloaded image", variant="stop")
                    purge_confirm = gr.Checkbox(label="Yes, delete them", value=False)
                    image_message = gr.HTML()

                # ---------------------------------------------------- civitai
                with gr.TabItem("Fetch from Civitai"):
                    gr.HTML(
                        mc_lora_render.notice(
                            "Hashes every <code>.safetensors</code> under the chosen folder, resolves each one "
                            "through Civitai's <code>by-hash</code> endpoint, and writes the result into the "
                            "library as a CSV. Runs in the background — you can keep generating.<br>"
                            "Text is fetched by default: model name, creator, base model, trigger words, gallery "
                            "prompts and the model description. Gallery <b>images are never downloaded by a fetch</b> — "
                            "ask for those from a LoRA's details dialog when you want them.",
                            kind="info",
                        )
                    )
                    with gr.Row():
                        folder = gr.Dropdown(
                            label="Folder to scan",
                            choices=_folder_choices(mc_lora_last_used.remembered_scan_folder()),
                            value=_default_folder,
                            allow_custom_value=True,
                            interactive=True,
                            scale=4,
                        )
                        refresh_folders = gr.Button("↻", size="sm", scale=0, min_width=50)
                    with gr.Row():
                        output_name = gr.Textbox(
                            label="CSV file name",
                            placeholder="leave empty to name it after the folder",
                            scale=3,
                        )
                        recursive = gr.Checkbox(label="Include subfolders", value=True, scale=1)
                    with gr.Row():
                        incremental = gr.Checkbox(
                            label="Only fetch LoRAs the library does not already have",
                            value=True,
                            info="anything already recorded is skipped, including ones Civitai had nothing for",
                        )
                        keep_existing = gr.Checkbox(
                            label="Keep rows already in the target CSV",
                            value=True,
                            info="uncheck to rewrite the CSV from scratch",
                        )
                    retry_unresolved = gr.Checkbox(
                        label="Also retry LoRAs Civitai could not resolve last time",
                        value=bool(opt("model_chain_lora_retry_unresolved", False)),
                        info="use this after uploading models to Civitai, or when a previous run hit errors",
                    )
                    with gr.Row():
                        fetch_descriptions = gr.Checkbox(
                            label="Fetch model descriptions",
                            value=bool(opt("model_chain_lora_fetch_descriptions", True)),
                            info="the description text from the model's Civitai page, shown on the panel's Description tab",
                        )
                        backfill_text = gr.Checkbox(
                            label="Fill in text details missing from rows already in the library",
                            value=bool(opt("model_chain_lora_backfill_text", True)),
                            info="looks up LoRAs that resolved before but carry no description yet",
                        )

                    # Seeded from Settings → Model Chain: LoRA Details; overridable per run.
                    with gr.Accordion("Fetch options", open=False):
                        with gr.Row():
                            base_model_filter = gr.Textbox(
                                label="Only accept these base models",
                                value=str(opt("model_chain_lora_base_model_filter", "")),
                                placeholder='e.g. "Krea 2, MOSS Image" — empty accepts any exact hash match',
                            )
                            require_label = gr.Checkbox(
                                label="Require an explicit base-model label",
                                value=bool(opt("model_chain_lora_require_explicit_label", False)),
                            )
                        with gr.Row():
                            api_key = gr.Textbox(
                                label="Civitai API key",
                                type="password",
                                value=mc_lora_last_used.remembered_api_key,
                                placeholder="uses the setting when empty",
                                info="filled in with the key of the last fetch Civitai accepted",
                            )
                            max_images = gr.Slider(
                                label="Gallery images per version (0 = all)",
                                minimum=0,
                                maximum=500,
                                step=10,
                                value=int(opt("model_chain_lora_max_images", 0)),
                            )
                        with gr.Row():
                            request_delay = gr.Slider(
                                label="Delay between requests (s)",
                                minimum=0.0,
                                maximum=5.0,
                                step=0.05,
                                value=float(opt("model_chain_lora_request_delay", 0.15)),
                            )
                            request_timeout = gr.Slider(
                                label="Request timeout (s)",
                                minimum=5.0,
                                maximum=180.0,
                                step=5.0,
                                value=float(opt("model_chain_lora_request_timeout", 30.0)),
                            )

                    with gr.Row():
                        start_button = gr.Button("Start fetch", variant="primary")
                        prehash_button = gr.Button("Precompute hashes")
                        cancel_button = gr.Button("Cancel running job", variant="stop")
                    queue_status = gr.HTML()

                # ------------------------------------------------------- monitor
                with gr.TabItem("Jobs"):
                    with gr.Row():
                        poll = gr.Checkbox(label="Auto-refresh", value=True, scale=1)
                        refresh_jobs = gr.Button("Refresh now", size="sm", scale=1)
                        clear_jobs = gr.Button("Clear finished", size="sm", scale=1)
                    jobs_view = gr.HTML(value=_jobs_html, elem_classes=["mc-lora-jobs"])

            # ------------------------------------------------------------ events

            def do_upload(paths):
                if not paths:
                    return "", _library_status(), _sources_table(), gr.update(choices=_source_names()), None

                if not isinstance(paths, (list, tuple)):
                    paths = [paths]

                messages = []
                library = get_library()
                for entry in paths:
                    path = getattr(entry, "name", entry)
                    try:
                        ok, message = library.import_csv(path)
                    except Exception as exc:  # noqa: BLE001 - shown to the user
                        report("could not import a CSV file")
                        ok, message = False, f"{type(exc).__name__}: {exc}"
                    messages.append(mc_lora_render.notice(mc_lora_render.escape(message), kind="ok" if ok else "error"))

                return (
                    "".join(messages),
                    _library_status(),
                    _sources_table(),
                    gr.update(choices=_source_names()),
                    None,
                )

            upload.upload(
                fn=do_upload,
                inputs=[upload],
                outputs=[upload_status, library_status, sources_table, source_picker, upload],
            )

            def do_refresh_images():
                return _image_status(), ""

            refresh_images.click(fn=do_refresh_images, outputs=[image_status, image_message], show_progress=False)

            def do_purge_images(confirmed):
                if not confirmed:
                    return _image_status(), mc_lora_render.notice("Tick the confirmation box first.", kind="info"), gr.update()
                try:
                    _removed, message = mc_lora_images.purge()
                    note = mc_lora_render.notice(mc_lora_render.escape(message), kind="ok")
                except Exception as exc:  # noqa: BLE001
                    report("could not delete the downloaded images")
                    note = mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
                return _image_status(), note, gr.update(value=False)

            purge_images.click(
                fn=do_purge_images,
                inputs=[purge_confirm],
                outputs=[image_status, image_message, purge_confirm],
            )

            def do_reload():
                try:
                    get_library().reload()
                    message = mc_lora_render.notice("Library reloaded.", kind="ok")
                except Exception as exc:  # noqa: BLE001
                    report("could not reload the library")
                    message = mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
                return message, _library_status(), _sources_table(), gr.update(choices=_source_names())

            reload_button.click(
                fn=do_reload,
                outputs=[upload_status, library_status, sources_table, source_picker],
            )

            def do_remove(name):
                try:
                    ok, message = get_library().remove_csv(name)
                except Exception as exc:  # noqa: BLE001
                    report("could not remove a CSV")
                    ok, message = False, f"{type(exc).__name__}: {exc}"
                return (
                    mc_lora_render.notice(mc_lora_render.escape(message), kind="ok" if ok else "error"),
                    _library_status(),
                    _sources_table(),
                    gr.update(choices=_source_names(), value=None),
                )

            remove_button.click(
                fn=do_remove,
                inputs=[source_picker],
                outputs=[upload_status, library_status, sources_table, source_picker],
            )

            def do_refresh_folders(current):
                """Re-list the folders without losing the one that is selected."""
                current = str(current or "").strip()
                choices = _folder_choices(current)
                return gr.update(choices=choices, value=current or (choices[0] if choices else None))

            refresh_folders.click(
                fn=do_refresh_folders,
                inputs=[folder],
                outputs=[folder],
                show_progress=False,
            )

            def queue_result(message, kind, *, armed=False, poll_value=False):
                """Status + job list, plus the timer arm/disarm when gr.Timer exists."""
                payload = [mc_lora_render.notice(message, kind=kind), _jobs_html()]
                if timer_available:
                    payload.append(gr.update(active=bool(armed and poll_value)))
                return tuple(payload)

            def do_start(
                folder_value,
                output_value,
                recursive_value,
                incremental_value,
                retry_value,
                backfill_value,
                descriptions_value,
                keep_existing_value,
                base_model_value,
                require_label_value,
                api_key_value,
                max_images_value,
                delay_value,
                timeout_value,
                poll_value=False,
            ):
                def result(message, kind, armed=False):
                    return queue_result(message, kind, armed=armed, poll_value=poll_value)

                target = (folder_value or "").strip()
                if not target:
                    return result("Choose a folder first.", "error")
                if not Path(target).expanduser().is_dir():
                    return result(f"Not a folder: {mc_lora_render.escape(target)}", "error")

                key = (api_key_value or "").strip() or str(opt("model_chain_lora_civitai_api_key", "")).strip() or os.environ.get("CIVITAI_API_KEY", "")

                try:
                    job = mc_lora_jobs.submit_scan(
                        target,
                        output_name=output_value or "",
                        api_key=key,
                        request_delay=float(delay_value),
                        timeout=float(timeout_value),
                        max_images=int(max_images_value),
                        recursive=bool(recursive_value),
                        base_model_filter=(base_model_value or "").strip(),
                        require_explicit_label=bool(require_label_value),
                        incremental=bool(incremental_value),
                        retry_unresolved=bool(retry_value),
                        backfill_text=bool(backfill_value),
                        fetch_descriptions=bool(descriptions_value),
                        keep_existing_rows=bool(keep_existing_value),
                    )
                except Exception as exc:  # noqa: BLE001
                    report("could not queue the fetch")
                    return result(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), "error")

                note = "" if key else " No API key set — public coverage only."
                return result(
                    f"Queued job #{job.id} for <code>{mc_lora_render.escape(target)}</code>. "
                    f"Progress is on the <b>Jobs</b> tab.{mc_lora_render.escape(note)}",
                    "ok",
                    armed=True,
                )

            start_inputs = [
                folder,
                output_name,
                recursive,
                incremental,
                retry_unresolved,
                backfill_text,
                fetch_descriptions,
                keep_existing,
                base_model_filter,
                require_label,
                api_key,
                max_images,
                request_delay,
                request_timeout,
            ]

            def do_prehash(poll_value=False):
                def result(message, kind, armed=False):
                    return queue_result(message, kind, armed=armed, poll_value=poll_value)

                try:
                    job = mc_lora_jobs.submit_prehash()
                except Exception as exc:  # noqa: BLE001
                    report("could not queue hashing")
                    return result(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), "error")
                return result(
                    f"Queued job #{job.id}. Hashing every LoRA up front keeps the details dialog instant.",
                    "ok",
                    armed=True,
                )

            def do_cancel():
                count = mc_lora_jobs.MANAGER.cancel_all()
                kind = "ok" if count else "info"
                message = f"Cancelling {count} job(s)." if count else "Nothing is running."
                return mc_lora_render.notice(message, kind=kind), _jobs_html()

            cancel_button.click(fn=do_cancel, outputs=[queue_status, jobs_view])
            refresh_jobs.click(fn=_jobs_html, outputs=[jobs_view], show_progress=False)

            def do_clear():
                mc_lora_jobs.MANAGER.clear_finished()
                return _jobs_html()

            clear_jobs.click(fn=do_clear, outputs=[jobs_view], show_progress=False)

            # Auto-refresh only while something is running: the timer arms itself when
            # a job is queued and disarms on the first tick with an empty queue, so an
            # idle tab costs no traffic. gr.Timer is a standard component; on a Gradio
            # build without it the tab simply keeps its manual Refresh button.
            timer_class = getattr(gr, "Timer", None)
            if timer_class is not None:
                timer = timer_class(POLL_SECONDS, active=False)

                def on_tick():
                    if mc_lora_jobs.MANAGER.has_active():
                        return _jobs_html(), gr.update()
                    # One last refresh so the finished state is shown, then stop.
                    return _jobs_html(), gr.update(active=False)

                timer.tick(fn=on_tick, outputs=[jobs_view, timer], show_progress=False)

                poll.change(
                    fn=lambda enabled: gr.update(active=bool(enabled) and mc_lora_jobs.MANAGER.has_active()),
                    inputs=[poll],
                    outputs=[timer],
                    show_progress=False,
                )

                start_button.click(
                    fn=do_start,
                    inputs=[*start_inputs, poll],
                    outputs=[queue_status, jobs_view, timer],
                )
                prehash_button.click(fn=do_prehash, inputs=[poll], outputs=[queue_status, jobs_view, timer])
            else:
                poll.visible = False
                start_button.click(fn=do_start, inputs=start_inputs, outputs=[queue_status, jobs_view])
                prehash_button.click(fn=do_prehash, outputs=[queue_status, jobs_view])

    return [(interface, TAB_LABEL, TAB_ID)]


def on_ui_tabs() -> list:
    """The ``on_ui_tabs`` callback. Never raises into the host."""
    try:
        return create_tab()
    except Exception:
        report("the LoRA Details tab could not be built")
        return []
