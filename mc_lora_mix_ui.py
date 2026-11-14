"""The LoRA Details tab's **Mix** page: two LoRAs, their strengths, a folder
chosen every time, a name, and the mixes made here.

The two pickers are Gradio dropdowns that filter as you type, listing every
safetensors LoRA Forge knows by its place under the LoRA folder; the folder
picker lists the LoRA folder and every folder under it, by the same kind of
label, and starts empty on every load and again after every mix, so the folder
is chosen each time ("I should be forced to choose output folder everytime").
The name box shows the default as its placeholder once both LoRAs are chosen.
Everything the server is sent back is a label it looks up afresh
(``mc_lora_mix.check``), never a path from the page.

The mix runs on the LoRA jobs' runner (``mc_lora_mix.submit``), so the page
stays usable; a Timer, armed only while that job runs, puts its answer on the
page and the new mix in the list. The handlers are module functions so the
tests can drive them without a page.
"""

from __future__ import annotations

import gradio as gr

import mc_lora_jobs
import mc_lora_mix
import mc_lora_render
from mc_lora_common import report

POLL_SECONDS = 1.0


def lora_choices() -> list[tuple[str, str]]:
    try:
        return mc_lora_mix.loras()
    except Exception:
        report("could not list the LoRAs for the mixer")
        return []


def folder_choices() -> list[str]:
    try:
        return [label for label, _path in mc_lora_mix.folders()]
    except Exception:
        report("could not list the LoRA folders for the mixer")
        return []


def mixes_html() -> str:
    try:
        mc_lora_mix.sweep_placeholders()
        return mc_lora_render.mixes_html(mc_lora_mix.mixes())
    except Exception:
        report("could not list the LoRA mixes")
        return mc_lora_render.notice("The mixes made here could not be listed — see the console.", kind="error")


def suggest_name(first, second):
    """The name box's placeholder: the default once both LoRAs are chosen."""
    paths = [mc_lora_mix.lora_file(str(value)) if value else None for value in (first, second)]
    if all(path is not None for path in paths):
        return gr.update(placeholder=mc_lora_mix.default_name(paths[0].stem, paths[1].stem))
    return gr.update(placeholder="mix_<first>_<second> when left empty")


def do_refresh():
    """Forge's LoRA list read again from disk, then both pickers and the folders."""
    networks = mc_lora_mix.forge_networks()
    if networks is not None and hasattr(networks, "list_available_networks"):
        try:
            networks.list_available_networks()
        except Exception:
            report("could not read the LoRA folders again")
    choices = lora_choices()
    return (gr.update(choices=choices), gr.update(choices=choices),
            gr.update(choices=folder_choices(), value=None), mixes_html())


def do_mix(first, first_strength, second, second_strength, folder, name):
    """Queue the mix, or say why not. Returns the status, the folder picker
    (emptied once a mix is queued), the job's id and whether to poll."""
    try:
        job = mc_lora_mix.submit(first, first_strength, second, second_strength, folder, name)
    except mc_lora_mix.MixError as exc:
        return (mc_lora_render.notice(mc_lora_render.escape(f"Not mixed: {exc}."), kind="error"),
                gr.update(), 0, False)
    except Exception:
        report("could not queue a LoRA mix")
        return (mc_lora_render.notice("The mix could not be queued — see the console.", kind="error"),
                gr.update(), 0, False)
    return (mc_lora_render.notice(f"Mixing as job #{job.id}…", kind="info"),
            gr.update(value=None), job.id, True)


def job_answer(job_id):
    """The mix's answer once its job is over: ``(status html, done)``."""
    job = mc_lora_jobs.MANAGER.get(int(job_id or 0)) if job_id else None
    if job is None:
        return "", True
    snapshot = job.snapshot()
    if snapshot.active:
        return mc_lora_render.notice(mc_lora_render.escape(snapshot.message or "Mixing…"), kind="info"), False
    if snapshot.state == mc_lora_jobs.DONE and not snapshot.summary.startswith(("Not made", "Cancelled")):
        return mc_lora_render.notice(mc_lora_render.escape(snapshot.summary) + ". Press refresh in the extra "
                                     "networks to see its card.", kind="ok"), True
    text = snapshot.summary or snapshot.error or snapshot.message or "The mix did not finish."
    return mc_lora_render.notice(mc_lora_render.escape(text), kind="error"), True


def on_tick(job_id):
    status, done = job_answer(job_id)
    if not done:
        return status, gr.update(), gr.update()
    return status, mixes_html(), gr.update(active=False)


def create_ui() -> None:
    """The Mix page, built inside the tab's ``TabItem``."""
    gr.HTML(
        mc_lora_render.notice(
            "Mixes two LoRAs into one new LoRA file, exactly: the new LoRA changes the model as the two "
            "would together at the strengths you choose, its rank the two ranks together. Both must be "
            "made for the same model. The new LoRA gets a placeholder picture until you set one with "
            "<i>Replace preview</i> in its card's dialog, and a description saying what it was mixed from.",
            kind="info",
        )
    )
    choices = lora_choices()
    with gr.Row():
        first = gr.Dropdown(label="First LoRA", choices=choices, value=None, filterable=True,
                            interactive=True, scale=4, elem_id="mc-lora-mix-first")
        first_strength = gr.Slider(label="Strength", minimum=mc_lora_mix.STRENGTH_MIN,
                                   maximum=mc_lora_mix.STRENGTH_MAX, step=0.05, value=1.0, scale=2)
    with gr.Row():
        second = gr.Dropdown(label="Second LoRA", choices=choices, value=None, filterable=True,
                             interactive=True, scale=4, elem_id="mc-lora-mix-second")
        second_strength = gr.Slider(label="Strength", minimum=mc_lora_mix.STRENGTH_MIN,
                                    maximum=mc_lora_mix.STRENGTH_MAX, step=0.05, value=1.0, scale=2)
    with gr.Row():
        folder = gr.Dropdown(label="Into folder — chosen for every mix", choices=folder_choices(), value=None,
                             filterable=True, interactive=True, scale=3, elem_id="mc-lora-mix-folder")
        name = gr.Textbox(label="Name", placeholder="mix_<first>_<second> when left empty", scale=3,
                          elem_id="mc-lora-mix-name")
        refresh = gr.Button("↻", size="sm", scale=0, min_width=50)
    with gr.Row():
        mix_button = gr.Button("Mix", variant="primary", elem_id="mc-lora-mix-run")
    status = gr.HTML()
    gr.HTML(mc_lora_render.column_title_html("Mixes made here"))
    made = gr.HTML(value=mixes_html, elem_id="mc-lora-mix-list")
    job_id = gr.State(0)

    for picker in (first, second):
        picker.change(fn=suggest_name, inputs=[first, second], outputs=[name], show_progress=False)
    refresh.click(fn=do_refresh, outputs=[first, second, folder, made], show_progress=False)

    inputs = [first, first_strength, second, second_strength, folder, name]
    timer_class = getattr(gr, "Timer", None)
    if timer_class is not None:
        timer = timer_class(POLL_SECONDS, active=False)

        def queue(*values):
            text, folder_update, ident, poll = do_mix(*values)
            return text, folder_update, ident, gr.update(active=poll)

        mix_button.click(fn=queue, inputs=inputs, outputs=[status, folder, job_id, timer])
        timer.tick(fn=on_tick, inputs=[job_id], outputs=[status, made, timer], show_progress=False)
    else:
        def queue_once(*values):
            text, folder_update, ident, _poll = do_mix(*values)
            return text, folder_update, ident

        mix_button.click(fn=queue_once, inputs=inputs, outputs=[status, folder, job_id])
