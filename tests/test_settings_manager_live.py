"""The settings as accordions on a real Gradio 4.40 page: ``tests/live/settings_manager_live.py``.

Opt-in, like ``tests/test_llm_studio_live.py``: the suite's own interpreter has
a stand-in Gradio, so the live script runs in the interpreter
``MC_GRADIO_PYTHON`` names -- a venv with Gradio 4.40.0, the pins Mini Paint
NEO's ``tests/requirements.txt`` lists, and Playwright -- and this file skips
without one. Run it whenever the script's reading of the column changes (what
it walks into, what it calls a section, the classes it borrows): the browser
tests in ``test_settings_manager_js.py`` stand in for Gradio's DOM, and the
first Lora Viewer placement passed a stand-in while it stood in the wrong
place on the real page.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "live" / "settings_manager_live.py"


def _interpreter() -> str:
    python = os.environ.get("MC_GRADIO_PYTHON", "").strip()
    if not python:
        pytest.skip("MC_GRADIO_PYTHON names no interpreter with Gradio 4.40 and Playwright")
    probe = subprocess.run(
        [python, "-c", "import gradio, playwright; print(gradio.__version__)"],
        capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        pytest.skip(f"{python} has no Gradio and Playwright: {probe.stderr.strip()[-200:]}")
    if not probe.stdout.strip().startswith("4.40"):
        pytest.skip(f"{python} has Gradio {probe.stdout.strip()}, not 4.40")
    return python


def test_the_sections_work_on_a_real_gradio_page():
    python = _interpreter()
    run = subprocess.run([python, str(LIVE)], capture_output=True, text=True, timeout=600,
                         cwd=str(ROOT))
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    report = None
    for start in range(len(lines)):
        if lines[start].startswith("{"):
            try:
                report = json.loads("\n".join(lines[start:]))
                break
            except json.JSONDecodeError:
                continue
    assert report is not None, run.stdout[-2000:] + run.stderr[-2000:]
    failed = [name for name, held in report["checks"].items() if not held]
    assert not failed, (failed, report.get("found"), report.get("errors"))
    assert run.returncode == 0, run.stderr[-2000:]
