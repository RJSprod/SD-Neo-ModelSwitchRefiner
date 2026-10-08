"""The action row on a real Gradio 4.40 page: ``tests/live/llm_studio_row_live.py``.

Opt-in, like ``tests/test_vibevoice_upstream.py``: the suite's own interpreter
has a stand-in Gradio, so the live script runs in the interpreter
``MC_GRADIO_PYTHON`` names -- a venv with Gradio 4.40.0, the pins Mini Paint
NEO's ``tests/requirements.txt`` lists, and Playwright -- and this file skips
without one. Run it whenever the row's selectors, the marker or the tap
handling change: the node harness in ``test_llm_studio_js.py`` stands in for
the host's DOM, and the first build of the row passed all of it while a tap did
nothing on the real page.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "live" / "llm_studio_row_live.py"


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


def test_the_row_works_on_a_real_gradio_page():
    python = _interpreter()
    run = subprocess.run([python, str(LIVE)], capture_output=True, text=True, timeout=600,
                         cwd=str(ROOT))
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    report = None
    # The script prints one JSON object; Gradio may print a line of its own first.
    for start in range(len(lines)):
        if lines[start].startswith("{"):
            try:
                report = json.loads("\n".join(lines[start:]))
                break
            except json.JSONDecodeError:
                continue
    assert report is not None, run.stdout[-2000:] + run.stderr[-2000:]
    failed = [name for name, held in report["checks"].items() if not held]
    assert not failed, (failed, report.get("errors"))
    assert run.returncode == 0, run.stderr[-2000:]
