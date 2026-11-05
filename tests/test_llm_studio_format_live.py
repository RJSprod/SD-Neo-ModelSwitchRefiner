"""LLM Studio's formatting on a real Gradio 4.40 page: ``tests/live/llm_studio_format_live.py``.

Opt-in, like ``tests/test_llm_studio_live.py``, whose interpreter check it
shares: the suite's own interpreter has a stand-in Gradio, so the live script
runs in the interpreter ``MC_GRADIO_PYTHON`` names -- a venv with Gradio
4.40.0, its pins and Playwright -- and this file skips without one. Run it
whenever ``mc_llm_format``, the Chatbot's ``latex_delimiters`` or the
transcript's quote and math rules in ``style.css`` change: the half of the
answer that is Gradio's -- its LaTeX placeholders around marked, DOMPurify and
its own KaTeX -- is nowhere else.
"""
import json
import subprocess
from pathlib import Path

from test_llm_studio_live import _interpreter

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "live" / "llm_studio_format_live.py"


def test_the_formatting_holds_on_a_real_gradio_page():
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
    assert not failed, (failed, report.get("errors"))
    assert run.returncode == 0, run.stderr[-2000:]
