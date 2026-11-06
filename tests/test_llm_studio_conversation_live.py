"""The Conversation pop-up on a real Gradio 4.40 page: ``tests/live/llm_studio_conversation_live.py``.

Opt-in, like ``tests/test_llm_studio_live.py``, whose interpreter check it
shares: the suite's own interpreter has a stand-in Gradio, so the live script
runs in the interpreter ``MC_GRADIO_PYTHON`` names -- a venv with Gradio
4.40.0, its pins, Playwright and pytest -- and this file skips without one. Run
it whenever the pop-up's lists, its sections or what a press on a row opens
change: that a favourite with another character lands on the thread starred
rests on Gradio running the drop-down's own ``change`` after the press that set
it, which no stand-in can say.
"""
import json
import subprocess
from pathlib import Path

from test_llm_studio_live import _interpreter

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "live" / "llm_studio_conversation_live.py"


def test_the_conversation_pop_up_holds_on_a_real_gradio_page():
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
