"""The operating system's own file dialog, when there is one to open.

The in-page picker in :mod:`mc_llm_browse` is correct everywhere and pleasant
nowhere: it navigates one folder per click, and somebody whose models are eight
levels down a drive is doing eight clicks to reach what Explorer would have
opened them at. On the overwhelmingly common installation -- a WebUI started
on the same Windows desktop that is looking at it -- there is a much better
answer available, which is the dialog the user already knows.

The reason the in-page picker still exists is that this one is not always
available, and the failure is silent from the browser's side. A WebUI run with
``--listen`` or ``--share`` is being looked at from another machine, and a
dialog opened here would appear on the *server's* screen, where nobody is
sitting: the browser would simply hang until the timeout. So availability is
decided before anything is opened, and the in-page picker takes over whenever
the answer is no.

Two routes, because one of them is not always there
--------------------------------------------------
On Windows the dialog is asked for through PowerShell and
``System.Windows.Forms``, which is part of the operating system and therefore
present on every machine this can run on. That matters because the popular
one-click Forge packages ship an *embedded* Python, and an embedded Python has
no ``tkinter`` -- so the obvious route is missing on exactly the installations
most likely to want this. Everywhere else, and as the fallback if PowerShell
will not run, it is ``tkinter``.

Why a subprocess rather than ``tkinter`` in the worker thread
-------------------------------------------------------------
Three reasons, and any one of them would be enough:

* **Threads.** Tk is not thread-safe and objects to being created off the main
  thread on some platforms. Gradio handlers do not run on the main thread.
* **Blast radius.** A Tk that segfaults, or hangs on a broken display, takes
  its own process down. Here that process is a two-second child, not the WebUI.
* **The timeout.** A dialog nobody dismisses would otherwise hold a worker
  thread forever; a child process can simply be killed.

What comes back is a path on stdout and nothing else, so the parsing is a
``strip()``. Anything else -- a non-zero exit, an empty answer, a timeout -- is
"no file was chosen", which is also what pressing Cancel means.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

TIMEOUT_SECONDS = 600.0
"""How long a dialog may stay open before the child is killed.

Long, because it is a person choosing a file and they may go and find it first.
Bounded, because the alternative to a bound is a worker thread held by a dialog
on a screen nobody is watching.
"""

# Run by the child. Deliberately one string with no imports from this package:
# it is executed by the same interpreter but shares nothing else with it, so a
# dialog cannot reach anything the WebUI owns.
_SCRIPT = r"""
import json, sys
request = json.loads(sys.argv[1])
try:
    import tkinter
    from tkinter import filedialog
except Exception as exc:
    sys.stderr.write("no tkinter: %s" % exc)
    raise SystemExit(2)

root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
try:
    if request["kind"] == "folder":
        chosen = filedialog.askdirectory(title=request["title"],
                                         initialdir=request["initial"] or None,
                                         mustexist=True)
    else:
        chosen = filedialog.askopenfilename(title=request["title"],
                                            initialdir=request["initial"] or None,
                                            filetypes=[tuple(entry) for entry
                                                       in request["patterns"]])
finally:
    try:
        root.destroy()
    except Exception:
        pass
sys.stdout.write(chosen or "")
"""


_PS_FILE = """
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '{title}'
$dialog.Filter = '{filter}'
$dialog.InitialDirectory = '{initial}'
$dialog.Multiselect = $false
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::Out.Write($dialog.FileName)
}}
$owner.Dispose()
"""

_PS_FOLDER = """
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{title}'
$dialog.SelectedPath = '{initial}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::Out.Write($dialog.SelectedPath)
}}
"""


class Unavailable(RuntimeError):
    """No native dialog can be opened here, with the reason as its message.

    ``final`` marks the reasons that are not worth trying another route for --
    a dialog somebody left open is one dialog too many already.
    """

    final = False


def available() -> str:
    """"" when a native dialog can be opened, or why it cannot.

    A string rather than a bool because every caller wants the reason: the
    panel says it out loud when it falls back to the in-page picker, and
    "Browse did nothing" is the failure this is trying to avoid.
    """
    if _served_remotely():
        return ("This WebUI is being served to other machines, so a file dialog opened here "
                "would appear on the server's screen rather than yours.")
    if sys.platform not in ("win32", "darwin") and not _has_display():
        return "This machine has no desktop session for a file dialog to open on."
    return ""


def choose_file(title: str, patterns: tuple[tuple[str, str], ...] = (),
                initial: str | None = None) -> str | None:
    """Open a file dialog and return what was picked, or ``None`` for Cancel."""
    return _run("file", title, patterns, initial)


def choose_folder(title: str, initial: str | None = None) -> str | None:
    """Open a folder dialog and return what was picked, or ``None`` for Cancel."""
    return _run("folder", title, (), initial)


def _run(kind: str, title: str, patterns, initial) -> str | None:
    reason = available()
    if reason:
        raise Unavailable(reason)

    problems = []
    for route in _routes():
        try:
            return route(kind, title, patterns, initial)
        except Unavailable as exc:
            # A route that is not installed is a reason to try the next one.
            # A dialog the user cancelled is not an Unavailable at all, and a
            # timeout stops here rather than opening a second dialog at the
            # end of somebody's ten minutes.
            problems.append(str(exc))
            if getattr(exc, "final", False):
                raise
    raise Unavailable(" ".join(problems) or "No file dialog is available on this machine.")


def _routes():
    """The dialogs to try, best first. See the module docstring."""
    if sys.platform == "win32":
        return (_powershell, _tkinter)
    return (_tkinter,)


def _tkinter(kind: str, title: str, patterns, initial) -> str | None:
    request = json.dumps({"kind": kind, "title": title, "initial": str(initial or ""),
                          "patterns": [list(pair) for pair in patterns]})
    finished = _spawn([sys.executable, "-c", _SCRIPT, request])
    if finished.returncode == 2:
        raise Unavailable("this Python has no tkinter")
    if finished.returncode != 0:
        raise Unavailable(f"the dialog failed ({_last_line(finished.stderr)})")
    return (finished.stdout or "").strip() or None


def _powershell(kind: str, title: str, patterns, initial) -> str | None:
    """The Windows common dialog, through an assembly Windows always has."""
    if kind == "folder":
        script = _PS_FOLDER.format(title=_ps(title), initial=_ps(str(initial or "")))
    else:
        script = _PS_FILE.format(title=_ps(title), initial=_ps(str(initial or "")),
                                 filter=_ps(_ps_filter(patterns)))
    finished = _spawn(["powershell", "-NoProfile", "-NonInteractive", "-STA",
                       "-ExecutionPolicy", "Bypass", "-Command", script])
    if finished.returncode != 0:
        raise Unavailable(f"PowerShell could not open a dialog ({_last_line(finished.stderr)})")
    return (finished.stdout or "").strip() or None


def _spawn(command):
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=TIMEOUT_SECONDS,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise _final(
            f"The file dialog was still open after {TIMEOUT_SECONDS / 60:.0f} minutes and was "
            f"closed. If no dialog appeared, it opened on a screen you are not looking at — "
            f"use the browser below instead.") from None
    except OSError as exc:
        raise Unavailable(f"it could not be started ({exc})") from None


def _final(message: str) -> Unavailable:
    """An Unavailable that stops the search rather than trying the next route."""
    failure = Unavailable(message)
    failure.final = True
    return failure


def _last_line(text: str) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else "no reason given"


def _ps(text: str) -> str:
    """``text`` inside a PowerShell single-quoted string."""
    return str(text).replace("'", "''")


def _ps_filter(patterns) -> str:
    """Tk's ``(label, glob)`` pairs as a Win32 dialog filter string."""
    if not patterns:
        return "All files|*.*"
    return "|".join(f"{label}|{glob}" for label, glob in patterns)


def _served_remotely() -> bool:
    """Whether this WebUI is being looked at from another machine.

    Read from the host's own command line, because that is where the answer is:
    ``--listen`` binds to every interface and ``--share`` publishes a tunnel,
    and both mean the browser is somewhere the server's screen is not. A
    default install binds to 127.0.0.1 and is therefore, necessarily, being
    looked at from the machine it is running on.
    """
    try:
        from modules import shared

        options = getattr(shared, "cmd_opts", None)
    except Exception:
        return False
    if options is None:
        return False
    for name in ("share", "listen", "ngrok", "server_name"):
        if getattr(options, name, None):
            return True
    return False


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
