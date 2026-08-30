"""Sopro validation: measure this machine, write it down, change nothing.

The half of I-12 that was never done. The invariant asks for a CPU policy that
is *measured*, fixed, and never auto-tuned from runtime measurements -- and only
the third was true. Four intra-op threads was chosen to match Kokoro's synthesis
lane so that a same-machine comparison between the two engines meant something,
and was never measured against six or eight. Precision was worse than unmeasured:
INT8 was labelled "faster" on the strength of what quantization is supposed to
do, and on the first machine anybody measured it was 40% slower.

This is the measurement, and it is a button rather than a script on purpose. The
script it replaces needed Forge's own interpreter, run from the Forge root, with
Forge's data root resolvable -- three conditions that are invisible until one of
them is wrong, at which point it says "the isolated Sopro runtime is not
installed" about a runtime that is installed perfectly well. Running inside the
WebUI process removes all three: the paths are already right because the process
that resolved them is the one asking.

What it does not do
-------------------
It does not change a setting, and it cannot. Each configuration is measured in a
process of its own, handed the precision to use as an argument; the Precision
control in the panel is never written to, so a sweep that is interrupted leaves
nothing behind. Moving the released policy is still a deliberate edit to
``INTRAOP_THREADS`` in ``sopro_worker/worker.py``, made by somebody who has read
the table this produces. I-12's "never auto-tuned" survives a tool that measures
precisely because the tool does not act.

Why a fresh process per configuration
-------------------------------------
OpenMP sizes its thread pool at the first parallel region and
``torch.set_num_interop_threads`` refuses outright once one has run. A sweep that
reconfigured one process between runs would be measuring the first thread count
several times under different labels -- and would look entirely plausible doing
it, which is the dangerous kind of wrong.

Why two lengths
---------------
Synthesis cost here is very nearly ``fixed + rate x audio`` (R² = 0.99 over 39
real segments), and the two halves are moved by different things: the fixed cost
per unit is amortised by longer *segments*, the rate by threads, precision and
solver steps. One real-time factor is their sum at one length, so it conflates
them and hides which one a machine needs. Two lengths separate them.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time

logger = logging.getLogger("model_chain")

#: About two seconds of speech, and about twelve. Far enough apart for the fit
#: to separate the intercept from the slope, and both things a character could
#: plausibly say -- a benchmark on "aaaa" would exercise a text frontend nobody
#: uses.
SHORT = "The kettle has boiled, and I have poured it."
LONG = (
    "There is a particular kind of quiet that arrives in the hour before dawn, "
    "when the traffic has not started and the birds have not either, and the "
    "house makes the small settling noises it never makes in daylight. I have "
    "come to like it more than I expected to."
)

#: The released policy is in here deliberately. A sweep whose table does not
#: contain the configuration you are actually running gives you no baseline to
#: read the others against.
THREADS = (2, 4, 6, 8)
PRECISIONS = ("full", "int8")

#: Long enough for a slow machine to load a model and speak fourteen seconds,
#: short enough that a wedged process does not hold the panel forever.
TIMEOUT = 420

#: The length the table's headline real-time factor is quoted at. A typical
#: committed unit in Conversation, so the number means something about the
#: thing the user actually hears.
QUOTED_SECONDS = 7.0

_lock = threading.Lock()
_state = {
    "running": False, "done": False, "step": 0, "total": 0,
    "message": "", "rows": [], "error": "", "best": None,
}


class BenchError(RuntimeError):
    """A refusal a person can act on, phrased for the panel."""


def state() -> dict:
    """A snapshot for the panel to poll. Never raises, never starts anything."""
    with _lock:
        return {
            "running": bool(_state["running"]),
            "done": bool(_state["done"]),
            "step": int(_state["step"]),
            "total": int(_state["total"]),
            "message": str(_state["message"]),
            "error": str(_state["error"]),
            "rows": list(_state["rows"]),
            "best": dict(_state["best"]) if _state["best"] else None,
        }


def _say(message: str, **fields) -> None:
    """One progress line, to the panel and to model_chain.log at once.

    Both, rather than either: the panel is where somebody watching finds out it
    is still going, and the log is what they send me afterwards.
    """
    with _lock:
        _state.update(fields)
        _state["message"] = message
    logger.info("Model Chain: Sopro validation — %s", message)


def _fit(points):
    """Least squares through (audio_ms, compute_ms).

    A regression rather than a pair of divisions, so one slow run is a residual
    rather than the answer. Returns ``(fixed_ms, rate, r_squared)``.

    ``r_squared`` is ``None`` unless there were more observations than
    parameters, and that is not a technicality. A line through two points fits
    them perfectly whatever they are, so the first version of this reported
    R² = 1.000 on every row of every table it ever printed -- a confidence
    figure that could not fail, sitting next to numbers a person was about to
    make a decision with. Two parameters need at least three observations
    before "how well did it fit" is a question with an answer, and below that
    the honest output is a blank.
    """
    xs = [float(audio) for audio, _compute in points]
    ys = [float(compute) for _audio, compute in points]
    count = len(xs)
    if count < 2:
        return None, None, None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread <= 0:
        # Every run came out the same length, so there is no leverage to
        # separate the intercept from the slope. Reported as a rate alone
        # rather than as a fit nobody should trust.
        return 0.0, (mean_y / mean_x if mean_x else None), None
    rate = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    fixed = mean_y - rate * mean_x
    if count < 3:
        return fixed, rate, None
    residual = sum((y - (fixed + rate * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return fixed, rate, (1.0 - residual / total) if total > 0 else 1.0


def _measure(interpreter, script, environ, request, cwd) -> dict:
    """One configuration, in a process of its own. Never raises."""
    try:
        finished = subprocess.run(  # noqa: S603 - a path this extension built
            [str(interpreter), str(script), "--benchmark"],
            input=json.dumps(request).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environ, cwd=str(cwd), timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer in {TIMEOUT} s"}
    except OSError as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    text = finished.stdout.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    detail = [line for line in
              finished.stderr.decode("utf-8", "replace").strip().splitlines() if line]
    return {"ok": False,
            "error": (detail[-1] if detail
                      else f"the worker exited {finished.returncode} saying nothing")}


def run(threads=None, precisions=None, repeats: int = 2) -> None:
    """The sweep. Long, blocking, and meant to be called on a thread."""
    import mc_voice_paths as paths
    import mc_voice_sopro as sopro
    import mc_voice_sopro_runtime as runtime

    with _lock:
        if _state["running"]:
            raise BenchError("A validation run is already going.")
        _state.update({"running": True, "done": False, "step": 0, "total": 0,
                       "message": "", "rows": [], "error": "", "best": None})
    try:
        _sweep(paths, sopro, runtime,
               [int(n) for n in (threads or THREADS)],
               [str(p) for p in (precisions or PRECISIONS)],
               max(1, min(5, int(repeats))))
    except BenchError as exc:
        with _lock:
            _state["error"] = str(exc)
        logger.warning("Model Chain: Sopro validation stopped — %s", exc)
    except Exception as exc:  # noqa: BLE001 - a panel button may not raise
        with _lock:
            _state["error"] = f"{exc.__class__.__name__}: {exc}"
        logger.warning("Model Chain: Sopro validation failed", exc_info=True)
    finally:
        with _lock:
            _state["running"] = False
            _state["done"] = True


def _sweep(paths, sopro, runtime, threads, precisions, repeats) -> None:
    found = sopro.status()
    if not found.ready:
        raise BenchError("Sopro is not installed, so there is nothing to measure.")
    interpreter = sopro.runtime_python()
    if interpreter is None:
        raise BenchError("The isolated Sopro runtime is not installed.")
    base = sopro.worker_config()
    if not base.get("voices"):
        raise BenchError("There is no Sopro voice to measure with. Create or add one "
                         "first — a benchmark that skipped the voice would not be "
                         "measuring the path Conversation uses.")
    live = runtime.engine()
    if live.get("state") in ("speaking", "preparing"):
        raise BenchError("Sopro is busy speaking. Try again when the reply has "
                         "finished.")

    # Stopped, not left running. Two Sopro processes on one CPU measure each
    # other, and the resident one holds a model this machine may be short of RAM
    # for. It starts again by itself on the next reply.
    _say("Stopping the resident Sopro worker so it does not compete…")
    runtime.stop("a validation run needs the CPU to itself")

    plan = [(precision, count) for precision in precisions for count in threads]
    _say(f"Measuring {len(plan)} configurations. This takes a few minutes.",
         total=len(plan))
    logger.info("Model Chain: Sopro validation — closure %s, %d voice(s), "
                "%d configurations, %d run(s) each at two lengths (%d observations "
                "per row, %d degrees of freedom)",
                base.get("fingerprint", "?"), len(base.get("voices") or {}),
                len(plan), repeats, repeats * 2, max(0, repeats * 2 - 2))

    script = paths.sopro_worker_script()
    root = paths.extension_root()
    rows = []
    for index, (precision, count) in enumerate(plan, start=1):
        _say(f"{index} of {len(plan)} — {precision} precision, {count} threads…",
             step=index)
        environ = dict(__import__("os").environ)
        environ.update(sopro.worker_environment())
        # After worker_environment, which pins these to the released count: the
        # sweep's number has to win, or OpenMP caps the pool at four while Torch
        # is asked for eight and the row belongs to neither.
        environ["MC_SOPRO_INTRAOP_THREADS"] = str(count)
        environ["OMP_NUM_THREADS"] = str(count)
        environ["MKL_NUM_THREADS"] = str(count)
        environ["OPENBLAS_NUM_THREADS"] = str(count)
        started = time.monotonic()
        measured = _measure(interpreter, script, environ,
                            {"config": dict(base, precision=precision),
                             "repeats": repeats, "texts": [SHORT, LONG]},
                            root)
        row = {"precision": precision, "asked_threads": count,
               "seconds": round(time.monotonic() - started, 1)}
        if not measured.get("ok"):
            row["error"] = str(measured.get("error") or "no reason given")
            logger.warning("Model Chain: Sopro validation — %s at %d threads failed: %s",
                           precision, count, row["error"])
        else:
            # Torch's own count rather than the one asked for: a machine with
            # fewer cores than the sweep asked about quietly gives you fewer,
            # and a table that reported the request would invent a data point.
            row["threads"] = int(measured.get("intraop_threads") or count)
            fixed, rate, quality = _fit([(item["audio_ms"], item["compute_ms"])
                                         for item in measured.get("runs") or []])
            row.update({"fixed_ms": fixed, "rate": rate, "r_squared": quality,
                        "load_seconds": measured.get("load_seconds")})
            if rate is not None:
                row["rtf"] = rate + (fixed or 0.0) / (QUOTED_SECONDS * 1000.0)
                row["break_even_speed"] = (1.0 / row["rtf"]) if row["rtf"] > 0 else None
            logger.info(
                "Model Chain: Sopro validation — %s, %d threads: "
                "synth = %s ms + %s x audio, RTF %s at %.0f s, break-even Speed %s, "
                "R² %s, model loaded in %s s",
                precision, row["threads"],
                "?" if fixed is None else f"{fixed:.0f}",
                "?" if rate is None else f"{rate:.3f}",
                "?" if row.get("rtf") is None else f"{row['rtf']:.3f}", QUOTED_SECONDS,
                "?" if row.get("break_even_speed") is None
                else f"{row['break_even_speed']:.2f}x",
                "?" if quality is None else f"{quality:.3f}",
                measured.get("load_seconds", "?"))
        rows.append(row)
        with _lock:
            _state["rows"] = list(rows)

    _report(rows)


def _report(rows) -> None:
    """The table, in the log, where it can be read and sent on.

    Written as one multi-line record rather than a line per row so that it
    survives being copied out of a log somebody is scrolling.
    """
    usable = [row for row in rows if row.get("rtf") is not None]
    lines = ["Sopro validation results — nothing was changed, this is a measurement",
             "  precision  threads     fixed     rate   RTF@%.0fs  break-even      R²"
             % QUOTED_SECONDS]
    for row in rows:
        if row.get("rtf") is None:
            lines.append("  %9s  %7s   %s" % (row["precision"],
                                              row.get("threads", row["asked_threads"]),
                                              row.get("error", "no result")))
            continue
        lines.append("  %9s  %7d  %6.0fms  %7.3f  %8.3f  %9.2fx  %6s"
                     % (row["precision"], row["threads"], row["fixed_ms"] or 0.0,
                        row["rate"], row["rtf"], row["break_even_speed"] or 0.0,
                        "?" if row.get("r_squared") is None
                        else f"{row['r_squared']:.3f}"))
    best = min(usable, key=lambda row: row["rtf"]) if usable else None
    if best:
        lines.append("")
        lines.append("  Fastest here: %s precision at %d intra-op threads — RTF %.3f, "
                     "which streams cleanly up to Speed %.2fx."
                     % (best["precision"], best["threads"], best["rtf"],
                        best["break_even_speed"]))
        lines.append("  Above that break-even, speech is produced more slowly than it "
                     "is heard and the")
        lines.append("  shortfall grows for as long as the reply lasts, which no amount "
                     "of buffering covers.")
        if not (best["precision"] == "full" and best["threads"] == 4):
            lines.append("  This is not the released policy (full, 4 threads). Nothing "
                         "has been changed.")
    logger.info("Model Chain: %s", "\n".join(lines))
    with _lock:
        _state["best"] = dict(best) if best else None
