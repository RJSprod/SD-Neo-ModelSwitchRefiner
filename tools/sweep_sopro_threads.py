"""Measure this machine's Sopro closure across thread counts and precisions.

Not part of the extension and never imported by it. A tool you run once, on the
machine that will do the speaking, to answer a question no amount of reading
can: how fast is Sopro *here*.

    python tools/sweep_sopro_threads.py                 # the default sweep
    python tools/sweep_sopro_threads.py --threads 4,8   # just these two
    python tools/sweep_sopro_threads.py --precision full

Run it from the Forge root, with Forge's own Python, and with Sopro installed
and at least one voice created. It starts and stops the isolated Sopro
interpreter itself and does not need the WebUI running -- and the WebUI should
*not* be running, or Sopro should be unloaded from the Voice panel, because two
Sopro processes on one CPU measure each other rather than the machine.

Why this exists
---------------
I-12 asks for a CPU policy that is *measured*, fixed, and never auto-tuned from
runtime measurements. Only the third of those was actually true: four intra-op
threads was chosen to match Kokoro's synthesis lane, so that a same-machine
comparison between the two engines meant something, and never measured against
six or eight. This is the missing measurement, and it is a tool rather than a
runtime feature precisely because I-12 forbids the runtime doing it.

Nothing here changes a setting. It prints a table and stops. Moving the released
policy is still a deliberate edit to ``INTRAOP_THREADS`` in
``sopro_worker/worker.py``, made by a person who has read a table like this one.

What it measures, and why it is two numbers
-------------------------------------------
Synthesis time on this engine fits ``fixed + rate x audio`` extremely tightly --
on 39 segments from a real conversation, R^2 = 0.99 -- and the two halves are
moved by different things:

    fixed, per unit   the prompt state and the first chunk. Amortised by
                      *longer segments*, not by threads.
    rate              the marginal cost of one more second of speech. This is
                      what threads, precision and solver steps move, and it is
                      the one that decides whether streaming works at all.

A single real-time factor is the sum of the two at one length, so it conflates
them and hides which one a machine needs. Two lengths separate them, which is
why the sweep speaks both a short line and a long one.

The number to read is **break-even Speed**. Sopro has no model-native speaking
rate, so the Speed control is a time-compression applied after the model has
produced full-length audio: the compute does not change and the result is
shorter, which multiplies the real-time factor by exactly the speed. A
configuration measured at RTF 0.85 will stream cleanly up to Speed 1.17 and
accumulate silence without bound above it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Two lengths, far enough apart that the fit can separate the intercept from
#: the slope, and both plausible things a character actually says. The short one
#: is about one committed unit; the long one is four or five.
SHORT = ("The kettle has boiled, and I have poured it.")
LONG = (
    "There is a particular kind of quiet that arrives in the hour before dawn, "
    "when the traffic has not started and the birds have not either, and the "
    "house makes the small settling noises it never makes in daylight. I have "
    "come to like it more than I expected to. It is the only part of the day "
    "that asks nothing of anybody, and it is over before most people notice it "
    "was there at all."
)

DEFAULT_THREADS = (2, 4, 6, 8, 12)


def _fit(points):
    """Least squares through (audio_ms, compute_ms), returning fixed and rate.

    Two lengths and several repeats, so this is a regression rather than a pair
    of divisions -- one slow run does not become the answer.
    """
    xs = [float(audio) for audio, _compute in points]
    ys = [float(compute) for _audio, compute in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread <= 0:
        return None, None, None
    rate = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    fixed = mean_y - rate * mean_x
    residual = sum((y - (fixed + rate * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return fixed, rate, (1.0 - residual / total) if total > 0 else 1.0


def _run_once(interpreter, script, environ, request, timeout):
    """One configuration, in a process of its own.

    A fresh process per configuration rather than one process reconfigured
    between runs, and this is the part that makes the table trustworthy: OpenMP
    sizes its pool the first time a parallel region runs, and
    ``torch.set_num_interop_threads`` refuses outright once one has. A sweep
    that reused a process would be measuring the first thread count several
    times under different labels.
    """
    started = time.monotonic()
    finished = subprocess.run(  # noqa: S603 - paths this module built
        [str(interpreter), str(script), "--benchmark"],
        input=json.dumps(request).encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environ, cwd=str(ROOT), timeout=timeout, check=False)
    elapsed = time.monotonic() - started
    line = ""
    for candidate in reversed(finished.stdout.decode("utf-8", "replace").splitlines()):
        if candidate.strip().startswith("{"):
            line = candidate
            break
    if not line:
        detail = finished.stderr.decode("utf-8", "replace").strip().splitlines()
        return {"ok": False, "wall_seconds": elapsed,
                "error": (detail[-1] if detail else
                          f"the worker exited {finished.returncode} saying nothing")}
    found = json.loads(line)
    found["wall_seconds"] = elapsed
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", default=",".join(str(n) for n in DEFAULT_THREADS),
                        help="comma-separated intra-op thread counts to try")
    parser.add_argument("--precision", default="full,int8",
                        help="comma-separated precisions to try (full, int8)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="synthesis runs per length per configuration")
    parser.add_argument("--voice", default="",
                        help="voice id to speak with (default: this install's first)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow one configuration before giving up")
    parser.add_argument("--json", default="",
                        help="also write the raw results to this file")
    asked = parser.parse_args(argv)

    import mc_voice_paths as paths
    import mc_voice_sopro as sopro

    # Deliberately not a check. A running WebUI is a different process, so this
    # one cannot see its Sopro worker, and a test that can never fire is worse
    # than none: it reads like a guarantee. Said out loud instead, where the
    # person running the sweep will read it.
    print("Close the WebUI, or unload Sopro from the Voice panel, before "
          "trusting these numbers — two Sopro processes on one CPU measure "
          "each other.\n")
    interpreter = sopro.runtime_python()
    if interpreter is None:
        print("The isolated Sopro runtime is not installed.", file=sys.stderr)
        return 2
    script = paths.sopro_worker_script()
    base = sopro.worker_config()
    if not base.get("voices"):
        print("This installation has no Sopro voice. Create or add one first — "
              "the benchmark speaks with a real voice because a synthetic one "
              "would not exercise the reconstruction path.", file=sys.stderr)
        return 2

    threads = [int(n) for n in asked.threads.split(",") if n.strip()]
    precisions = [p.strip() for p in asked.precision.split(",") if p.strip()]
    print(f"Sopro closure {base.get('fingerprint', '?')}, "
          f"{len(base.get('voices') or {})} voice(s), "
          f"{len(threads) * len(precisions)} configurations, "
          f"{os.cpu_count()} logical CPUs visible.\n")

    results = []
    for precision in precisions:
        for count in threads:
            environ = dict(os.environ)
            environ.update(sopro.worker_environment())
            # After worker_environment, so the sweep's count wins over the
            # released one it pinned.
            environ["MC_SOPRO_INTRAOP_THREADS"] = str(count)
            environ["OMP_NUM_THREADS"] = str(count)
            environ["MKL_NUM_THREADS"] = str(count)
            environ["OPENBLAS_NUM_THREADS"] = str(count)
            request = {"config": dict(base, precision=precision),
                       "voice_id": asked.voice, "repeats": asked.repeats,
                       "texts": [SHORT, LONG]}
            label = f"{precision:>4} / {count:>2} threads"
            print(f"  measuring {label} …", end="", flush=True)
            try:
                found = _run_once(interpreter, script, environ, request, asked.timeout)
            except subprocess.TimeoutExpired:
                found = {"ok": False, "error": f"no answer in {asked.timeout} s"}
            found["asked_threads"] = count
            found["asked_precision"] = precision
            results.append(found)
            if not found.get("ok"):
                print(f" failed: {found.get('error')}")
                continue
            points = [(r["audio_ms"], r["compute_ms"]) for r in found["runs"]]
            fixed, rate, quality = _fit(points)
            found["fixed_ms"], found["rate"], found["r_squared"] = fixed, rate, quality
            print(f" done in {found['wall_seconds']:.0f} s")

    print("\n" + "=" * 78)
    print(f"{'precision':>9} {'threads':>8} {'fixed':>9} {'rate':>7} {'RTF@7s':>8} "
          f"{'break-even':>11} {'R²':>6}")
    print("-" * 78)
    best = None
    for found in results:
        if not found.get("ok") or found.get("rate") is None:
            print(f"{found['asked_precision']:>9} {found['asked_threads']:>8}   "
                  f"{found.get('error', 'failed')}")
            continue
        # Torch's own count, not the one that was asked for: a machine with
        # fewer cores than the sweep asked about silently gives you fewer.
        got = found.get("intraop_threads", found["asked_threads"])
        rtf = found["rate"] + found["fixed_ms"] / 7000.0
        speed = (1.0 / rtf) if rtf > 0 else float("inf")
        print(f"{found['precision']:>9} {got:>8} {found['fixed_ms']:>8.0f}ms "
              f"{found['rate']:>7.3f} {rtf:>8.3f} {speed:>10.2f}x {found['r_squared']:>6.3f}")
        if best is None or rtf < best[0]:
            best = (rtf, found["precision"], got, speed)
    print("=" * 78)

    if best:
        rtf, precision, count, speed = best
        print(f"\nFastest here: {precision} precision at {count} intra-op threads — "
              f"RTF {rtf:.3f}, which streams cleanly up to Speed {speed:.2f}x.")
        print("Anything above that break-even accumulates silence for as long as the "
              "reply lasts, and no amount of buffering covers it.")
        released = (precision == "full" and count == 4)
        if not released:
            print("\nThis is not the released policy. Nothing has been changed: moving it "
                  "means editing INTRAOP_THREADS in sopro_worker/worker.py (and the "
                  "Precision setting in the Voice panel), with this table as the reason.")
    if asked.json:
        Path(asked.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nRaw results written to {asked.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
