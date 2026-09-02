"""Time DPDFNet on every execution provider this machine offers, before buying one.

Never imported by the extension. It answers one question that costs gigabytes
to answer the other way: is a graphics card faster than the processor *for this
model*, on this machine?

Why the question is not obvious
-------------------------------
DPDFNet is a streaming, recurrent denoiser. Upstream runs it in roughly 10 ms
hops, which at its 48 kHz native rate is 480 samples per inference -- so a ten
second reply is about a thousand separate calls, and each one depends on the
state the last one left behind. Nothing can be batched and nothing can overlap.

That is the shape of work a graphics card is worst at. Every call pays a host to
device copy, a set of kernel launches, a copy back and a synchronisation, and
none of it can be hidden behind parallelism that does not exist here. A model
whose per-call arithmetic is smaller than that overhead runs slower on the card
than on the processor, and the measurement that says so looks exactly like the
measurement that says the card is not being used.

This has already been measured once, on a machine with a 3090 and a 5090:

    DPDFNet on DmlExecutionProvider   RTF 1.27
    DPDFNet on CPUExecutionProvider   RTF 1.03 - 1.06

DirectML lost. It is not obvious that CUDA would too -- its launch overhead is
lower and it can capture a graph -- but the shape of the work is the same, and
finding out by installing onnxruntime-gpu means a second runtime closure, a few
gigabytes of CUDA and cuDNN, and a driver compatibility matrix. This script is
the cheap version of that experiment.

What it does
------------
Drives upstream's own ``StreamEnhancer`` -- the same class the Voice Pipeline
worker drives, through the same seam the worker replaces to choose a provider --
over synthetic speech in real packet sizes, and reports the real-time factor per
provider. Under 1.0 means the stage can keep up with playback; the Voice
Pipeline wants comfortably under, because PocketTTS is synthesising beside it on
the same cores.

Usage
-----
    <pipeline python> tools/bench_dpdfnet_providers.py --model <path to .onnx>

The interpreter matters: ``dpdfnet`` and ``onnxruntime`` live in the Voice
Pipeline's own runtime rather than in Forge's. On Windows that is

    model_chain_voice\\pipeline\\runtime\\env\\Scripts\\python.exe

and the model is under ``model_chain_voice\\pipeline\\models\\dpdfnet``.

To test CUDA, install ``onnxruntime-gpu`` into a scratch virtual environment
with ``dpdfnet`` beside it and run this there. Nothing in the extension's own
runtimes needs to change to find out the answer.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HOP_MS = 10
"""The packet size the Voice Pipeline actually delivers, in milliseconds.

Benchmarking with one long buffer would measure a different program: it would
amortise the per-call overhead this script exists to expose, and report a number
the streaming path can never reach.
"""


def speech(seconds: float, rate: int) -> list:
    """Something with the spectral shape of voiced speech rather than silence.

    A denoiser given digital silence can take a different path through its own
    gating, so a benchmark on zeros is a benchmark of the wrong branch.
    """
    total = int(seconds * rate)
    return [0.25 * math.sin(2.0 * math.pi * 220.0 * i / rate)
            + 0.08 * math.sin(2.0 * math.pi * 1750.0 * i / rate)
            + 0.03 * math.sin(2.0 * math.pi * 4400.0 * i / rate)
            for i in range(total)]


def time_one(provider: str, model: Path, seconds: float, rate: int,
             threads: int, adapter: int) -> dict:
    """One provider, start to finish, or the reason it could not be used."""
    import numpy as np
    import onnxruntime
    from dpdfnet import onnx_backend
    from dpdfnet.stream import StreamEnhancer

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if provider == "DmlExecutionProvider":
        # Required by ONNX Runtime for DirectML, not tuning -- and part of why
        # DirectML can lose: the operators DirectML does not implement fall back
        # to the processor and run there single file, with the memory pattern
        # optimiser off, where a plain CPU session would have had both.
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    if provider == "CPUExecutionProvider":
        wanted = ["CPUExecutionProvider"]
    elif provider == "DmlExecutionProvider":
        wanted = [("DmlExecutionProvider", {"device_id": adapter}),
                  "CPUExecutionProvider"]
    else:
        wanted = [(provider, {"device_id": adapter}), "CPUExecutionProvider"]

    built = []

    def opened(onnx_path):
        session = onnxruntime.InferenceSession(str(onnx_path), sess_options=options,
                                               providers=wanted)
        built.extend(session.get_providers())
        return session

    original = onnx_backend.create_cpu_session
    onnx_backend.create_cpu_session = opened
    try:
        enhancer = StreamEnhancer(model=model.stem, onnx_path=model, verbose=False)
    finally:
        onnx_backend.create_cpu_session = original

    if provider not in built:
        # ONNX Runtime does not raise for a provider it cannot use. It drops it
        # and builds on the next one, and the session it hands back works -- so
        # a run that never touched the card would otherwise be reported as the
        # card's own time.
        return {"provider": provider, "error": f"fell back to {', '.join(built)}"}

    enhancer.reset()
    hop = max(1, int(rate * HOP_MS / 1000))
    audio = speech(seconds, rate)
    # One hop first, untimed. The first call builds kernels, allocates the
    # device arena and pages the weights in; charging a whole run for that
    # reports a start-up cost as a steady-state one.
    enhancer.process(np.asarray(audio[:hop], dtype=np.float32), sample_rate=rate)

    began = time.perf_counter()
    calls = 0
    for at in range(hop, len(audio), hop):
        enhancer.process(np.asarray(audio[at:at + hop], dtype=np.float32),
                         sample_rate=rate)
        calls += 1
    spent = time.perf_counter() - began

    played = (len(audio) - hop) / float(rate)
    return {"provider": provider, "providers": built, "seconds": spent,
            "rtf": spent / played if played else 0.0,
            "per_call_ms": 1000.0 * spent / calls if calls else 0.0, "calls": calls}


def main(argv=None) -> int:
    parsed = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parsed.add_argument("--model", required=True, type=Path,
                        help="dpdfnet8_48khz_hr.onnx, as installed")
    parsed.add_argument("--seconds", type=float, default=10.0,
                        help="how much audio to push through (default 10)")
    parsed.add_argument("--rate", type=int, default=24000,
                        help="the rate the pipeline hands it, not the model's "
                             "own -- PocketTTS speaks at 24000 (default 24000)")
    parsed.add_argument("--threads", type=int, default=4,
                        help="intra-op threads, to match the panel's setting")
    parsed.add_argument("--adapter", type=int, default=0,
                        help="which card, for the providers that take one")
    found = parsed.parse_args(argv)

    if not found.model.is_file():
        print(f"no model at {found.model}", file=sys.stderr)
        return 2

    try:
        import onnxruntime
    except ImportError:
        print("this interpreter has no onnxruntime -- run it with the Voice "
              "Pipeline's own python, or a scratch environment that has "
              "onnxruntime-gpu and dpdfnet in it", file=sys.stderr)
        return 2

    offered = list(onnxruntime.get_available_providers())
    print(f"onnxruntime {onnxruntime.__version__} offers: {', '.join(offered)}")
    print(f"{found.seconds:.0f}s of audio in {HOP_MS} ms hops at {found.rate} Hz, "
          f"{found.threads} thread(s)\n")

    rows = []
    for provider in offered:
        try:
            rows.append(time_one(provider, found.model, found.seconds, found.rate,
                                 found.threads, found.adapter))
        except Exception as exc:
            rows.append({"provider": provider, "error": f"{type(exc).__name__}: {exc}"})

    width = max(len(row["provider"]) for row in rows)
    for row in rows:
        if row.get("error"):
            print(f"  {row['provider']:<{width}}  — {row['error']}")
            continue
        print(f"  {row['provider']:<{width}}  RTF {row['rtf']:.2f}   "
              f"{row['per_call_ms']:.2f} ms per {HOP_MS} ms hop   "
              f"({row['calls']} calls in {row['seconds']:.2f}s)")

    ran = [row for row in rows if not row.get("error")]
    if ran:
        best = min(ran, key=lambda row: row["rtf"])
        print(f"\nfastest here: {best['provider']} at RTF {best['rtf']:.2f}")
        if best["rtf"] >= 1.0:
            print("every provider is over real time, so this model cannot keep up "
                  "with playback on this machine whatever it is placed on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
