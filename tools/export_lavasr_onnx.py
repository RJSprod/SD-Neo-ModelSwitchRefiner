"""Export LavaSR to ONNX, and refuse to bless an export that does not match.

Never imported by the extension. It is the maintainer's tool for turning
upstream's PyTorch LavaSR into an artifact the light Voice Pipeline runtime can
execute, on a machine that can reach huggingface.co.

Why this tool exists
--------------------
LavaSR upstream needs PyTorch, a vocos fork pinned to a git branch, and an
``encodec`` release that publishes no wheel. Exported to ONNX it needs none of
them: it drops into the 138 MB closure DPDFNet already runs in, beside DPDFNet,
in the one worker process the pipeline already contains -- and it inherits the
DirectML placement control without a line of new code.

What it refuses to write
------------------------
An export is not a translation, it is a re-implementation by a compiler, and
the failure mode that matters here is silent. ``LavaBWE`` monkey-patches the
vocos head to build a complex spectrogram and run an inverse STFT; complex
dtypes and ``torch.istft`` are exactly where exporters quietly substitute
something almost right. Almost right, for this feature, means audio that still
sounds like speech and is the wrong length or the wrong brightness.

So this tool exports and then *proves* the export, and writes nothing unless
every one of these holds against the PyTorch model it came from:

    the output length matches sample for sample, at several input lengths;
    the sample values match within ``--tolerance`` (default 1e-3);
    the measured input rate the graph implements is recorded, not assumed;
    a second run over the same input gives the same bytes.

A failure prints what diverged and exits 1. That is the useful outcome: it
means the ONNX path is not ready, which is worth knowing before it is offered
in settings rather than after somebody enables it.

Usage
-----
    python tools/export_lavasr_onnx.py --out build/lavasr-onnx
    python tools/export_lavasr_onnx.py --out build/lavasr-onnx --opset 17
    python tools/export_lavasr_onnx.py --out build/lavasr-onnx --tolerance 5e-4

On success it prints the manifest rows -- filename, byte count, SHA-256 -- to
paste into ``voice/managed-pipeline-models.json``, plus the measured rate
contract to pass to ``pin_pipeline_models.py --lava-contract``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# The rates upstream's reviewed inference path actually implements. Recorded
# here as the expectation the export is checked against rather than as a
# constant the exporter trusts: if a future LavaSR changes them, this tool
# should fail loudly rather than write a graph that resamples to the wrong
# place. See docs/19 and the manifest's contract notes.
UPSTREAM_INPUT_RATE = 16000
UPSTREAM_OUTPUT_RATE = 48000

# Lengths in samples, at the backend's input rate, that the export is proved
# over. Chosen to include one shorter than a single analysis window and one
# that is not a whole number of windows, because a graph that silently pads or
# truncates only shows it at a length that does not divide evenly.
PROOF_LENGTHS = (1600, 16000, 23456, 48000)


class ExportError(RuntimeError):
    """Something that must stop the export, said in a sentence."""


def compare(reference, produced, tolerance: float) -> dict:
    """Whether one array matches another closely enough to ship.

    A pure function of two sequences so that the decision this tool turns on
    can be tested without a GPU, a network, or a 122 MB wheel -- the numerical
    verdict is the whole value of the tool and it should not be the one part
    that is only exercised by running it for real.

    Length is checked before values and reported separately, because the two
    failures mean different things. A length mismatch is a resampling or
    padding bug and no tolerance makes it acceptable; a value mismatch within a
    matching length is a precision question that ``tolerance`` is allowed to
    settle.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(produced, dtype=np.float64).reshape(-1)
    if ref.size != got.size:
        return {"ok": False, "reason": "length",
                "detail": f"reference is {ref.size} samples, the export gave {got.size}",
                "worst": None, "samples": int(ref.size)}
    if not ref.size:
        return {"ok": False, "reason": "empty",
                "detail": "the reference produced no audio to compare against",
                "worst": None, "samples": 0}
    worst = float(np.max(np.abs(ref - got)))
    if not np.isfinite(worst):
        return {"ok": False, "reason": "not-finite",
                "detail": "the export produced NaN or infinity where the reference did not",
                "worst": worst, "samples": int(ref.size)}
    if worst > tolerance:
        where = int(np.argmax(np.abs(ref - got)))
        return {"ok": False, "reason": "value",
                "detail": (f"worst difference {worst:.3e} at sample {where} of {ref.size}, "
                           f"which is over the {tolerance:.3e} allowed"),
                "worst": worst, "samples": int(ref.size)}
    return {"ok": True, "reason": "", "detail": "", "worst": worst,
            "samples": int(ref.size)}


def digest(path: Path) -> dict:
    """The manifest row for one produced file."""
    blob = path.read_bytes()
    return {"filename": path.name, "local_name": path.name,
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def load_upstream(device: str):
    """Import LavaSR and build the v2 model, or say precisely what is missing."""
    try:
        from LavaSR.model import LavaEnhance2
    except ImportError as exc:
        raise ExportError(
            f"LavaSR is not importable here ({exc}). This tool runs where upstream "
            f"already works: pip install torch torchaudio einops soundfile librosa, "
            f"then 'pip install git+https://github.com/langtech-bsc/vocos.git@"
            f"451e522f7a11c3652c9522f63ea6780736d93de0' and "
            f"'pip install git+https://github.com/ysharma3501/LavaSR.git'.") from None
    return LavaEnhance2(device=device)


def export(model, out: Path, opset: int, tolerance: float, device: str) -> list:
    """Export both stages of LavaSR and prove each against its PyTorch original."""
    import numpy as np
    import torch

    try:
        import onnxruntime
    except ImportError:
        raise ExportError("onnxruntime is not installed here, so the export cannot be "
                          "proved. Install it before running this tool: an unproved "
                          "export is the one thing this tool exists to refuse.") from None

    out.mkdir(parents=True, exist_ok=True)
    produced = []

    # One graph, not two. Upstream's enhance() is denoise -> resample -> BWE,
    # and the resample between them is part of what the pipeline has to get
    # right; exporting the halves separately would leave that seam in Python on
    # this side and in the graph on the other, which is precisely the kind of
    # difference that shows up as audio of the wrong length.
    class Whole(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, wav):
            return self.inner.enhance(wav, enhance=True, denoise=True, batch=False)

    wrapped = Whole(model).eval()
    target = out / "lavasr-v2.onnx"
    example = torch.zeros(1, PROOF_LENGTHS[1], dtype=torch.float32, device=device)

    print(f"  exporting to opset {opset}…")
    try:
        torch.onnx.export(
            wrapped, (example,), str(target),
            input_names=["audio"], output_names=["enhanced"],
            dynamic_axes={"audio": {1: "samples"}, "enhanced": {0: "out_samples"}},
            opset_version=opset, do_constant_folding=True)
    except Exception as exc:
        raise ExportError(
            f"The export itself failed: {exc.__class__.__name__}: {exc}\n"
            f"  The likely cause is the inverse STFT in the monkey-patched vocos head, "
            f"which builds a complex spectrogram -- complex dtypes and torch.istft are "
            f"where ONNX exporters most often stop. Try --opset 17 or higher, and if it "
            f"still fails the ONNX path is not available for this model without "
            f"rewriting that head, which is a change to upstream rather than to this "
            f"repository.") from None

    session = onnxruntime.InferenceSession(str(target),
                                           providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    print("  proving the export against PyTorch…")
    for count in PROOF_LENGTHS:
        rng = np.random.default_rng(count)
        audio = rng.standard_normal((1, count)).astype(np.float32) * 0.1
        with torch.no_grad():
            reference = wrapped(torch.from_numpy(audio).to(device)).cpu().numpy()
        got = session.run(None, {name: audio})[0]
        verdict = compare(reference, got, tolerance)
        ratio = verdict["samples"] / count if count else 0
        worst = "n/a" if verdict["worst"] is None else f"{verdict['worst']:.2e}"
        print(f"    {count:>6} samples in -> {verdict['samples']:>7} out "
              f"({ratio:.2f}x)  worst {worst}")
        if not verdict["ok"]:
            raise ExportError(f"The export does not match PyTorch at {count} samples "
                              f"({verdict['reason']}): {verdict['detail']}")

    # The rate the graph actually implements, measured rather than declared.
    rng = np.random.default_rng(7)
    probe = rng.standard_normal((1, UPSTREAM_INPUT_RATE)).astype(np.float32) * 0.1
    out_len = session.run(None, {name: probe})[0].reshape(-1).size
    measured = round(out_len / (probe.shape[1] / UPSTREAM_INPUT_RATE))
    if measured != UPSTREAM_OUTPUT_RATE:
        raise ExportError(
            f"One second of {UPSTREAM_INPUT_RATE} Hz input came back as {out_len} "
            f"samples, so this graph outputs {measured} Hz rather than "
            f"{UPSTREAM_OUTPUT_RATE} Hz. Recording it as 48 kHz would play every reply "
            f"at the wrong speed; fix the export or change the contract deliberately.")

    # Determinism: the same bytes twice. A graph with a nondeterministic op in
    # it is one whose output changes between two replies of the same text.
    first = session.run(None, {name: probe})[0]
    again = session.run(None, {name: probe})[0]
    if not compare(first, again, 0.0)["ok"]:
        raise ExportError("The export gives different audio for the same input twice.")

    produced.append(digest(target))
    return produced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export LavaSR to ONNX and prove it.")
    parser.add_argument("--out", required=True, type=Path,
                        help="directory to write the graph into")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset (17 is the first with a real STFT)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="worst allowed per-sample difference from PyTorch")
    parser.add_argument("--device", default="cpu",
                        help="where to run the PyTorch reference (cpu, cuda:0, …)")
    found = parser.parse_args(argv)

    try:
        print("Loading upstream LavaSR…")
        model = load_upstream(found.device)
        rows = export(model, found.out, found.opset, found.tolerance, found.device)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nProved. Manifest rows:\n")
    print(json.dumps(rows, indent=1))
    print(f"\nContract for the pin:\n"
          f"  --lava-contract backend=onnx,rate={UPSTREAM_INPUT_RATE},"
          f"analysis_ms=<measured>,context_ms=<measured>\n"
          f"The two measured windows still come from the Phase-0 sweep in "
          f"docs/19-voice-chat-pipeline.md; this tool proves the graph, not the "
          f"latency budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
