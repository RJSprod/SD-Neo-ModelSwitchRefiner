#!/usr/bin/env python3
"""Build a Civitai LoRA CSV from the command line.

This is the same code the extension runs in the background, exposed as a CLI so
a library can be built without starting the WebUI. The CSV it writes is what the
extension reads: drop the result into the extension's ``library`` directory (or
point ``--output`` straight at it).

Identity rules, unchanged from the original tool
------------------------------------------------
* every file is SHA256-hashed and resolved via ``/model-versions/by-hash``
* if the matched version exposes SHA256 values, one must equal the local digest
* the returned version id is authoritative for trigger words and gallery filtering

Columns
-------
``safetensor_file``, ``sha256``, ``addnet_hash``, ``civitai_model_name``,
``civitai_model_id``, ``civitai_version_id``, ``civitai_version_name``,
``base_model``, ``civitai_creator``, ``civitai_url``, ``trigger_words``,
``status``, ``note``, ``model_description``, ``version_description``,
``positive_prompt_1..N`` with the URL and id of the gallery image each prompt
came from.

Descriptions are converted from Civitai's HTML to plain text and are collected by
default; ``--no-descriptions`` turns that off. No image is ever downloaded here -
only the URL is recorded.

The ``sha256`` column is what lets the extension match a row to a LoRA by hash
rather than by file name.

Requirements: Python 3.10+ and ``requests``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

import mc_lora_civitai  # noqa: E402
import mc_lora_paths  # noqa: E402

DEFAULT_OUTPUT = "civitai_lora_library.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a folder of LoRAs, resolve each one on Civitai by SHA256, and write a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="folder to scan")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"CSV to write (default: the extension library's {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--api-key", default=os.environ.get("CIVITAI_API_KEY"), help="Civitai API key; defaults to CIVITAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--request-delay", type=float, default=0.15, help="minimum delay between requests, in seconds")
    parser.add_argument("--max-images", type=int, default=0, help="gallery images inspected per version; 0 means all")
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="skip the model description (saves one request per model page)",
    )
    parser.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    parser.add_argument(
        "--base-model",
        default="",
        help='only accept these base models, e.g. "Krea 2" or "Krea 2, MOSS Image"; empty accepts any exact hash match',
    )
    parser.add_argument(
        "--require-base-model-label",
        action="store_true",
        help="reject exact hash matches that carry no explicit base-model label (needs --base-model)",
    )
    parser.add_argument("--overwrite", action="store_true", help="rewrite the CSV instead of merging into it")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any file could not be resolved")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] not a directory: {root}", file=sys.stderr)
        return 2
    for name, value in (("--timeout", args.timeout), ("--request-delay", args.request_delay)):
        if value < 0 or (name == "--timeout" and value <= 0):
            print(f"[ERROR] {name} must be positive", file=sys.stderr)
            return 2
    if args.max_images < 0:
        print("[ERROR] --max-images cannot be negative", file=sys.stderr)
        return 2

    output = args.output.expanduser() if args.output else mc_lora_paths.library_dir() / DEFAULT_OUTPUT

    print(f"Root:   {root}")
    print(f"Output: {output}")
    print(f"Policy: {args.base_model or 'any base model'}"
          f"{', explicit label required' if args.require_base_model_label else ''}")
    print(f"Text:   trigger words, gallery prompts"
          f"{', model descriptions' if not args.no_descriptions else ''}")

    report = mc_lora_civitai.scan_folder(
        root,
        output,
        api_key=args.api_key,
        timeout=args.timeout,
        request_delay=args.request_delay,
        max_images=args.max_images,
        recursive=not args.no_recursive,
        base_model_filter=args.base_model,
        require_explicit_label=args.require_base_model_label,
        fetch_descriptions=not args.no_descriptions,
        keep_existing_rows=not args.overwrite,
        log=lambda message: print(message, flush=True),
    )

    print("\nDone.")
    print(f"  Files:    {report.total}")
    print(f"  Resolved: {report.resolved}")
    print(f"  Skipped:  {report.skipped}")
    print(f"  Rows:     {len(report.rows)}")
    for status in sorted(report.counts):
        print(f"  {status}: {report.counts[status]}")

    if args.strict and report.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
