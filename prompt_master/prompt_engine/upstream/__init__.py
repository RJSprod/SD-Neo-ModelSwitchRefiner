"""Vendored Prompt-Master-LD engine.

Every module beside this one is a copy of the upstream file of the same name.
Ten of the eleven engine modules are byte-identical to upstream; ``imaging.py``
carries the single approved change (Torch tensor helpers removed). See
``UPSTREAM_DIFF_NOTES.md`` at the repository root, and verify with
``tools/check_upstream_sync.py``.

This file is NOT a copy of upstream's ``__init__.py``. Upstream's registers
ComfyUI nodes and imports ``node.py``, which pulls in ComfyUI and Torch; the
standalone application must import the engine without either. Nothing is
re-exported here on purpose — callers import the module they need, exactly as
``brain.py`` does internally, so the upstream relative imports keep working
unchanged.

``node.py``, ``routes.py``, ``backend.py`` and ``selftest.py`` are vendored as
implementation references only. They still import ComfyUI, aiohttp and Torch,
and nothing in the application imports them.
"""
