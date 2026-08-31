"""The PocketTTS sidecar, as a package the parent can read the protocol out of.

A package only so that :mod:`mc_voice_pocket_runtime` can import
``pocket_worker.worker`` for the frame format and the protocol version, rather
than keeping a second copy of a wire protocol in the module on the other end of
the pipe.

``worker.py`` is also run directly, by path, under the *isolated PocketTTS
interpreter* -- so it never imports anything from this package and never assumes
it was imported as part of one.

Separate from both ``voice_worker`` and ``sopro_worker`` on purpose, and the
reason is I-PKT-6 rather than tidiness. Three engines, three dependency
closures: Kokoro's sherpa-onnx, Sopro's PyTorch, and PocketTTS's own PyTorch --
which is *not* Sopro's, because two engines pinned to one Torch build would make
either engine's upgrade the other engine's regression. A single worker file that
had to be importable under all three would be one import away from a PocketTTS
runtime reaching for Sopro's tensors or the reverse.

The wire format is shared by *agreement* rather than by import, and
``tests/test_voice_pocket_worker.py`` holds that agreement to byte equality
against ``voice_worker`` and ``sopro_worker``.

Nothing here may import a Forge module, a Model Chain module, Torch or
PocketTTS at module level. ``tests/test_voice_independence.py`` asserts exactly
that, which is what makes the parent able to read this module's constants
without paying for a machine-learning framework it is not going to use.
"""
