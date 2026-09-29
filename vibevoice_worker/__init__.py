"""The VibeVoice sidecar, as a package the parent can read the protocol out of.

A package only so that :mod:`mc_voice_vibevoice_runtime` can import
``vibevoice_worker.worker`` for the frame format, the marker and the protocol
version, rather than keeping a second copy of a wire protocol in the module on
the other end of the pipe.

``worker.py`` is also run directly, by path, under the *isolated VibeVoice
interpreter* -- so it never imports anything from this package and never
assumes it was imported as part of one.

Separate from ``voice_worker``, ``sopro_worker``, ``pocket_worker`` and
``pipeline_worker`` for the reason those are separate from one another: one
dependency closure per engine, so that no engine's upgrade is another engine's
regression. This one is the first that wants a graphics card. The four Voice
Chat workers empty every GPU variable before they start and refuse a device
that is not the CPU; this one is started with exactly one card visible and
refuses to work unless it can prove which card that is.

The wire format is shared by *agreement* rather than by import, and
``tests/test_vibevoice_worker.py`` holds that agreement to byte equality
against ``pocket_worker``.

Nothing here may import a Forge module, a Model Chain module, Torch,
transformers or vibevoice at module level. ``tests/test_voice_independence.py``
asserts exactly that, which is what makes the parent able to read this
module's constants without paying for a machine-learning framework it is not
going to use.
"""
