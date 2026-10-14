"""The Breeze TTS 2 sidecar, as a package the parent can read the protocol out of.

A package only so that :mod:`mc_voice_breeze_runtime` can import
``breeze_worker.worker`` for the frame format, the marker and the protocol
version, rather than keeping a second copy of a wire protocol in the module on
the other end of the pipe.

``worker.py`` is also run directly, by path, under the *isolated Breeze
interpreter* -- so it never imports anything from this package and never
assumes it was imported as part of one.

Separate from ``vibevoice_worker`` for the reason every engine's worker is
separate from every other's: one dependency closure per engine. Breeze pins
torch 2.9.1 and transformers 4.57.3 where VibeVoice pins 2.8.0 and 4.51.3, so
the two could not share a runtime even if they wanted to.

The wire format is shared by *agreement* rather than by import, and
``tests/test_voice_breeze_worker.py`` holds that agreement to byte equality
against ``vibevoice_worker`` and ``pocket_worker``.

Nothing here may import a Forge module, a Model Chain module, Torch,
transformers, ``qwen_tts`` or Breeze's own code at module level.
``tests/test_voice_independence.py`` asserts exactly that.
"""
