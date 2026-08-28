"""The Voice Chat sidecar.

A package only so that the parent can import :mod:`voice_worker.worker` for the
frame format and the protocol version, rather than keeping a second copy of a
wire protocol in the module on the other end of the pipe. ``worker.py`` is also
run directly, by path, under a different interpreter -- so it never imports
anything from this package and never assumes it was imported as part of one.

Nothing here may import a Forge module, a Model Chain module, or the speech
engine at module level. ``tests/test_voice_independence.py`` asserts exactly
that: the parent has to be able to import this file to read a struct format,
and the day it cannot is the day the WebUI stops starting.
"""
