"""The Sopro V2 sidecar.

A package only so that the parent can import :mod:`sopro_worker.worker` for the
frame format and the protocol version, rather than keeping a second copy of a
wire protocol in the module on the other end of the pipe. ``worker.py`` is also
run directly, by path, under the *isolated Sopro interpreter* -- so it never
imports anything from this package and never assumes it was imported as part of
one.

Separate from :mod:`voice_worker` on purpose, and the reason is invariant I-6
rather than tidiness. Kokoro's worker runs out of two unpacked sherpa wheels;
this one runs out of a hundred and forty megabytes of PyTorch. One file that had
to be importable under both closures is one import away from a Torch runtime
reaching for sherpa, or the reverse, and neither closure would ever be
verifiable again.

The wire format is shared by *agreement* rather than by import, and
``tests/test_voice_sopro_worker.py`` holds that agreement to byte equality: a
frame written by either worker is read by the other. That is the shape of
sharing that survives two independent dependency closures.

Nothing here may import a Forge module, a Model Chain module, Torch, or Sopro at
module level. ``tests/test_voice_independence.py`` asserts exactly that: the
parent has to be able to import this file to read a struct format on an
installation where Sopro was never installed at all.
"""
