"""MiniMax H3: WanGP's prompt enhancer, run on the model this application installs.

Two files, and the split between them is the same one ``prompt_engine`` makes.
``prompt_enhancer.py`` is vendored verbatim from the WanGP MiniMax H3 module and
holds every word of the instructions; ``enhancer.py`` is the calling convention
around them, ported from ``wgp.py`` and ``shared/prompt_enhancer``. Nothing here
writes prompt text of its own.
"""
