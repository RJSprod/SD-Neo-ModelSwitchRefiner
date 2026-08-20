"""Krea 2: Krea's own prompt expansion instruction, plus ordered references.

Three files, and the split between them is the one ``prompt_master.minimax``
already makes. ``expansion.txt`` is vendored verbatim from Krea's Krea 2
repository and holds every word of the base instruction; ``enhancer.py`` is the
calling convention around it and the one thing Krea does not cover, which is
what to do when the user has attached reference images; ``references.py`` is
the ordered-reference model that keeps "Image 1" meaning what the user meant by
it, all the way from the upload slot to whatever eventually generates a picture.

The separation is deliberate and is checked by the tests: upstream's text is
never edited in place to add local behaviour, and everything local is appended
as a clearly marked addendum, so a reader can always tell which half of the
system message came from Krea.
"""
