"""Krea 2: Krea's own prompt expansion instruction, plus ordered references.

Four files, and the split between them is the one ``prompt_master.minimax``
already makes. ``expansion.txt`` is vendored verbatim from Krea's Krea 2
repository and holds every word of the base instruction; ``enhancer.py`` is the
calling convention around it and the one thing Krea does not cover, which is
what to do when the user has attached reference images; ``references.py`` is
the ordered-reference model that keeps "Image 1" meaning what the user meant by
it, all the way from the upload slot to whatever eventually generates a picture;
``variation.py`` is the Creativity control, which is a table of sampler settings
and deliberately nothing else.

``variation.py`` is on the near side of the same line. A creativity control is
the most tempting possible reason to reach into ``expansion.txt`` and add "be
more adventurous" to it, and that is exactly why it may not: it has no file
access, no message building and no import of ``enhancer``, so the only thing it
can change is how the model samples. What the model is *asked* stays Krea's, at
every position on the slider.

The separation is deliberate and is checked by the tests: upstream's text is
never edited in place to add local behaviour, and everything local is appended
as a clearly marked addendum, so a reader can always tell which half of the
system message came from Krea.
"""
