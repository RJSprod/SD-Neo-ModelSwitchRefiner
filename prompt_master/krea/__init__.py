"""Krea 2: Krea's own instruction, ordered references, and local art direction.

``expansion.txt`` is vendored verbatim from Krea's Krea 2 repository and holds
every word of the base instruction. ``enhancer.py`` is the calling convention
around it and the one thing Krea does not cover, which is what to do when the
user has attached reference images. ``references.py`` is the ordered-reference
model that keeps "Image 1" meaning what the user meant by it, all the way from
the upload slot to whatever eventually generates a picture.

The other three are Creative Mode, and they are the reason this package now has
opinions about *content*:

* ``creativity/`` is a versioned, data-only vocabulary -- ten axes, 164 variant
  families, four written expression tiers each, plus the activation,
  compatibility and anti-repetition policies. Vendored whole, with its own
  provenance in ``CREATIVITY_LIBRARY_SOURCE.txt``.
* ``library.py`` reads and validates that package and hands back typed objects.
* ``director.py`` chooses one art-direction brief out of it, with a seeded PRNG
  and no model whatsoever.
* ``variation.py`` maps the same Creativity position to the writer's sampling.

The whole point of the split is where the line falls. Choosing what to vary is
the most tempting possible reason to reach into ``expansion.txt`` and add "be
more adventurous" to it, and none of these may: the Director's output travels in
the *user* turn under its own label, ``variation.py`` cannot read a file or
import ``enhancer``, and upstream's text is never edited in place. A reader of a
transcript can always tell which half of the system message came from Krea --
because all of it did.
"""
