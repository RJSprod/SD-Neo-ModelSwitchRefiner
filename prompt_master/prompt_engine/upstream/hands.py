"""
hands.py — Prompt Master LD

POV HANDS. Its own module so it can be tuned or reverted without touching the
rest of the first-person law.

Written from a render that commissioned its own artifact: the script said
"the viewer's hands reach out from the bottom of the frame, fingers splayed as
they graze the air near her waist", and the frame came back with two open palms
turned toward the camera, fingers spread, touching nothing. Every rule below
targets one thing that sentence got wrong — no count, no job for the hand, and
a palm rotated the wrong way for the viewpoint.
"""

_LAW = """
POV HANDS
The viewer's hands are the only part of the viewer on screen, nearest the camera \
and the largest thing in frame while they are in it, entering from the bottom or \
a bottom corner with forearm behind them. Small distant hands read as somebody \
else's.
COUNT. Say which — one hand or both — and hold that number until they leave \
frame. Only ONE pair is ever up: the viewer's and hers are never both in frame, \
and hands resting or parked count the same as reaching ones. When she needs hers, \
the viewer's go first and you write them going.
PURPOSE. A hand in frame is always doing something to something already named — \
gripping, pressing, lifting, pulling, resting on, sliding along it. Reaching \
toward nothing or grazing the air is a hand with no job, and it renders as a \
floating object; nothing to hold means keep it out of frame.
GEOMETRY. You are looking down your own arms, so the BACK of the hand faces the \
view and the fingers point away. An open palm turned back at the camera is \
twisted the wrong way for this viewpoint, and splayed fingers held up is the pose \
that comes back melted. Contact also hides fingers: a hand closed round \
something, flat on a surface or half out of frame has less of itself exposed to \
go wrong.
WHOSE HANDS. They are a {vg}'s. Say so the FIRST time they appear, as a \
possessive on the hands ("a {vg}'s hands"), inside the opening two sentences \
whenever they are in frame, and never as a person standing there — nor in the \
required opening words, where a bare "{vg}" reads as a body in shot. \
Then show it: size against the frame, finger width, knuckle and tendon, forearm \
hair or none, nail length — chosen once and held. Hands with only a skin tone \
read as nobody's.
EXIT. Hands leave by an edge, in writing — never by fading out or going \
unmentioned."""

# Visual artifact names, in the same register as negative.py: what the failure
# looks like, never what it means.
_NEG = ("palm facing camera, splayed fingers, floating hands, "
        "hands without arms, oversized hands, tangled fingers")


def hand_law(pov: str) -> str:
    p = (pov or "off").strip().lower()
    if p not in ("male", "female"):
        return ""
    return _LAW.format(vg="man" if p == "male" else "woman")


def hand_negative(pov: str) -> str:
    return _NEG if (pov or "off").strip().lower() in ("male", "female") else ""
