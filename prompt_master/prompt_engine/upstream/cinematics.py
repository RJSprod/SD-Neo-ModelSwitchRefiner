"""
cinematics.py — Claude Prompt LD
CAMERA and TRANSITION laws for the brief. Each entry is HOW; the intent is WHAT.
"off" adds nothing.
"""

_CAM_ENFORCE = (
    " MANDATORY: this camera behavior is the law for the whole shot — it "
    "overrides the generic single-move default. Write the frame's motion (or "
    "stillness) in plain physical words. At least one clear camera beat per "
    "short paragraph. Do not invent a different camera style."
)

CAMERA = {
    "off": "",
    "handheld_restless": (
        "CAMERA — restless handheld. Frame never fully settles: live micro-drift, "
        "small reframes chasing action. Documentary hand, not tripod. A locked "
        "rock-steady frame fails this mode."
    ),
    "hunting": (
        "CAMERA — hunting the action. Moves with intent: swings low and rises, "
        "arcs for profile, drops close then pulls wide, whip-corrects overshoots. "
        "Name a specific camera move in EACH beat — never static while the "
        "subject moves."
    ),
    "arms_reach_pov": (
        "CAMERA — arm's length, body-borne: live sway and bob, subject filling "
        "most of the frame. Closeness = subject growing against the view, not a "
        "mechanical push-in. Keep the sway alive."
    ),
    "slow_push": (
        "CAMERA — one slow continuous push across the WHOLE shot. Distance "
        "shrinks beat by beat; end tighter than start. No pull-backs or mid-shot "
        "static holds — write the tightening in more than one beat."
    ),
    "slow_pull": (
        "CAMERA — one slow continuous pull across the WHOLE shot. Reveal more "
        "space; end wider than start. No cancelling push-ins; write the widening "
        "in more than one beat."
    ),
    "orbit": (
        "CAMERA — slow arc around the subject. Curved path, background slides, "
        "subject centered. Write changing angle and sliding background at least "
        "twice — not one jump to a new angle."
    ),
    "locked_off": (
        "CAMERA — locked and still. Frame does not move; only subject and light "
        "move inside it. OVERRIDES any default camera move — no push, drift, "
        "orbit, or reframe."
    ),
    "slow_rise": (
        "CAMERA — slow vertical rise, low to high across the shot. Write rising "
        "viewpoint in more than one beat — start lower, end higher."
    ),
    "circling_close": (
        "CAMERA — circling tight. Held close, drifting around subject; background "
        "soft blur, subject sharp. Keep close distance and slow circle; do not "
        "jump wide."
    ),
    "float": (
        "CAMERA — slow untethered float. Glides on no fixed axis, weightless "
        "angles a locked rig would not hold. Write free drift — not one static setup."
    ),
}

CAMERA_LABELS = {
    "off": "Camera — default",
    "handheld_restless": "Restless handheld",
    "hunting": "Hunting for action",
    "arms_reach_pov": "Handheld at arm's reach",
    "slow_push": "Slow push in",
    "slow_pull": "Slow pull back",
    "orbit": "Slow orbit",
    "locked_off": "Locked / still",
    "slow_rise": "Slow rise",
    "circling_close": "Circling close",
    "float": "Untethered float",
}

_T_HEAD = (
    "TRANSITION — intent supplies two states (or two phases). Write ONE "
    "continuous prompt joined by this seam, never two unrelated clips. If "
    "intent is a single beat, invent a mid-shot change of pose/distance/energy "
    "so the seam has work. Seam gets its OWN beat (blank line before and after):"
)

TRANSITION = {
    "off": "",
    "morph": (
        f"{_T_HEAD} first state melts into second in one continuous warp — "
        "features, pose, setting flow mid-frame. Begin near midpoint; several "
        "written stages of bending/softening. Name what changes as it changes."
    ),
    "hard_cut": (
        f"{_T_HEAD} clean hard cut on a beat or gesture — state one, then one "
        "instant switch. Write the cut as its own line; everything after is state two."
    ),
    "whip_pan": (
        f"{_T_HEAD} fast whip blurs to streaks; settle into state two. Write the "
        "whip, then the new frame resolving."
    ),
    "match_cut": (
        f"{_T_HEAD} a shape or motion in state one lines up with the same in "
        "state two; change on that match. Name the matching element both sides."
    ),
    "push_through": (
        f"{_T_HEAD} push into a dark or bright detail until the frame fills, "
        "emerge already in state two. The fill is the seam."
    ),
    "smash_zoom": (
        f"{_T_HEAD} violent zoom snaps in (or out); scale change lands in state "
        "two. Write the snap and new distance."
    ),
    "dissolve": (
        f"{_T_HEAD} soft double-exposure blend; both briefly visible, then state "
        "two. Name what overlaps during the blend."
    ),
    "spin_blur": (
        f"{_T_HEAD} frame spins to smear, unwinds into state two at a new angle. "
        "Write the spin, then the settle."
    ),
    "flash": (
        f"{_T_HEAD} hard white bloom clears into state two. Write the flash and "
        "what is revealed after."
    ),
    "pull_reveal": (
        f"{_T_HEAD} pull back or pan off state one uncovers state two in the same "
        "continuous space — no cut; the reveal is the seam."
    ),
}

TRANSITION_LABELS = {
    "off": "Transition — none",
    "morph": "Morph / melt",
    "hard_cut": "Hard cut",
    "whip_pan": "Whip pan",
    "match_cut": "Match cut",
    "push_through": "Push through detail",
    "smash_zoom": "Smash zoom",
    "dissolve": "Dissolve / double-exposure",
    "spin_blur": "Spin blur",
    "flash": "Flash cut",
    "pull_reveal": "Pull-back reveal",
}

CAMERA_KEYS = list(CAMERA.keys())
TRANSITION_KEYS = list(TRANSITION.keys())

_NEG_CAM = {
    "handheld_restless": "locked tripod frame, perfectly static camera, steadicam glide",
    "hunting": "static locked frame, tripod hold, unmoving camera",
    "arms_reach_pov": "wide distant master shot, locked tripod, clinical static frame",
    "slow_push": "static frame, pull back, widening shot, camera retreat",
    "slow_pull": "push in, tightening close-up only, static locked frame",
    "orbit": "static single angle, locked frontal hold, no parallax",
    "locked_off": "handheld shake, drifting camera, push-in, orbit, reframing",
    "slow_rise": "static eye-level hold, descending camera, no vertical move",
    "circling_close": "wide master shot, static distant frame, jump to wide",
    "float": "locked tripod, rigid axial move only, static frame",
}


def camera_law(key: str) -> str:
    k = (key or "off").strip().lower()
    body = CAMERA.get(k, "")
    if not body:
        return ""
    return body + _CAM_ENFORCE


def transition_law(key: str) -> str:
    return TRANSITION.get((key or "off").strip().lower(), "")


def camera_negative(key: str) -> str:
    return _NEG_CAM.get((key or "off").strip().lower(), "")
