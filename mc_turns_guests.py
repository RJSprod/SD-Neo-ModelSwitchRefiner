"""The speech guests' side of :mod:`mc_turns`, and the client it hands the Voice Box.

:mod:`mc_turns` decides whose turn it is on each card and knows its guests only
as three methods about a card -- what they hold there, whether they are working
there, and how to unload them. It never imports an engine, and no ``mc_voice_*``
module may import it (invariant I-3, ``tests/test_voice_independence.py``: the
voice side may not reach the memory side at any depth). So the two are joined
here, in a module that is neither: it registers every Voice Box engine as a
guest -- VibeVoice and Breeze, one row of :data:`mc_voice_box_engines.SPECS`
each -- and it hands the Voice Box a client through which a render asks for
its card.

Everything the Voice Box knows about turns comes through :class:`TurnClient`,
and everything the turn system knows about an engine comes through its
:class:`EngineGuest`. Both are thin on purpose: a test of the render service
replaces the client with a double, and a test of the turn system replaces the
guest with one, and neither test needs a GPU or the other module.

Only one engine is ever active (:func:`mc_voice_box_engines.select` ends the
other's workers before it switches), so only one guest ever has anything on a
card, and the image gate, the language model's wait and WanGP's lease see what
they saw when there was one engine. Two guests are registered anyway because a
guest is a name the turn system keeps per card -- the warm stay, the eviction --
and that name has to lead to the engine that is really there.
"""

from __future__ import annotations

import logging

import importlib

import mc_broker
import mc_turns
import mc_voice_box_engines as engines

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

GUEST = "VibeVoice"
"""The name VibeVoice is registered under. ``mc_voice_vibevoice.GUEST`` spells the
same word, and a test holds the two equal."""

GUESTS = {spec.guest: spec for spec in engines.SPECS}
"""Every engine's guest name, and the engine it leads to. Each adapter's
``GUEST`` spells its row's word, and a test holds them equal."""

PHASES = {"granted": mc_turns.GRANTED, "blocked": mc_turns.BLOCKED,
          "cancelled": mc_turns.CANCELLED, "done": mc_turns.DONE}
"""The turn phases the Voice Box acts on, by the words it uses for them. The
words are the turn system's own; the table exists so a test can prove it."""


class EngineGuest:
    """An engine as :mod:`mc_turns` sees it: memory per card, working or not, evictable.

    The engine's runtime is looked up by its module name on every call rather
    than held, so a runtime replaced in ``sys.modules`` -- a test's stand-in --
    is the one reached.
    """

    def __init__(self, spec=None):
        self.spec = spec or engines.spec(engines.DEFAULT_ENGINE)

    @property
    def label(self) -> str:
        """What a sentence of the turn system calls this guest: ``Breeze TTS 2``."""
        return self.spec.label

    def _runtime(self):
        return importlib.import_module(self.spec.runtime)

    def resident_bytes(self, card_uuid: str) -> int:
        return max(int(self._runtime().resident_bytes(card_uuid) or 0), 0)

    def rendering(self, card_uuid: str) -> bool:
        return bool(self._runtime().rendering(card_uuid))

    def evict(self, card_uuid: str, reason: str) -> int:
        return max(int(self._runtime().evict(card_uuid, reason) or 0), 0)


class VibeVoiceGuest(EngineGuest):
    """VibeVoice's guest, by its old name."""

    def __init__(self):
        super().__init__(engines.spec("vibevoice"))


def _guest_name(guest: str = "") -> str:
    """A registered guest's name: ``guest`` when it is one, else the active engine's."""
    wanted = str(guest or "")
    if wanted in GUESTS:
        return wanted
    try:
        return engines.active().guest
    except Exception:
        return GUEST


class TurnClient:
    """What the Voice Box may ask of the turn system, and of the machine's cards."""

    def request(self, card_uuid: str, *, need_vram: int, need_ram: int = 0,
                label: str = "", guest: str = "") -> mc_turns.Turn:
        """Ask for ``card_uuid`` on an engine's behalf. Returns at once; wait on the turn.

        ``guest`` is the engine's guest name; ``""`` is the active engine's.
        """
        name = _guest_name(guest)
        return mc_turns.request(name, card_uuid, need_vram=need_vram, need_ram=need_ram,
                                label=label or name)

    def snapshot(self) -> list[dict]:
        return mc_turns.snapshot()

    def unload(self, card_uuid: str, reason: str, guest: str = "") -> int:
        """Free a guest from ``card_uuid`` by the user's hand. Returns bytes freed.

        Through the turn system when it has the card as a warm stay, so that
        what was parked for the guest comes back and the lease is released;
        straight to the runtime of ``guest`` (the active engine's for ``""``)
        when it does not know the card, which is a worker somebody started
        outside any turn.
        """
        state = mc_turns._state(card_uuid)
        if state is not None and (state.warm or state.busy):
            return mc_turns.end_warm(state, reason)
        return EngineGuest(GUESTS[_guest_name(guest)]).evict(card_uuid, reason)

    def cards(self) -> list[dict]:
        """The machine's NVIDIA cards, each with its role on this machine.

        ``uuid`` is spelled the way nvidia-smi spells it (``GPU-8-4-4-4-12``),
        which is the spelling ``CUDA_VISIBLE_DEVICES`` accepts, so a worker
        started for a card from this list lands on that card and no other.
        """
        import mc_memory

        image = ""
        try:
            image = mc_memory._uuid_key(mc_broker.image_device_uuid())
        except Exception:
            logger.debug("Model Chain: the image card's identity could not be read",
                         exc_info=True)
        wangp = ""
        try:
            import mc_wangp

            found = mc_wangp.presence()
            if found.available and found.uuid:
                wangp = mc_memory._uuid_key(found.uuid)
        except Exception:
            logger.debug("Model Chain: Mini Paint's card could not be read", exc_info=True)

        rows = []
        for index, key, name in _physical_cards():
            rows.append({
                "uuid": spell_uuid(key),
                "key": key,
                "index": index,
                "name": name or f"GPU {index}",
                "image_card": bool(key) and key == image,
                "wangp_card": bool(key) and key == wangp,
            })
        return rows


def _physical_cards() -> list[tuple[int, str, str]]:
    """``(physical index, uuid key, name)`` per card, in physical order."""
    import mc_memory

    topology = mc_memory._cards() or {}
    by_index = {}
    for key, index in (topology.get("by_uuid") or {}).items():
        by_index[int(index)] = str(key)
    names = topology.get("names") or {}
    found = []
    for index in sorted(set(by_index) | {int(i) for i in names}):
        found.append((index, by_index.get(index, ""), str(names.get(index, "") or "")))
    return found


def spell_uuid(key: str) -> str:
    """nvidia-smi's spelling of a card's UUID from the hex the topology keeps.

    The topology reduces every spelling to its hex digits so that torch's and
    nvidia-smi's agree; the driver wants them back with the hyphens and the
    ``GPU-`` prefix. A key that is not thirty-two digits is handed back as it
    is, prefixed, which the driver will refuse loudly rather than misread.
    """
    text = str(key or "").strip().lower()
    if text.startswith("gpu-"):
        return "GPU-" + text[4:]
    digits = "".join(character for character in text if character in "0123456789abcdef")
    if len(digits) != 32:
        return f"GPU-{text}" if text else ""
    return "GPU-" + "-".join((digits[:8], digits[8:12], digits[12:16], digits[16:20],
                              digits[20:]))


_installed = False


def install(_demo=None, app=None) -> None:
    """Register the guest and hand the Voice Box its client. Idempotent.

    Signature is ``script_callbacks.on_app_started``'s, which is where it is
    called from: after the settings are loaded and before any page can ask.
    """
    global _installed

    import mc_voice_box

    for name, spec in GUESTS.items():
        mc_turns.register_guest(name, EngineGuest(spec))
    mc_voice_box.use_turns(TurnClient())
    if not _installed:
        logger.info("Model Chain: %s are registered as guests of the cards",
                    " and ".join(GUESTS))
    _installed = True
