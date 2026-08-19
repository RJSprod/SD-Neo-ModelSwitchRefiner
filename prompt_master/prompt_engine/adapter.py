"""Thin adapter between the standalone app and the vendored upstream engine.

This module contains NO prompt text. Every rule, budget, negative term and
output contract comes from ``prompt_engine.upstream``; the job here is only to
hand upstream the values it expects and hand the caller back what upstream
returned.

The one addition it makes is the motion preset, and it makes it at the seams
rather than inside: a directive appended after upstream's finished system
prompt, and terms merged into the extra negatives upstream already accepts as
an input. On the default preset both are empty, so a default build is upstream
byte-for-byte — which ``test_prompt_engine.py`` asserts against ``build_system``
directly.

The call order mirrors ``upstream/routes.py::_build_messages`` exactly, and for
one reason that module records: ``send_vision`` must be decided BEFORE
``build_system``, because the i2v opener changes completely depending on whether
the still is actually on the wire. Computing it afterwards is how the system
prompt ended up insisting "Frame one is the attached image" while the user turn
said the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_master.core.models import PromptRequest

from . import motion
from .upstream import brain
from .upstream import negative as neg
from .upstream.imaging import b64_to_pil, jpeg_b64, style_hint

# Upstream's own vision policy, from routes.py's single call site.
VISION_MAX_SIDE = 768


@dataclass(slots=True)
class EngineOutput:
    """Everything one generation needs, all of it produced upstream."""

    system: str
    user: str
    messages: list[dict[str, Any]]
    base_negative: str
    max_tokens: int
    frames: int
    send_vision: bool
    style_hint: str
    word_budget: tuple[int, int]
    beat_budget: tuple[int, int]
    dialogue_lines: int
    fmt: str
    meta: dict[str, Any] = field(default_factory=dict)


class VisionUnavailable(RuntimeError):
    """Raised when an i2v request carries an image the model cannot receive.

    The standalone app never degrades an i2v request to text-only behind the
    user's back: upstream's blind opener exists for that case and says out loud
    that the still is not visible, so a silent downgrade would produce a prompt
    describing a scene nobody chose.
    """


class PromptEngine:
    """Upstream engine, standalone calling convention."""

    def build(self, request: PromptRequest, *, vision_available: bool = True) -> EngineOutput:
        mode = (request.video_mode or "i2v").strip().lower()
        is_i2v = mode == "i2v"

        pil = b64_to_pil(request.image_data_url or "")
        if is_i2v and request.image_data_url and pil is None:
            raise VisionUnavailable("The attached image could not be decoded.")

        # Medium cue for I2V, from upstream's own detector over the same pixels.
        hint = ""
        if is_i2v and pil is not None:
            hint = style_hint(pil, name=request.image_name or "")

        send_vision = is_i2v and pil is not None and bool(vision_available)
        if is_i2v and pil is not None and not vision_available:
            raise VisionUnavailable(
                "This request has an image but the vision projector is not "
                "loaded. Configure the projector in Models and Hardware, or "
                "switch to T2V."
            )

        system = brain.build_system(
            mode=mode,
            pov=request.pov,
            accent=request.accent,
            accent_strength=request.accent_strength,
            dialogue=request.dialogue,
            wardrobe=request.wardrobe,
            undress=bool(request.undress),
            seed=int(request.seed or 0),
            intent=request.intent,
            camera=request.camera,
            transition=request.transition,
            music=request.music,
            music_bg=bool(request.music_bg),
            lexicon=request.lexicon,
            fmt=request.fmt,
            fps=int(request.fps or 24),
            seconds=float(request.seconds or 12),
            style=request.style,
            style_hint=hint,
            has_image=send_vision,
        )
        # After upstream has finished, never woven through it: removing the
        # preset removes the whole of the difference it makes.
        system = motion.applied(system, request.motion)
        user = brain.build_user(
            intent=request.intent,
            mode=mode,
            has_image=send_vision,
            style_hint=hint,
        )

        wsec = brain.write_seconds(request.seconds)
        pct = brain.talk_pct(request.dialogue)
        nb_lo, nb_hi = brain.beat_budget(wsec)

        return EngineOutput(
            system=system,
            user=user,
            messages=self._messages(system, user, pil if send_vision else None),
            base_negative=self.base_negative(request),
            max_tokens=brain.max_tokens(request.seconds, request.dialogue, fmt=request.fmt),
            frames=brain.frame_count(request.fps, request.seconds),
            send_vision=send_vision,
            style_hint=hint,
            word_budget=brain.word_budget(wsec, request.dialogue),
            beat_budget=(nb_lo, nb_hi),
            dialogue_lines=brain.dialogue_lines(wsec, pct, beats=nb_lo),
            fmt=(request.fmt or "flowing").strip().lower(),
            meta={
                "mode": mode,
                "talk_pct": pct,
                "write_seconds": wsec,
                "width": int(request.output_width),
                "height": int(request.output_height),
                "fps": int(request.fps or 24),
            },
        )

    @staticmethod
    def _messages(system: str, user: str, pil) -> list[dict[str, Any]]:
        """The upstream multimodal shape: image part first, then the text part."""
        if pil is not None:
            content: Any = [
                {"type": "image_url",
                 "image_url": {"url": jpeg_b64(pil, max_side=VISION_MAX_SIDE)}},
                {"type": "text", "text": user},
            ]
        else:
            content = user
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def retry_text_only(self, request: PromptRequest) -> EngineOutput:
        """Rebuild with the image stripped, after a vision call actually failed.

        Upstream does the same on an empty or 400 vision response, and this is
        not a silent downgrade: the rebuilt brief uses upstream's blind i2v
        opener, which tells the model in as many words that it cannot see the
        still and must write motion only. Callers must surface that to the user.
        """
        stripped = replace_image(request, None)
        return self.build(stripped, vision_available=False)

    # ── negatives ────────────────────────────────────────────────────────────

    @staticmethod
    def base_negative(request: PromptRequest, auto: str = "") -> str:
        return brain.build_negative(
            pov=request.pov,
            dialogue=request.dialogue,
            undress=bool(request.undress),
            fmt=request.fmt,
            transition=request.transition,
            intent=request.intent,
            # The preset's terms arrive as extra terms — the same input the user
            # types into — so upstream's dedupe sees them beside its own banks.
            extra=motion.with_terms(request.motion, request.negative_extra),
            camera=request.camera,
            style=request.style,
            mode=request.video_mode,
            auto=auto,
        )

    @staticmethod
    def sampling(request: PromptRequest) -> tuple[float, float]:
        """``(temperature, top_p)`` for the writer pass. The smart-negative pass
        keeps upstream's own cooler numbers and does not consult this."""
        chosen = motion.preset(request.motion)
        return chosen.temperature, chosen.top_p

    @staticmethod
    def smart_negative_messages(script: str) -> list[dict[str, Any]]:
        return neg.auto_messages(script)

    @staticmethod
    def clean_smart_negative(raw: str, script: str, limit: int = 14) -> str:
        return neg.clean_auto(raw, script=script, limit=limit)

    def merge_negative(self, request: PromptRequest, auto: str = "") -> str:
        """Final negative: the gated banks with the auto terms folded in.

        The auto terms go through ``build_negative`` rather than being appended,
        because upstream's dedupe has to see them alongside the static banks —
        a term stated twice in a negative weights that concept twice.
        """
        return self.base_negative(request, auto=auto)

    def run_smart_negative(self, script: str, chat_stream, *, seed=None,
                           max_tokens: int = 220, limit: int = 14) -> str:
        """Upstream's second pass, including its never-raises contract."""
        return neg.run_auto(script, chat_stream, seed=seed,
                            max_tokens=max_tokens, limit=limit)

    # ── transport cleanup ────────────────────────────────────────────────────

    @staticmethod
    def clean_positive(text: str) -> str:
        """Strip reasoning leaks, fences and meta-planning — upstream's own."""
        return brain.clean_script(text)


def replace_image(request: PromptRequest, data_url: str | None) -> PromptRequest:
    """Copy of a request with a different image. Kept here so the adapter never
    mutates a request the UI still holds."""
    from dataclasses import replace

    return replace(request, image_data_url=data_url)
