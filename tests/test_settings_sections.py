"""The Settings page: one section per area, every description short.

The user's word on the old page -- "way too complicated ... less text that is
more clear, and better organization" -- without changing what any setting
does: the keys, defaults, components and choices are what they were, so every
saved value is read as before. What changed is where the page draws a setting
and what it says under it.
"""

from __future__ import annotations

import re

import pytest

import mc_assistant_settings
import mc_batch_once
import mc_broker
import mc_llm_paths
import mc_llm_studio
import mc_memory
import mc_plan
import mc_progress
import mc_turns
import mc_wangp

LONGEST_DESCRIPTION = 400
"""Characters. The old page's longest ran past 900."""


@pytest.fixture
def registered(host):
    import model_chain

    return model_chain, host.shared.options_templates


def section_of(templates, key):
    return templates[key].section


class TestTheSections:
    def test_every_section_has_its_own_id_and_is_drawn(self, registered):
        model_chain, _templates = registered
        ids = [section[0] for section in model_chain.SECTIONS]
        assert len(set(ids)) == len(ids)
        assert all(section[0] for section in model_chain.SECTIONS)
        assert model_chain.VOICE_SECTION not in model_chain.SECTIONS

    def test_every_title_starts_with_the_extensions_name(self, registered):
        model_chain, _templates = registered
        for section in model_chain.SECTIONS:
            assert section[1].startswith("Model Chain"), section

    @pytest.mark.parametrize("key, section", [
        ("model_chain_save_stage1", "SETTINGS_SECTION"),
        ("model_chain_ram_budget_gb", "SETTINGS_SECTION"),
        (mc_memory.OPT_PRELOAD, "SETTINGS_SECTION"),
        (mc_memory.OPT_RESTORE_ENCODERS, "SETTINGS_SECTION"),
        (mc_batch_once.OPT_ENABLED, "SETTINGS_SECTION"),
        (mc_memory.OPT_VRAM_RESERVE, "SETTINGS_SECTION"),
        (mc_progress.OPT_PROGRESS, "PROGRESS_SECTION"),
        (mc_llm_studio.OPT_ENABLE, "LLM_SECTION"),
        (mc_broker.OPT_MODE, "LLM_SECTION"),
        (mc_plan.OPT_CAP_MODE, "LLM_SECTION"),
        (mc_wangp.OPT_MODE, "LLM_SECTION"),
        (mc_llm_paths.OPT_ROOT, "LLM_SECTION"),
        (mc_turns.OPT_IMAGE_RETURN, "VOICE_BOX_SECTION"),
        (mc_turns.OPT_KEEP_WARM, "VOICE_BOX_SECTION"),
        (mc_assistant_settings.OPT_ENABLED, "ASSISTANT_SECTION"),
        (mc_assistant_settings.OPT_AVATARS, "ASSISTANT_SECTION"),
    ])
    def test_a_setting_is_drawn_in_its_areas_section(self, registered, key, section):
        model_chain, templates = registered
        assert section_of(templates, key) == getattr(model_chain, section)

    def test_the_page_and_progress_sections_hold_what_they_say(self, registered):
        model_chain, templates = registered
        assert section_of(templates, model_chain.OPT_HIDE_FOOTER) == model_chain.PAGE_SECTION
        assert section_of(templates, model_chain.OPT_QUIET_NIGHT) == model_chain.PAGE_SECTION
        for key in (model_chain.OPT_SMOOTH, model_chain.OPT_STYLE_ENABLE,
                    model_chain.OPT_STYLE_THEME, model_chain.OPT_STYLE_COMPLETE):
            assert section_of(templates, key) == model_chain.PROGRESS_SECTION


class TestTheDescriptions:
    def ours(self, registered):
        model_chain, templates = registered
        sections = set(model_chain.SECTIONS)
        return {key: info for key, info in templates.items() if info.section in sections}

    def test_every_setting_has_one(self, registered):
        for key, info in self.ours(registered).items():
            assert (info.comment_after or "").strip(), f"{key} says nothing about itself"

    def test_none_runs_long(self, registered):
        long = {key: len(re.sub(r"<[^>]+>", "", info.comment_after or ""))
                for key, info in self.ours(registered).items()}
        over = {key: n for key, n in long.items() if n > LONGEST_DESCRIPTION}
        assert over == {}, over

    def test_a_description_is_a_sentence_or_two_not_a_paragraph(self, registered):
        for key, info in self.ours(registered).items():
            text = re.sub(r"<[^>]+>", "", info.comment_after or "")
            # Sentences end in ". " here; a trailing one carries no stop.
            assert text.count(". ") <= 3, f"{key}: {text}"

    def test_nothing_changed_hands(self, registered):
        """The regrouping moved controls between sections and nothing else."""
        _model_chain, templates = registered
        assert templates[mc_memory.OPT_PRELOAD].default is mc_memory.PRELOAD_DEFAULT
        assert templates[mc_memory.OPT_PIN_ENCODERS].default is True
        assert templates[mc_broker.OPT_MODE].default == mc_broker.MODE_HYBRID
        assert templates[mc_turns.OPT_KEEP_WARM].default == mc_turns.WARM_EVERY
        assert templates[mc_assistant_settings.OPT_ENABLED].default is True
        assert templates[mc_llm_paths.OPT_ROOT].default == ""
