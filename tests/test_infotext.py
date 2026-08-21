"""Infotext write and restore (section 7)."""

from __future__ import annotations

import pytest

import mc_infotext
from conftest import parse_generation_parameters, quote, unquote


def build(**overrides):
    params = dict(
        target="fluxKlein9B.safetensors",
        prompt_mode="Replace",
        prompt="a serene lake at dawn",
        negative="blurry",
        styles=["Cinematic"],
        seed_mode="Inherit",
        seed_offset=0,
        fixed_seed=-1,
        cfg=1.0,
        steps=20,
        sampler=mc_infotext.INHERIT,
        scheduler=mc_infotext.INHERIT,
        denoise=0.35,
        size_multiplier=1.5,
        stage1_size="1024x1024",
    )
    params.update(overrides)
    return mc_infotext.build_params(**params)


class TestBuildParams:
    def test_writes_every_specified_key(self):
        params = build()
        for key in (
            mc_infotext.ENABLED,
            mc_infotext.TARGET,
            mc_infotext.PROMPT_MODE,
            mc_infotext.PROMPT,
            mc_infotext.NEGATIVE,
            mc_infotext.STYLES,
            mc_infotext.SEED_MODE,
            mc_infotext.CFG,
            mc_infotext.STEPS,
            mc_infotext.DENOISE,
            mc_infotext.SIZE_MULTIPLIER,
            mc_infotext.STAGE1_SIZE,
        ):
            assert key in params

    def test_every_key_is_namespaced(self):
        assert all(key.startswith("Model Chain") for key in build())

    def test_seed_offset_is_omitted_when_zero(self):
        """Section 7.2 asks for the offset to be omitted at its default."""
        assert mc_infotext.SEED_OFFSET not in build(seed_mode="Offset", seed_offset=0)
        assert build(seed_mode="Offset", seed_offset=5)[mc_infotext.SEED_OFFSET] == 5

    def test_offset_is_not_recorded_when_the_mode_does_not_use_it(self):
        assert mc_infotext.SEED_OFFSET not in build(seed_mode="Inherit", seed_offset=5)

    def test_fixed_seed_is_recorded_only_in_fixed_mode(self):
        assert mc_infotext.SEED_FIXED not in build(seed_mode="Inherit")
        assert build(seed_mode="Fixed", fixed_seed=12345)[mc_infotext.SEED_FIXED] == 12345

    def test_inherited_sampler_is_not_recorded(self):
        assert mc_infotext.SAMPLER not in build(sampler=mc_infotext.INHERIT)
        assert build(sampler="Euler")[mc_infotext.SAMPLER] == "Euler"

    def test_inherited_scheduler_is_not_recorded(self):
        assert mc_infotext.SCHEDULER not in build(scheduler=mc_infotext.INHERIT)
        assert build(scheduler="Karras")[mc_infotext.SCHEDULER] == "Karras"

    def test_sampler_and_scheduler_are_independent(self):
        params = build(sampler="Euler", scheduler=mc_infotext.INHERIT)
        assert params[mc_infotext.SAMPLER] == "Euler"
        assert mc_infotext.SCHEDULER not in params

    def test_styles_are_recorded_as_names_for_readability(self):
        assert build(styles=["Cinematic", "Detailed"])[mc_infotext.STYLES] == "Cinematic, Detailed"

    def test_no_styles_key_when_none_selected(self):
        assert mc_infotext.STYLES not in build(styles=[])

    def test_accepts_per_image_prompt_lists(self):
        """create_infotext indexes list values, giving per-image accuracy."""
        params = build(prompt=["first", "second"], negative=["", ""])
        assert params[mc_infotext.PROMPT] == ["first", "second"]


class TestQuotingRoundTrip:
    """Values containing commas, colons or newlines must not break parsing."""

    @pytest.mark.parametrize(
        "value",
        [
            "a serene lake at dawn",
            "a lake, a mountain, a castle",
            "cinematic still of a castle <lora:filmgrain:0.6>",
            "<lora:sdxl_detail:0.8> <lora:other:1.2>",
            "line one\nline two",
            'quotes "inside" the prompt',
            "colons: everywhere: here",
        ],
    )
    def test_value_survives_quote_unquote(self, value):
        assert unquote(quote(value)) == value

    def test_lora_tag_survives_a_full_infotext_round_trip(self):
        """Section 5.4: LoRA tags must restore intact on paste."""
        prompt = "a castle at dusk <lora:filmgrain:0.6>, moody"
        params = build(prompt=prompt)

        line = ", ".join(f"{k}: {quote(v)}" for k, v in params.items())
        infotext = f"stage one prompt\nNegative prompt: bad\n{line}"

        parsed = parse_generation_parameters(infotext)
        assert parsed[mc_infotext.PROMPT] == prompt

    def test_multiline_prompt_survives_the_round_trip(self):
        prompt = "first line\nsecond line"
        params = build(prompt=prompt)
        line = ", ".join(f"{k}: {quote(v)}" for k, v in params.items())
        parsed = parse_generation_parameters(f"main\n{line}")
        assert parsed[mc_infotext.PROMPT] == prompt

    def test_commas_in_a_prompt_do_not_split_into_extra_keys(self):
        params = build(prompt="a, b, c, d", negative="e, f")
        line = ", ".join(f"{k}: {quote(v)}" for k, v in params.items())
        parsed = parse_generation_parameters(f"main\n{line}")
        assert parsed[mc_infotext.PROMPT] == "a, b, c, d"
        assert parsed[mc_infotext.NEGATIVE] == "e, f"

    def test_numeric_values_round_trip(self):
        params = build(cfg=1.0, steps=24, denoise=0.42, size_multiplier=1.25)
        line = ", ".join(f"{k}: {quote(v)}" for k, v in params.items())
        parsed = parse_generation_parameters(f"main\n{line}")
        assert float(parsed[mc_infotext.CFG]) == pytest.approx(1.0)
        assert int(parsed[mc_infotext.STEPS]) == 24
        assert float(parsed[mc_infotext.DENOISE]) == pytest.approx(0.42)
        assert float(parsed[mc_infotext.SIZE_MULTIPLIER]) == pytest.approx(1.25)


class TestParseStyles:
    def test_splits_a_recorded_list(self):
        assert mc_infotext.parse_styles("Cinematic, Detailed") == ["Cinematic", "Detailed"]

    def test_tolerates_ragged_spacing(self):
        assert mc_infotext.parse_styles(" Cinematic ,Detailed ,, ") == ["Cinematic", "Detailed"]

    def test_empty_values(self):
        assert mc_infotext.parse_styles("") == []
        assert mc_infotext.parse_styles(None) == []

    def test_accepts_an_already_parsed_list(self):
        assert mc_infotext.parse_styles(["Cinematic"]) == ["Cinematic"]


class TestPasteFieldNames:
    def test_covers_every_written_key(self):
        written = set(build(seed_mode="Fixed", fixed_seed=1, sampler="Euler", scheduler="Karras"))
        declared = set(mc_infotext.paste_field_names())
        assert written <= declared, f"keys written but never pasted: {written - declared}"


class TestTheCreativeConfiguration:
    """The axis configuration written into a PNG, and read back out of one.

    Short by design: Natural axes are absent, because absence is what Natural
    means, and ids rather than labels, because a label is display text a package
    update may rewrite while an id is stable by the package's own contract.
    """

    def test_only_the_axes_that_direct_anything_are_written(self):
        line = mc_infotext.creative_axes(
            {"medium": "vary", "texture": "natural", "mood": "fixed"},
            {"mood": "monumental"})

        assert "texture" not in line
        assert line == "medium=vary, mood=fixed:monumental"

    def test_it_round_trips(self):
        modes = {"medium": "vary", "mood": "fixed"}
        fixed = {"mood": "monumental"}
        back_modes, back_fixed = mc_infotext.parse_creative_axes(
            mc_infotext.creative_axes(modes, fixed))

        assert (back_modes, back_fixed) == (modes, fixed)

    def test_exclusions_round_trip(self):
        excluded = {"lighting": ["harsh_noon", "golden_hour"], "texture": ["gloss"]}
        line = mc_infotext.creative_exclusions(excluded)

        assert mc_infotext.parse_creative_exclusions(line) == excluded

    def test_an_axis_with_nothing_excluded_is_not_written(self):
        assert mc_infotext.creative_exclusions({"lighting": []}) == ""

    def test_the_lines_survive_the_hosts_own_quoting(self):
        """Both carry commas and one carries a colon, which is exactly what the
        host's quote/unquote pair exists for."""
        axes = mc_infotext.creative_axes({"medium": "fixed", "mood": "vary"},
                                         {"medium": "oil_impasto"})
        excluded = mc_infotext.creative_exclusions(
            {"lighting": ["harsh_noon", "golden_hour"]})
        line = ", ".join([f"{mc_infotext.CREATIVE_AXES}: {quote(axes)}",
                          f"{mc_infotext.CREATIVE_EXCLUDED}: {quote(excluded)}"])
        parsed = parse_generation_parameters(f"a prompt\nSteps: 20, Seed: 1, {line}")

        assert parsed[mc_infotext.CREATIVE_AXES] == axes
        assert parsed[mc_infotext.CREATIVE_EXCLUDED] == excluded

    def test_a_mode_that_is_not_a_mode_is_ignored_rather_than_restored(self):
        modes, _fixed = mc_infotext.parse_creative_axes("medium=enthusiastic, mood=vary")

        assert modes == {"mood": "vary"}

    def test_nothing_recorded_reads_back_as_nothing(self):
        assert mc_infotext.parse_creative_axes("") == ({}, {})
        assert mc_infotext.parse_creative_exclusions(None) == {}

    def test_every_creative_key_is_forwarded(self):
        """The "send to txt2img" buttons forward by exact name, so a key not
        declared here simply does not arrive."""
        assert set(mc_infotext.CREATIVE_KEYS) <= set(
            mc_infotext.creative_paste_field_names())

    def test_the_creative_keys_are_namespaced_like_everything_else(self):
        for key in mc_infotext.CREATIVE_KEYS:
            assert key.startswith("Krea ")

    def test_they_do_not_collide_with_the_chain_s_own_keys(self):
        assert not set(mc_infotext.CREATIVE_KEYS) & set(mc_infotext.paste_field_names())
