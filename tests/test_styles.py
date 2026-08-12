"""Style library integration (sections 5.2, 5.3)."""

from __future__ import annotations

import mc_styles


class TestAvailableStyles:
    def test_lists_names_from_the_live_store(self, style_store):
        assert mc_styles.available_styles() == ["Cinematic", "Detailed", "WithLora"]

    def test_sees_a_style_added_mid_session(self, style_store):
        """Section 5.3: no WebUI restart needed to pick up a new style."""
        from collections import namedtuple

        PromptStyle = namedtuple("PromptStyle", "name prompt negative_prompt path")
        style_store.styles["Fresh"] = PromptStyle("Fresh", "freshly added", "", None)

        assert "Fresh" in mc_styles.available_styles()

    def test_reload_rereads_the_store(self, style_store):
        mc_styles.reload_styles()
        assert style_store.reloaded == 1

    def test_degrades_to_empty_when_the_store_is_missing(self, host):
        host.shared.prompt_styles = None
        assert mc_styles.available_styles() == []


class TestApply:
    def test_placeholder_substitution(self, style_store):
        positive, _ = mc_styles.apply("a castle", "", ["Cinematic"])
        assert positive == "cinematic still of a castle, film grain"

    def test_style_without_placeholder_appends(self, style_store):
        positive, _ = mc_styles.apply("a castle", "", ["Detailed"])
        assert positive == "a castle, highly detailed"

    def test_applies_to_negative_prompt_too(self, style_store):
        """Section 5.2: styles must apply to both components."""
        _, negative = mc_styles.apply("a castle", "lowres", ["Cinematic"])
        assert negative == "lowres, cartoon, anime"

    def test_multiple_styles_apply_in_listed_order(self, style_store):
        positive, _ = mc_styles.apply("a castle", "", ["Cinematic", "Detailed"])
        assert positive == "cinematic still of a castle, film grain, highly detailed"

        reversed_order, _ = mc_styles.apply("a castle", "", ["Detailed", "Cinematic"])
        assert reversed_order == "cinematic still of a castle, highly detailed, film grain"
        assert reversed_order != positive

    def test_lora_tag_inside_a_style_survives_expansion(self, style_store):
        """Section 5.4: a saved style containing a LoRA tag must expand intact.

        Expansion happens first; the resulting prompt is then handed to the
        host's extra-networks parser, so the tag has to still be there.
        """
        positive, _ = mc_styles.apply("a castle", "", ["WithLora"])
        assert positive == "a castle <lora:filmgrain:0.6>"
        assert "<lora:filmgrain:0.6>" in positive

    def test_empty_selection_is_a_no_op(self, style_store):
        assert mc_styles.apply("a castle", "lowres", []) == ("a castle", "lowres")
        assert mc_styles.apply("a castle", "lowres", None) == ("a castle", "lowres")

    def test_unknown_style_name_is_ignored_rather_than_fatal(self, style_store):
        positive, negative = mc_styles.apply("a castle", "lowres", ["NoSuchStyle"])
        assert positive == "a castle"
        assert negative == "lowres"


class TestPruneSelection:
    def test_splits_known_from_missing(self, style_store):
        kept, missing = mc_styles.prune_selection(["Cinematic", "Gone", "Detailed"])
        assert kept == ["Cinematic", "Detailed"]
        assert missing == ["Gone"]

    def test_preserves_order_of_the_kept_styles(self, style_store):
        kept, _ = mc_styles.prune_selection(["Detailed", "Cinematic"])
        assert kept == ["Detailed", "Cinematic"]

    def test_empty_selection(self, style_store):
        assert mc_styles.prune_selection([]) == ([], [])
        assert mc_styles.prune_selection(None) == ([], [])
