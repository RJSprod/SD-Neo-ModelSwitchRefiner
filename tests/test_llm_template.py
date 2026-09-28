"""Gemma 4's chat template, with the empty thought marker kept on past replies.

From the user's llama-server log, every continuing turn on Gemma 4 parted from
the cached prompt four tokens before the last reply, and read that reply again:
one turn read 394 new tokens after a reply of 331. The cause is the template's
own asymmetry -- the generation prompt ends with an empty thought block and
``strip_thinking`` removes it from every past model turn -- and the fix is a
copy of the model's template that keeps the block there, given to llama-server
as ``--chat-template-file``.

The template below is written for these tests. It has the three parts of
Gemma 4's that the asymmetry is made of -- the turn opening, ``strip_thinking``
on model turns, and the empty block at the end of the generation prompt -- and
renders the way Gemma 4's does. Checked separately against llama-server b10621
with Gemma 4's own template: the match moved from 254 of a 258-token prompt to
all 258. Every test here was checked against the decision it guards by
reverting it.
"""

from __future__ import annotations

import types

import pytest

import mc_llm_context as ctx
import mc_llm_paths
import mc_llm_runtime as runtime
import mc_llm_slot_cache as cache
import mc_llm_template as template
from test_gguf import _text, _u32, write_gguf

GEMMA4_LIKE = r"""{%- macro strip_thinking(text) -%}
    {%- set ns = namespace(result='') -%}
    {%- for part in text.split('<channel|>') -%}
        {%- if '<|channel>' in part -%}
            {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}
        {%- else -%}
            {%- set ns.result = ns.result + part -%}
        {%- endif -%}
    {%- endfor -%}
    {{- ns.result | trim -}}
{%- endmacro -%}
{{ bos_token }}
{%- for message in messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
        {{- '<|turn>' + role + '\n' }}
            {%- if role == 'model' -%}
                {{- strip_thinking(message['content']) -}}
            {%- else -%}
                {{- message['content'] | trim -}}
            {%- endif -%}
        {{- '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
    {%- if not enable_thinking | default(false) -%}
        {{- '<|channel>thought\n<channel|>' -}}
    {%- endif -%}
{%- endif -%}
"""

QWEN_LIKE = r"""{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}{%- endif -%}
"""


def a_model(where, name="model.gguf", chat_template=GEMMA4_LIKE, window=1024):
    entries = [_text("general.architecture", "llama"), _u32("llama.block_count", 4),
               _u32("llama.context_length", 8192), _u32("llama.embedding_length", 256),
               _u32("llama.attention.head_count", 4), _u32("llama.attention.head_count_kv", 2)]
    if window:
        entries.append(_u32("llama.attention.sliding_window", window))
    if chat_template is not None:
        entries.append(_text(template.TEMPLATE_KEY, chat_template))
    return write_gguf(where / name, b"".join(entries), len(entries), padding=1024)


def render(source, messages, add_generation_prompt=True, enable_thinking=False):
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(source).render(
        messages=messages, add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking, bos_token="<bos>")


FIRST = [{"role": "system", "content": "You are Ada, a reader of maps."},
         {"role": "user", "content": "Tell me about the coast."}]
REPLY = "The coast runs north for forty miles."
SECOND = FIRST + [{"role": "assistant", "content": REPLY},
                  {"role": "user", "content": "And the islands beyond it?"}]


class TestTheHistoryExtendsTheLastPrompt:
    """What llama.cpp's cache needs: after a reply, the next prompt begins with
    the last prompt followed by that reply, character for character."""

    def test_as_the_model_ships_it_the_history_parts_before_the_reply(self):
        first = render(GEMMA4_LIKE, FIRST)
        second = render(GEMMA4_LIKE, SECOND)

        assert first.endswith("<|turn>model\n<|channel>thought\n<channel|>")
        assert not second.startswith(first + REPLY)
        assert second.startswith(first[:-len("<|channel>thought\n<channel|>")])

    def test_with_the_marker_kept_it_extends_it(self):
        fixed = template.patched(GEMMA4_LIKE)

        first = render(fixed, FIRST)
        second = render(fixed, SECOND)

        assert first == render(GEMMA4_LIKE, FIRST)
        assert second.startswith(first + REPLY + "<turn|>\n<|turn>user\n")

    def test_every_past_reply_keeps_it_not_only_the_last(self):
        fixed = template.patched(GEMMA4_LIKE)
        third = render(fixed, SECOND + [{"role": "assistant", "content": "Small and green."},
                                        {"role": "user", "content": "How far?"}])

        assert third.count("<|turn>model\n<|channel>thought\n<channel|>") == 3

    def test_with_thinking_on_nothing_is_added(self):
        fixed = template.patched(GEMMA4_LIKE)

        for messages in (FIRST, SECOND):
            assert render(fixed, messages, enable_thinking=True) == render(
                GEMMA4_LIKE, messages, enable_thinking=True)

    def test_a_single_turn_request_renders_exactly_as_before(self):
        """The Krea writer and the Neutralizer share this server and send no
        past model turns: their prompts, and their caches, are untouched."""
        fixed = template.patched(GEMMA4_LIKE)

        assert render(fixed, FIRST) == render(GEMMA4_LIKE, FIRST)
        assert render(fixed, FIRST, add_generation_prompt=False) == render(
            GEMMA4_LIKE, FIRST, add_generation_prompt=False)


class TestOnlyGemma4sTemplateIsPatched:
    def test_the_marker_goes_right_after_the_turn_opening(self):
        fixed = template.patched(GEMMA4_LIKE)

        assert fixed.count(template.KEPT) == 1
        assert template.TURN_OPENING + template.KEPT in fixed
        assert fixed.replace(template.KEPT, "") == GEMMA4_LIKE

    def test_a_template_missing_any_part_is_left_alone(self):
        for missing in (template.TURN_OPENING, template.STRIPS_THINKING,
                        template.EMPTY_THOUGHT):
            assert template.patched(GEMMA4_LIKE.replace(missing, "")) is None, missing

    def test_other_templates_are_left_alone(self):
        assert template.patched(QWEN_LIKE) is None
        assert template.patched("") is None

    def test_a_template_that_opens_turns_in_two_places_is_left_alone(self):
        doubled = GEMMA4_LIKE + template.TURN_OPENING

        assert template.patched(doubled) is None

    def test_a_patched_template_is_never_patched_again(self):
        assert template.patched(template.patched(GEMMA4_LIKE)) is None


class TestTheStartIsGivenTheCopy:
    LISTS_IT = "  --chat-template-file JINJA_TEMPLATE_FILE   set custom jinja chat template\n"

    @pytest.fixture(autouse=True)
    def forget(self, host, tmp_path, monkeypatch):
        runtime._capabilities.clear()
        runtime._arm_flags([])
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        yield
        runtime._capabilities.clear()
        runtime._arm_flags([])

    @pytest.fixture
    def build(self, tmp_path, monkeypatch):
        executable = tmp_path / "llama-server"
        executable.write_text("")

        def announce(text, model=None):
            monkeypatch.setattr(
                runtime.subprocess, "run",
                lambda *args, **kwargs: types.SimpleNamespace(stdout=text, stderr=""))
            return runtime.Config(
                runtime=executable, model=model or a_model(tmp_path), mmproj=None,
                gpu_index=0, device="SYCL0", gpu_layers="all", context_size=8192,
                context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16",
                kv_type_v="f16")

        return announce

    def test_a_gemma4_template_is_written_patched_and_passed(self, build, tmp_path):
        flags = template.launch_flags(build(self.LISTS_IT))

        assert flags[0] == template.TEMPLATE_FILE_FLAG
        written = tmp_path / "root" / template.DIRNAME
        assert flags[1].startswith(str(written))
        with open(flags[1], encoding="utf-8") as handle:
            assert handle.read() == template.patched(GEMMA4_LIKE)

    def test_the_same_template_is_the_same_file(self, build):
        configuration = build(self.LISTS_IT)

        assert template.launch_flags(configuration) == template.launch_flags(configuration)

    def test_another_template_starts_as_it_always_did(self, build, tmp_path):
        configuration = build(self.LISTS_IT, model=a_model(tmp_path, "qwen.gguf", QWEN_LIKE))

        assert template.launch_flags(configuration) == []

    def test_a_model_without_a_template_starts_as_it_always_did(self, build, tmp_path):
        configuration = build(self.LISTS_IT, model=a_model(tmp_path, "bare.gguf", None))

        assert template.launch_flags(configuration) == []

    def test_a_build_without_the_flag_starts_as_it_always_did(self, build):
        assert template.launch_flags(build("  --jinja\n")) == []

    def test_the_setting_off_starts_as_it_always_did(self, build, host):
        host.shared.opts.set(template.OPT_KEEP_MARKER, False)

        assert template.launch_flags(build(self.LISTS_IT)) == []

    def test_the_launch_line_carries_it(self, build):
        configuration = build(self.LISTS_IT)

        flags = runtime._launch_flags(configuration, ctx.Placement(uma=True), None)

        assert template.TEMPLATE_FILE_FLAG in flags

    def test_a_refusal_of_it_is_not_blamed_on_the_model(self):
        assert template.TEMPLATE_FILE_FLAG in runtime.OPTIONAL_FLAGS

    def test_the_settings_page_offers_it_on_by_default(self, host):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        source = (root / "scripts" / "model_chain.py").read_text(encoding="utf-8")

        assert "mc_llm_template.OPT_KEEP_MARKER: shared.OptionInfo(\n                True," in source
        assert template.enabled() is True


class TestTheSavedCacheKnowsTheTemplate:
    """A sliding-window model's cache is saved only on the full cache, and --
    for a template that needs the marker kept -- only with the copy on the line:
    without it no prompt extends the last one, and no file could be read back."""

    def test_a_gemma4_template_needs_the_copy_to_be_saved(self, tmp_path):
        model = a_model(tmp_path)

        assert cache.resumable(model, full_cache=True, template_fixed=False) is False
        assert cache.resumable(model, full_cache=True, template_fixed=True) is True
        assert cache.resumable(model, full_cache=False, template_fixed=True) is False

    def test_a_windowed_model_whose_template_is_stable_needs_only_the_full_cache(self,
                                                                                  tmp_path):
        model = a_model(tmp_path, "qwen.gguf", QWEN_LIKE)

        assert cache.resumable(model, full_cache=True) is True
        assert cache.resumable(model, full_cache=False) is False

    def test_the_template_is_part_of_what_a_file_is_valid_for(self, tmp_path):
        runtime_file = tmp_path / "llama-server"
        runtime_file.write_text("x")
        model = a_model(tmp_path)
        copy = tmp_path / "copy.jinja"
        copy.write_text(template.patched(GEMMA4_LIKE))

        without = cache.identity(runtime_file, model, None, True, "")
        with_copy = cache.identity(runtime_file, model, None, True, "", template=copy)

        assert without != with_copy

    def test_the_start_passes_both_facts_to_the_saved_cache(self, tmp_path, monkeypatch,
                                                            host):
        seen = {}

        def launch_flags(configuration, full_cache=False, template_fixed=False):
            seen.update(full_cache=full_cache, template_fixed=template_fixed)
            return []

        monkeypatch.setattr(cache, "launch_flags", launch_flags)
        monkeypatch.setattr(runtime, "accelerator_flags",
                            lambda configuration, placement: [runtime.FULL_ATTENTION_WINDOW_FLAG])
        monkeypatch.setattr(template, "launch_flags",
                            lambda configuration: [template.TEMPLATE_FILE_FLAG, "copy.jinja"])
        configuration = types.SimpleNamespace(runtime=None, model=None)
        monkeypatch.setattr(runtime, "_with_runtime", lambda configuration, _runtime: configuration)
        monkeypatch.setattr(runtime, "_wangp_thread_flags", lambda configuration, placement: [])

        runtime._launch_flags(configuration, ctx.Placement(), None)

        assert seen == {"full_cache": True, "template_fixed": True}
