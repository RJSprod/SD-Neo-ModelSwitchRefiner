"""A batch's prompts of different lengths padded to the longest, as Forge did before 2026-10-30.

Forge Neo's commit 49bc706 ("batching") made ``prompt_parser.stack_conds``
repeat every prompt of a batch up to the least common multiple of their
lengths. With Krea 2's unpadded conditioning (12 x 2560 float32 values a token)
two wildcard picks of 250 and 254 tokens became 31,750 tokens each, and every
wildcard batch ran the user's 24 GB card out of memory. ``mc_prompt_padding``
wraps the host's function and answers such a call with the old padding.

The host's code below is Forge Neo's own, copied as it stands at 49bc706
(``FORGE_TODAY``, ``RECONSTRUCT``) and as it stood before it
(``FORGE_BEFORE``), run against a numpy stand-in for torch: this container has
no torch, and the stand-in raises where torch raises (``stack`` of unequal
shapes) and tiles where torch's ``repeat`` tiles.
"""

from __future__ import annotations

import logging
import math
import sys
import types

import numpy as np
import pytest

import mc_prompt_padding

FORGE_TODAY = '''
def stack_conds(tensors: list[torch.Tensor]) -> torch.Tensor:
    try:
        return torch.stack(tensors)
    except RuntimeError:
        import math

        token_count = math.lcm(*(x.shape[0] for x in tensors))
        return torch.stack([x.repeat(token_count // x.shape[0], *([1] * (x.dim() - 1))) for x in tensors])
'''

FORGE_BEFORE = '''
def stack_conds(tensors: list[torch.Tensor]) -> torch.Tensor:
    try:
        return torch.stack(tensors)
    except Exception:
        # handle prompts with different lengths (e.g. LLM)

        token_count: int = max([x.shape[0] for x in tensors])
        new_tensors: list[torch.Tensor] = []

        for x in tensors:
            if x.shape[0] == token_count:
                new_tensors.append(x)
            else:
                diff = token_count - x.shape[0]
                last_vector = x[-1:]
                repeats = [diff] + [1] * (x.dim() - 1)
                last_vector_repeated = last_vector.repeat(repeats)
                x_padded = torch.cat([x, last_vector_repeated], dim=0)
                new_tensors.append(x_padded)

        return torch.stack(new_tensors)
'''

RECONSTRUCT = '''
def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step: int):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                if current_step <= entry.end_at_step:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked)
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked
'''


class Tensor:
    """Enough of a torch tensor for the code above, over a numpy array."""

    device = "cpu"

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return tuple(self.array.shape)

    @property
    def dtype(self):
        return self.array.dtype

    def dim(self):
        return self.array.ndim

    def element_size(self):
        return self.array.itemsize

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = tuple(sizes[0])
        return Tensor(np.tile(self.array, sizes))

    def to(self, **_kwargs):
        return self


def _stack(tensors):
    if len({t.shape for t in tensors}) > 1:
        raise RuntimeError("stack expects each tensor to be equal size")
    return Tensor(np.stack([t.array for t in tensors]))


def _cat(tensors, dim=0):
    return Tensor(np.concatenate([t.array for t in tensors], axis=dim))


@pytest.fixture
def torch(monkeypatch):
    module = types.ModuleType("torch")
    module.Tensor = Tensor
    module.stack = _stack
    module.cat = _cat
    module.zeros = lambda *shape: Tensor(np.zeros(shape, dtype=np.float32))
    module.ones = lambda *shape: Tensor(np.ones(shape, dtype=np.float32))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def forge_module(torch, *sources):
    """A ``modules.prompt_parser`` built from Forge's own source."""
    module = types.ModuleType("modules.prompt_parser")
    module.torch = torch
    module.MulticondLearnedConditioning = object
    module.DictWithShape = dict
    for source in sources:
        exec(source, module.__dict__)  # noqa: S102 -- the host's code, verbatim
    return module


@pytest.fixture
def today(torch, monkeypatch, host):
    """Forge Neo at 49bc706, as ``modules.prompt_parser``."""
    module = forge_module(torch, FORGE_TODAY, RECONSTRUCT)
    _as_the_hosts(monkeypatch, module)
    mc_prompt_padding.forget()
    yield module
    mc_prompt_padding.forget()


@pytest.fixture
def before(torch, monkeypatch, host):
    """Forge Neo before 49bc706."""
    module = forge_module(torch, FORGE_BEFORE, RECONSTRUCT)
    _as_the_hosts(monkeypatch, module)
    mc_prompt_padding.forget()
    yield module
    mc_prompt_padding.forget()


def _as_the_hosts(monkeypatch, module):
    import modules

    monkeypatch.setitem(sys.modules, "modules.prompt_parser", module)
    monkeypatch.setattr(modules, "prompt_parser", module, raising=False)


def prompt(length, width=(3, 4), seed=0):
    """One prompt's conditioning: ``length`` tokens, each a distinct block of numbers."""
    rows = np.arange(length * math.prod(width), dtype=np.float32).reshape(length, *width)
    return Tensor(rows + seed * 1000)


def batch(*conds):
    """A ``MulticondLearnedConditioning`` with one plain prompt per image."""
    def one(cond):
        schedule = types.SimpleNamespace(end_at_step=99, cond=cond)
        return types.SimpleNamespace(schedules=[schedule], weight=1.0)

    return types.SimpleNamespace(batch=[[one(cond)] for cond in conds])


def switched_off(host):
    host.shared.opts.set(mc_prompt_padding.OPT_ENABLED, False)


class TestTheHostsOwnCode:
    """What the wrapper is up against, shown on the host's own source."""

    def test_today_a_wildcard_batch_is_repeated_to_the_least_common_multiple(self, today):
        _, stacked = today.reconstruct_multicond_batch(batch(prompt(250), prompt(254, seed=1)), 0)
        assert stacked.shape == (2, 31750, 3, 4)

    def test_before_it_was_padded_to_the_longest(self, before):
        _, stacked = before.reconstruct_multicond_batch(batch(prompt(250), prompt(254, seed=1)), 0)
        assert stacked.shape == (2, 254, 3, 4)

    def test_at_krea_twos_width_that_is_gigabytes_for_two_prompts(self):
        token = 12 * 2560 * 4  # Krea 2: 12 hidden layers of 2560, float32
        today = 2 * math.lcm(250, 254) * token
        before = 2 * 254 * token
        assert today / 1024**3 > 7.0
        assert before / 1024**2 < 60


class TestThePadding:
    def test_a_wildcard_batch_is_padded_to_the_longest_through_forges_own_caller(self, today):
        assert mc_prompt_padding.install() is True
        short, long = prompt(250), prompt(254, seed=1)

        _, stacked = today.reconstruct_multicond_batch(batch(short, long), 0)

        assert stacked.shape == (2, 254, 3, 4)
        assert np.array_equal(stacked.array[0, :250], short.array)
        assert np.array_equal(stacked.array[0, 250:], np.repeat(short.array[-1:], 4, axis=0))
        assert np.array_equal(stacked.array[1], long.array)

    @pytest.mark.parametrize("lengths", [(250, 254), (7, 3), (5, 9, 5, 2), (1, 300)])
    def test_the_padding_is_the_old_code_exactly(self, today, torch, lengths):
        old = forge_module(torch, FORGE_BEFORE)
        conds = [prompt(n, seed=i) for i, n in enumerate(lengths)]
        mc_prompt_padding.install()

        ours = today.stack_conds(conds)

        assert np.array_equal(ours.array, old.stack_conds(conds).array)

    def test_a_dictionary_condition_is_padded_key_by_key(self, today, torch):
        mc_prompt_padding.install()
        conds = [{"crossattn": prompt(6), "vector": prompt(1, width=(8,))},
                 {"crossattn": prompt(9, seed=1), "vector": prompt(1, width=(8,), seed=1)}]

        _, stacked = today.reconstruct_multicond_batch(batch(*conds), 0)

        assert stacked["crossattn"].shape == (2, 9, 3, 4)
        assert stacked["vector"].shape == (2, 1, 8)

    def test_one_length_is_the_hosts_own_stack_and_two_never_reach_it(self, today):
        calls = []
        host_stack = today.stack_conds

        def spy(tensors):
            calls.append([t.shape[0] for t in tensors])
            return host_stack(tensors)

        module = types.ModuleType("spy")
        module.stack_conds = spy
        mc_prompt_padding.install(module)
        calls.clear()  # the probe's call

        stacked = module.stack_conds([prompt(250), prompt(250, seed=1)])
        assert calls == [[250, 250]]
        assert stacked.shape == (2, 250, 3, 4)

        module.stack_conds([prompt(250), prompt(254, seed=1)])
        assert calls == [[250, 250]]

    def test_shapes_that_differ_past_their_length_are_left_to_the_host(self, today):
        calls = []
        host_stack = today.stack_conds

        def spy(tensors):
            calls.append([t.shape for t in tensors])
            return host_stack(tensors)

        module = types.ModuleType("spy")
        module.stack_conds = spy
        mc_prompt_padding.install(module)
        calls.clear()  # the probe's call

        with pytest.raises(RuntimeError):
            module.stack_conds([prompt(5, width=(3,)), prompt(6, width=(4,))])
        assert calls == [[(5, 3), (6, 4)]]

    def test_the_switch_off_leaves_forges_repeat(self, today, host):
        mc_prompt_padding.install()
        switched_off(host)

        _, stacked = today.reconstruct_multicond_batch(batch(prompt(6), prompt(9, seed=1)), 0)

        assert stacked.shape == (2, 18, 3, 4)

    def test_the_switch_is_read_on_every_call(self, today, host):
        mc_prompt_padding.install()
        switched_off(host)
        assert today.stack_conds([prompt(6), prompt(9)]).shape[1] == 18
        host.shared.opts.set(mc_prompt_padding.OPT_ENABLED, True)
        assert today.stack_conds([prompt(6), prompt(9)]).shape[1] == 9

    def test_the_setting_is_on_until_turned_off(self, host):
        import model_chain  # noqa: F401  -- registers the settings

        option = host.shared.options_templates[mc_prompt_padding.OPT_ENABLED]
        assert option.default is True
        assert mc_prompt_padding.enabled() is True


class TestTheWrapper:
    def test_it_goes_in_once(self, today):
        original = today.stack_conds
        assert mc_prompt_padding.install() is True
        wrapper = today.stack_conds
        assert mc_prompt_padding.install() is True

        assert today.stack_conds is wrapper
        assert wrapper.__wrapped__ is original

    def test_unloading_puts_the_hosts_function_back(self, today):
        original = today.stack_conds
        mc_prompt_padding.install()

        assert mc_prompt_padding.uninstall() is True
        assert today.stack_conds is original
        assert mc_prompt_padding.uninstall() is False

    def test_a_forge_without_stack_conds_is_left_alone(self, torch, monkeypatch, host):
        module = types.ModuleType("modules.prompt_parser")
        _as_the_hosts(monkeypatch, module)

        assert mc_prompt_padding.install() is False
        assert not hasattr(module, "stack_conds")

    def test_a_host_this_was_not_checked_against_keeps_its_own_function(self, torch, monkeypatch,
                                                                        host, caplog):
        """A host whose answer to the probe is neither the repeat nor the padding:
        here, one that pads every prompt to a fixed 512 tokens."""
        module = forge_module(torch, '''
def stack_conds(tensors):
    return torch.stack([torch.cat([x, x[-1:].repeat(512 - x.shape[0], *([1] * (x.dim() - 1)))])
                        for x in tensors])
''')
        _as_the_hosts(monkeypatch, module)
        own = module.stack_conds
        mc_prompt_padding.forget()

        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_prompt_padding.on_app_started()

        assert module.stack_conds is own
        assert module.stack_conds([prompt(6), prompt(9)]).shape == (2, 512, 3, 4)
        [line] = [r for r in caplog.records if "padding of a batch's prompts is off" in r.getMessage()]
        assert line.levelno == logging.WARNING
        mc_prompt_padding.forget()

    def test_a_host_without_prompt_parser_is_left_alone(self, host, monkeypatch):
        import modules

        monkeypatch.delitem(sys.modules, "modules.prompt_parser", raising=False)
        monkeypatch.delattr(modules, "prompt_parser", raising=False)

        assert mc_prompt_padding.install() is False

    def test_the_extensions_unload_puts_it_back(self, today, host, monkeypatch):
        import model_chain

        monkeypatch.setattr(model_chain.mc_memory, "release_all", lambda: None)
        monkeypatch.setattr(model_chain.mc_llm_runtime, "shutdown", lambda: None)
        monkeypatch.setattr(model_chain.mc_broker, "clear", lambda: None)
        original = today.stack_conds
        mc_prompt_padding.install()

        model_chain._on_script_unloaded()

        assert today.stack_conds is original

    def test_it_is_installed_at_app_start(self):
        import model_chain  # noqa: F401  -- imported for its registrations
        from modules import script_callbacks

        assert mc_prompt_padding.on_app_started in script_callbacks.registered["app_started"]


class TestTheProbe:
    def test_todays_forge_repeats(self, today):
        assert mc_prompt_padding.probe(today.stack_conds) == mc_prompt_padding.REPEATS

    def test_the_old_forge_pads(self, before):
        assert mc_prompt_padding.probe(before.stack_conds) == mc_prompt_padding.PADS

    def test_a_host_that_fails_is_unknown(self, torch):
        def broken(_tensors):
            raise RuntimeError("no")

        assert mc_prompt_padding.probe(broken) == mc_prompt_padding.UNKNOWN

    def test_without_torch_it_is_unknown(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert mc_prompt_padding.probe(lambda tensors: tensors) == mc_prompt_padding.UNKNOWN


class TestTheLog:
    def test_the_start_says_forge_repeats_and_the_batch_is_padded(self, today, caplog):
        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_prompt_padding.on_app_started()

        [line] = [m for m in caplog.messages if "common multiple" in m]
        assert "padded to the longest instead" in line
        assert mc_prompt_padding.FORGE_CHANGE in line

    def test_the_start_says_the_switch_is_off(self, today, host, caplog):
        switched_off(host)
        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_prompt_padding.on_app_started()

        [line] = [m for m in caplog.messages if "common multiple" in m]
        assert "switch is off" in line

    def test_an_old_forge_says_nothing_at_the_start(self, before, caplog):
        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_prompt_padding.on_app_started()

        assert not [m for m in caplog.messages if "common multiple" in m]
        assert getattr(before.stack_conds, "__wrapped__", None) is not None

    def test_a_padded_batch_is_said_once_for_each_set_of_lengths(self, today, caplog):
        mc_prompt_padding.install()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            today.stack_conds([prompt(6), prompt(9)])
            today.stack_conds([prompt(9), prompt(6)])
            today.stack_conds([prompt(6), prompt(10)])

        lines = [m for m in caplog.messages if "were padded" in m]
        assert len(lines) == 2
        assert "6 and 9 tokens long and were padded to 9" in lines[0]
        assert "repeated each to 18 tokens" in lines[0]
        assert "6 and 10 tokens long and were padded to 10" in lines[1]

    def test_the_line_gives_what_forge_would_have_built_at_krea_twos_width(self, today, caplog):
        mc_prompt_padding.install()
        # The shape of Krea 2's conditioning without its 60 MB: _say reads
        # shapes and the element size, never the numbers.
        conds = [types.SimpleNamespace(shape=(250, 12, 2560), element_size=lambda: 4),
                 types.SimpleNamespace(shape=(254, 12, 2560), element_size=lambda: 4)]
        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_prompt_padding._say(conds)

        [line] = [m for m in caplog.messages if "were padded" in m]
        assert "250 and 254 tokens long and were padded to 254" in line
        assert "31,750 tokens, 7.3 GB for the batch" in line

    def test_an_old_forge_is_not_said_to_repeat(self, before, caplog):
        mc_prompt_padding.install()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            before.stack_conds([prompt(6), prompt(9)])

        [line] = [m for m in caplog.messages if "were padded" in m]
        assert "repeated" not in line

    def test_a_batch_of_one_length_says_nothing(self, today, caplog):
        mc_prompt_padding.install()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            today.stack_conds([prompt(6), prompt(6)])

        assert not [m for m in caplog.messages if "were padded" in m]
