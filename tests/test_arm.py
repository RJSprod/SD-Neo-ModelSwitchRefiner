"""Arming: whether the next generation has to load anything, and loading it early.

The thing under test is an ordering, not a speed. Nothing here makes a model
faster; it decides when the model load is paid for -- while somebody watches a
progress bar, or before they press anything. So the tests are about *when*
loading happens and about never doing it twice, and about a readout that can be
called on every panel refresh without becoming the reason its own answer is yes.
"""

from __future__ import annotations

import threading

import pytest

import mc_arm

_GB = 1024**3


class Server:
    """A llama-server, as much of one as ``mc_arm`` looks at."""

    def __init__(self, layers=-1, on_gpu=True):
        import mc_llm_context as ctx

        self._placement = ctx.Placement(gpu_layers=layers, on_gpu=on_gpu)

    def placement(self):
        return self._placement


class Registry:
    def __init__(self, *running):
        self._running = running
        self.clients = 0

    def running(self, **_):
        return self._running

    def for_role(self, role="", configuration=None):
        return self

    def client(self, *args, **kwargs):
        self.clients += 1
        return object()


@pytest.fixture
def pipeline(host, monkeypatch):
    """A configured install, nothing loaded, and no real models anywhere."""
    import mc_llm_runtime
    import mc_memory

    monkeypatch.setattr(mc_memory, "stage_1_readiness",
                        lambda: ("cold", "Model Chain: Stage 1 is cold — 13.9 GB still to "
                                         "move from system RAM."))
    monkeypatch.setattr(mc_llm_runtime, "config",
                        lambda role="": type("C", (), {"configured": True})())
    registry = Registry()
    monkeypatch.setattr(mc_llm_runtime, "registry", registry)
    return registry


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #


class TestReadingTheState:
    def test_nothing_loaded_is_cold(self, pipeline):
        found = mc_arm.readiness()

        assert found.state == mc_arm.COLD
        assert not found.armed

    def test_a_running_server_and_a_warm_checkpoint_is_armed(self, pipeline, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Model Chain: Stage 1 is warm — 13.9 GB "
                                             "already in VRAM."))
        pipeline._running = (Server(),)

        found = mc_arm.readiness()

        assert found.armed
        assert found.cold == ()

    def test_the_worst_part_decides(self, pipeline, monkeypatch):
        """A resident checkpoint and no llama-server is warm in one place and
        cold in the other, and what matters is when the next Generate ends."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm — 13.9 GB already in VRAM."))

        found = mc_arm.readiness()

        assert found.state == mc_arm.COLD
        assert [part.name for part in found.cold] == ["Language model"]

    def test_a_degraded_server_is_not_ready_and_is_not_cold(self, pipeline, monkeypatch):
        """The first start of a session is placed conservatively, because the
        runtime reserve has not been calibrated. In the log this came from it
        put 37 of 65 layers on a card that turned out to hold all of them."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))
        pipeline._running = (Server(layers=37),)

        found = mc_arm.readiness()

        assert found.state == mc_arm.PARTIAL

    def test_a_processor_placement_is_not_called_degraded(self, pipeline, monkeypatch):
        """It has no layers on the card because it was never going to have any,
        which is a configuration rather than a shortfall."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))
        pipeline._running = (Server(layers=0, on_gpu=False),)

        assert mc_arm.readiness().armed

    def test_an_unconfigured_install_has_no_language_model_part(self, pipeline,
                                                                monkeypatch):
        """Nothing to arm is not something to report as cold for ever."""
        import mc_llm_runtime
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))
        monkeypatch.setattr(mc_llm_runtime, "config",
                            lambda role="": type("C", (), {"configured": False})())

        found = mc_arm.readiness()

        assert [part.name for part in found.parts] == ["Image model"]
        assert found.armed

    def test_reading_it_starts_nothing(self, pipeline):
        """A readout that loaded a model to find out whether a model was loaded
        would make its own answer true by asking the question."""
        mc_arm.readiness()

        assert pipeline.clients == 0

    def test_a_module_that_raises_does_not_break_the_readout(self, pipeline, monkeypatch):
        import mc_memory

        def explode():
            raise RuntimeError("no")

        monkeypatch.setattr(mc_memory, "stage_1_readiness", explode)

        found = mc_arm.readiness()

        assert [part.name for part in found.parts] == ["Language model"]

    def test_it_says_which_parts_are_cold(self, pipeline):
        said = mc_arm.readiness().describe()

        assert "language model" in said
        assert "image model" in said

    def test_it_can_also_say_why(self, pipeline):
        """"warm-up finished in 0.0s — image model is cold" answers nothing.

        The reason was measured a line earlier; every part carries it.
        """
        said = mc_arm.readiness().explain()

        assert "13.9 GB still to move from system RAM" in said

    def test_an_armed_pipeline_has_nothing_to_explain(self, pipeline, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))
        pipeline._running = (Server(),)

        found = mc_arm.readiness()

        assert found.explain() == found.describe()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestArming:
    def test_it_starts_the_language_model(self, pipeline, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)

        mc_arm.arm(reason="a test")

        assert pipeline.clients == 1

    def test_it_waits_for_the_image_preload(self, pipeline, monkeypatch):
        """An asynchronous warm-up is not a warm-up as far as the generation
        behind it is concerned."""
        import mc_memory

        joined: list = []
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: True)
        monkeypatch.setattr(mc_memory, "join_preload",
                            lambda timeout=None: joined.append(True))

        mc_arm.arm(1024, 1024, reason="a test")

        assert joined == [True]

    def test_it_does_not_wait_for_a_preload_that_never_started(self, pipeline,
                                                               monkeypatch):
        import mc_memory

        joined: list = []
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)
        monkeypatch.setattr(mc_memory, "join_preload",
                            lambda timeout=None: joined.append(True))

        mc_arm.arm(reason="a test")

        assert joined == []

    def test_an_armed_pipeline_is_left_alone(self, pipeline, monkeypatch):
        """After the first run of a session this is every call, so it has to
        cost a measurement and nothing else."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)
        pipeline._running = (Server(),)

        mc_arm.arm(reason="a test")

        assert pipeline.clients == 0

    def test_a_start_that_fails_leaves_the_pipeline_as_it_was(self, pipeline,
                                                              monkeypatch):
        """The generation behind it already knows how to load its own models;
        that is what it did before this existed."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)

        def explode(*args, **kwargs):
            raise RuntimeError("llama-server would not start")

        monkeypatch.setattr(pipeline, "client", explode)

        found = mc_arm.arm(reason="a test")  # must not raise

        assert not found.armed

    def test_two_warm_ups_do_not_both_pay_for_the_load(self, pipeline, monkeypatch):
        """A startup warm-up and a Generate pressed two seconds later."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)
        started = threading.Event()
        release = threading.Event()

        def slow(*args, **kwargs):
            pipeline.clients += 1
            started.set()
            release.wait(5)
            return object()

        monkeypatch.setattr(pipeline, "client", slow)
        first = threading.Thread(target=mc_arm.arm, kwargs={"reason": "startup"})
        first.start()
        assert started.wait(5)

        mc_arm.arm(reason="a generation")
        release.set()
        first.join(timeout=5)

        assert pipeline.clients == 1


class TestTheImageModelComesFirst:
    """Generate is what somebody is sitting in front of.

    From a user's log, a warm-up that ran llama-server first:

        warming up for this generation — cold — image model is cold; language
            model is cold
        llama-server ready — all layers on the GPU, 8,192 token context
        warm-up finished in 20.3s — cold — image model is cold

    Twenty seconds of somebody's wait, spent entirely on the half they were not
    waiting for, and the half they were still in system RAM at the end of it.

    Nothing is lost on the language side by going second: its VRAM allowance is
    the plan's remainder, and ``mc_plan.usable_vram_bytes`` adds the image
    family's own residency back before dividing, so llama-server is placed at
    the same size either way.
    """

    @pytest.fixture
    def order(self, pipeline, monkeypatch):
        import mc_memory

        seen: list = []
        monkeypatch.setattr(
            mc_memory, "preload_async",
            lambda w=0, h=0, **kwargs: seen.append(("image", kwargs)) or False)
        monkeypatch.setattr(pipeline, "client",
                            lambda *a, **k: seen.append(("llm", {})) or object())
        return seen

    def test_the_image_model_is_warmed_before_the_language_model(self, order):
        mc_arm.arm(1024, 1024, reason="a test")

        assert [step for step, _ in order] == ["image", "llm"]

    def test_it_asks_for_the_load_the_background_pass_will_not_do(self, order):
        """"Never a cold run" has to include the first run of a session."""
        mc_arm.arm(1024, 1024, reason="a test")

        _, kwargs = order[0]
        assert kwargs["allow_disk_load"] is True

    def test_it_does_not_need_the_preload_setting_turned_on_as_well(self, order):
        """Turning the warm-up on is already an answer about warmth."""
        mc_arm.arm(1024, 1024, reason="a test")

        _, kwargs = order[0]
        assert kwargs["force"] is True

    def test_a_retired_preload_is_said_out_loud(self, pipeline, monkeypatch, caplog):
        """Otherwise an image model that is never warmed is an unexplained 0.0s."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)
        monkeypatch.setattr(mc_memory, "preload_disabled_reason",
                            lambda: "it failed 2 times in a row")

        with caplog.at_level("INFO", logger="model_chain"):
            mc_arm.arm(1024, 1024, reason="a test")

        assert any("failed 2 times in a row" in record.getMessage()
                   for record in caplog.records)


class TestFreeingTheLlmForEveryImage:
    """On that setting, warming llama-server is warming it to be stopped.

    ``request_vram(FAMILY_IMAGE, ...)`` sweeps the image card before anything
    else in ``before_process`` happens, and it sits a handful of lines below the
    warm-up. So the warm-up was starting a language model for the next statement
    to stop -- twenty of the twenty-and-a-bit seconds a Generate click waited
    for in a user's log, on a process that never reached a sampling step, every
    press.
    """

    @pytest.fixture
    def exclusive(self, pipeline, monkeypatch):
        import mc_broker
        import mc_llm_runtime
        import mc_memory

        monkeypatch.setattr(mc_broker, "mode", lambda: mc_broker.MODE_EXCLUSIVE)
        monkeypatch.setattr(mc_llm_runtime, "card_of", lambda configuration: 0)
        monkeypatch.setattr(mc_llm_runtime, "shares_the_image_card",
                            lambda card, configuration=None: True)
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)
        return pipeline

    def test_the_language_model_is_not_started(self, exclusive):
        mc_arm.arm(1024, 1024, reason="a test")

        assert exclusive.clients == 0

    def test_the_image_model_is_still_warmed(self, exclusive, monkeypatch):
        """It is the half a generation is actually waiting on."""
        import mc_memory

        warmed: list = []
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: warmed.append((w, h)) or False)

        mc_arm.arm(1024, 1024, reason="a test")

        assert warmed == [(1024, 1024)]

    def test_a_stopped_server_is_not_reported_as_a_cold_half_of_the_pipeline(
            self, exclusive, monkeypatch):
        """It is the state the next generation is asking for, not a shortfall.

        Reported cold, it would also make readiness() unable to ever say armed,
        so the fast path every warm-up after the first one takes would never
        fire again.
        """
        import mc_memory

        monkeypatch.setattr(mc_memory, "stage_1_readiness",
                            lambda: ("warm", "Stage 1 is warm."))

        found = mc_arm.readiness()

        assert [part.name for part in found.parts] == ["Image model"]
        assert found.armed

    def test_it_says_why_it_did_not_start_one(self, exclusive, caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            mc_arm.arm(1024, 1024, reason="a test")

        assert any("free the LLM for every image" in record.getMessage()
                   for record in caplog.records)

    def test_a_role_on_another_card_is_warmed_as_it_always_was(self, exclusive,
                                                               monkeypatch):
        """The sweep is scoped to the image card, so this is scoped to it too."""
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "shares_the_image_card",
                            lambda card, configuration=None: False)

        mc_arm.arm(1024, 1024, reason="a test")

        assert exclusive.clients == 1

    def test_keeping_the_llm_loaded_still_warms_it(self, pipeline, monkeypatch):
        import mc_broker
        import mc_memory

        monkeypatch.setattr(mc_broker, "mode", lambda: mc_broker.MODE_HYBRID)
        monkeypatch.setattr(mc_memory, "preload_async",
                            lambda w=0, h=0, **kwargs: False)

        mc_arm.arm(1024, 1024, reason="a test")

        assert pipeline.clients == 1


class TestTheSetting:
    def test_off_is_the_default(self, host):
        assert mc_arm.mode() == mc_arm.WARM_OFF

    def test_startup_arming_only_happens_on_the_startup_setting(self, pipeline,
                                                                monkeypatch, host):
        from modules import shared

        armed: list = []
        monkeypatch.setattr(mc_arm, "arm_later",
                            lambda *a, **k: armed.append(k.get("reason", "")))

        shared.opts.model_chain_warm_up = mc_arm.WARM_OFF
        mc_arm.on_app_started()
        shared.opts.model_chain_warm_up = mc_arm.WARM_BEFORE
        mc_arm.on_app_started()

        assert armed == []

        shared.opts.model_chain_warm_up = mc_arm.WARM_STARTUP
        mc_arm.on_app_started()

        assert len(armed) == 1

    def test_a_label_resolves_to_its_value(self, host):
        from modules import shared

        shared.opts.model_chain_warm_up = mc_arm.WARM_MODES[1][1]

        assert mc_arm.mode() == mc_arm.WARM_BEFORE
