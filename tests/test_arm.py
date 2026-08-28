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


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestArming:
    def test_it_starts_the_language_model(self, pipeline, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: False)

        mc_arm.arm(reason="a test")

        assert pipeline.clients == 1

    def test_it_waits_for_the_image_preload(self, pipeline, monkeypatch):
        """An asynchronous warm-up is not a warm-up as far as the generation
        behind it is concerned."""
        import mc_memory

        joined: list = []
        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: True)
        monkeypatch.setattr(mc_memory, "join_preload",
                            lambda timeout=None: joined.append(True))

        mc_arm.arm(1024, 1024, reason="a test")

        assert joined == [True]

    def test_it_does_not_wait_for_a_preload_that_never_started(self, pipeline,
                                                               monkeypatch):
        import mc_memory

        joined: list = []
        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: False)
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
        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: False)
        pipeline._running = (Server(),)

        mc_arm.arm(reason="a test")

        assert pipeline.clients == 0

    def test_a_start_that_fails_leaves_the_pipeline_as_it_was(self, pipeline,
                                                              monkeypatch):
        """The generation behind it already knows how to load its own models;
        that is what it did before this existed."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: False)

        def explode(*args, **kwargs):
            raise RuntimeError("llama-server would not start")

        monkeypatch.setattr(pipeline, "client", explode)

        found = mc_arm.arm(reason="a test")  # must not raise

        assert not found.armed

    def test_two_warm_ups_do_not_both_pay_for_the_load(self, pipeline, monkeypatch):
        """A startup warm-up and a Generate pressed two seconds later."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "preload_async", lambda w=0, h=0: False)
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
