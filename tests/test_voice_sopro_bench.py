"""The Sopro validation sweep: the fit, the refusals, and the promise it keeps.

The promise is the one that matters most and is the easiest to break by
accident: I-12 allows a measurement and forbids a tuner, and the only thing
keeping this on the right side of that line is that it *acts on nothing*. A
sweep that quietly left the fastest precision selected would be an auto-tuner
with a progress bar. So there is a test that runs a whole sweep and asserts the
settings file is byte-identical afterwards.

The subprocess is stubbed throughout. What is being checked is the arithmetic,
the refusals and the environment handed to each configuration -- running the
real closure needs Windows, Torch and a cloned voice, and is what the button
does on the user's machine.
"""

from __future__ import annotations

import json

import pytest

import mc_voice_sopro_bench as bench


@pytest.fixture(autouse=True)
def _quiet_state():
    """Each test starts from an idle module. The state is deliberately global --
    one machine, one sweep -- so it has to be reset between tests."""
    with bench._lock:
        bench._state.update({"running": False, "done": False, "step": 0, "total": 0,
                             "message": "", "rows": [], "error": "", "best": None})
    yield


class TestTheFit:
    """Two lengths, so the intercept and the slope come apart. One length would
    give a real-time factor that is their sum and hide which one is the
    problem."""

    def test_it_recovers_a_line_it_was_given(self):
        points = [(2000, 300 + 0.8 * 2000), (12000, 300 + 0.8 * 12000)]
        fixed, rate, quality = bench._fit(points)
        assert fixed == pytest.approx(300, abs=1)
        assert rate == pytest.approx(0.8, abs=0.001)
        assert quality == pytest.approx(1.0)

    def test_one_slow_run_is_a_residual_rather_than_the_answer(self):
        """The reason this is a regression and not a pair of divisions.

        The comparison is against what the obvious implementation would have
        done: divide compute by audio for a run and call that the rate. A
        machine that hiccups once — a background scan, a thermal dip — makes
        that single-run reading wrong by an order of magnitude more than it
        moves the fit.
        """
        truth = 0.8
        clean = [(2000, 1900), (2000, 1900), (12000, 9900), (12000, 9900)]
        noisy = list(clean)
        noisy[0] = (2000, 3400)          # one run took nearly twice as long

        _fixed, fitted, _quality = bench._fit(noisy)
        naive = noisy[0][1] / noisy[0][0]

        assert abs(naive - truth) > 0.8, "the outlier was not severe enough to test with"
        assert abs(fitted - truth) < 0.1, fitted
        assert abs(fitted - truth) * 10 < abs(naive - truth), (fitted, naive)

    def test_the_fit_reports_how_well_it_fitted(self):
        """R² is on the table because a rate from points that do not lie on a
        line is a number with no claim behind it."""
        _fixed, _rate, clean = bench._fit([(2000, 2000), (12000, 10000)])
        _fixed2, _rate2, messy = bench._fit(
            [(2000, 2000), (2000, 9000), (12000, 10000), (12000, 3000)])
        assert clean > 0.99
        assert messy < 0.5, messy

    def test_one_length_is_reported_as_a_rate_and_no_fit(self):
        """No leverage to separate the intercept, so it does not invent one."""
        fixed, rate, quality = bench._fit([(4000, 3400), (4000, 3400)])
        assert fixed == 0.0
        assert rate == pytest.approx(0.85)
        assert quality is None

    def test_a_single_run_is_not_a_line(self):
        assert bench._fit([(4000, 3400)]) == (None, None, None)


class TestReadingTheAnswer:
    """The number a person acts on is the break-even Speed, because Sopro's
    Speed is a post-synthesis time compression and therefore multiplies the
    real-time factor exactly."""

    def test_the_quoted_rtf_includes_the_per_segment_cost(self):
        """Quoting the marginal rate alone would flatter every configuration by
        however long the prompt state takes."""
        fixed, rate = 420.0, 0.798
        rtf = rate + fixed / (bench.QUOTED_SECONDS * 1000.0)
        assert rtf == pytest.approx(0.858, abs=0.001)
        assert 1.0 / rtf == pytest.approx(1.166, abs=0.002)


class TestWhatItRefuses:
    def test_it_will_not_measure_with_no_voice(self, monkeypatch):
        """A benchmark that skipped the voice would not be measuring the path
        Conversation uses -- the reconstruction is part of the cost."""
        found = _sweep_against(monkeypatch, voices={})
        assert "no Sopro voice" in found["error"], found

    def test_it_will_not_measure_while_a_reply_is_being_spoken(self, monkeypatch):
        found = _sweep_against(monkeypatch, state="speaking")
        assert "busy speaking" in found["error"], found

    def test_it_will_not_measure_without_the_isolated_runtime(self, monkeypatch):
        found = _sweep_against(monkeypatch, interpreter=None)
        assert "isolated Sopro runtime" in found["error"], found

    def test_two_sweeps_at_once_are_refused(self):
        with bench._lock:
            bench._state["running"] = True
        with pytest.raises(bench.BenchError):
            bench.run()


class TestItChangesNothing:
    """I-12 allows a measurement and forbids a tuner. This is the line."""

    def test_a_whole_sweep_leaves_every_setting_where_it_was(self, monkeypatch, tmp_path):
        import mc_voice_paths as paths

        settings = tmp_path / "sopro-settings.json"
        settings.write_text(json.dumps({"precision": "full", "steps": 2}),
                            encoding="utf-8")
        monkeypatch.setattr(paths, "sopro_settings_path", lambda: settings)
        before = settings.read_bytes()
        # The winner is deliberately *not* what is stored. A sweep whose fastest
        # configuration happened to match the current setting would let a tuner
        # through unnoticed — writing "full" over "full" changes no bytes — and
        # this test exists to catch exactly that write.
        found = _sweep_against(monkeypatch, precisions=("full", "int8"),
                               threads=(2, 8),
                               rate_by_precision={"full": 0.9, "int8": 0.4})
        assert not found["error"], found
        assert len(found["rows"]) == 4, found
        assert found["best"]["precision"] == "int8", \
            "the fixture did not produce a winner that differs from the setting"
        assert settings.read_bytes() == before, \
            "the sweep wrote a setting; that is a tuner, not a measurement"

    def test_the_precision_is_an_argument_not_a_setting(self, monkeypatch):
        """Each configuration is told which precision to use. Nothing selects
        one, which is why an interrupted sweep leaves nothing behind."""
        found = _sweep_against(monkeypatch, precisions=("full", "int8"), threads=(4,))
        asked = [json.loads(request)["config"]["precision"]
                 for request in found["requests"]]
        assert asked == ["full", "int8"], asked


class TestTheEnvironmentEachRunGets:
    def test_openmp_tracks_the_thread_count_being_measured(self, monkeypatch):
        """The subtle one. ``worker_environment`` pins OMP_NUM_THREADS to the
        released count, and OpenMP sizes its pool before any of our code runs --
        so a row measured at eight with OMP capped at four belongs to neither
        number."""
        found = _sweep_against(monkeypatch, threads=(2, 8), precisions=("full",))
        caps = [(env["MC_SOPRO_INTRAOP_THREADS"], env["OMP_NUM_THREADS"],
                 env["MKL_NUM_THREADS"], env["OPENBLAS_NUM_THREADS"])
                for env in found["environs"]]
        assert caps == [("2", "2", "2", "2"), ("8", "8", "8", "8")], caps

    def test_each_configuration_is_a_process_of_its_own(self, monkeypatch):
        """OpenMP sizes its pool at the first parallel region and
        ``set_num_interop_threads`` refuses after one, so a reused process would
        measure the first thread count several times under different labels."""
        found = _sweep_against(monkeypatch, threads=(2, 4, 8), precisions=("full",))
        assert len(found["requests"]) == 3, "configurations shared a process"

    def test_the_resident_worker_is_stopped_first(self, monkeypatch):
        """Two Sopro processes on one CPU measure each other."""
        found = _sweep_against(monkeypatch)
        assert found["stopped"], "the resident worker was left running"


class TestWhatTheRowsSay:
    def test_torchs_own_thread_count_is_reported_not_the_one_asked_for(self, monkeypatch):
        """A machine with fewer cores than the sweep asked about quietly gives
        you fewer, and a table that reported the request would invent a data
        point that was never measured."""
        found = _sweep_against(monkeypatch, threads=(64,), precisions=("full",),
                               reported_threads=6)
        assert found["rows"][0]["threads"] == 6
        assert found["rows"][0]["asked_threads"] == 64

    def test_a_configuration_that_failed_is_a_row_rather_than_the_end(self, monkeypatch):
        """One precision this Torch build cannot do must not cost the table the
        other one."""
        found = _sweep_against(monkeypatch, precisions=("full", "int8"), threads=(4,),
                               fail_precision="int8")
        assert len(found["rows"]) == 2, found["rows"]
        assert found["rows"][0].get("rtf") is not None
        assert "error" in found["rows"][1]
        assert found["best"]["precision"] == "full"

    def test_the_fastest_row_is_the_one_with_the_lowest_rtf(self, monkeypatch):
        found = _sweep_against(monkeypatch, threads=(2, 8), precisions=("full",),
                               rate_by_threads={2: 1.4, 8: 0.6})
        assert found["best"]["threads"] == 8, found["rows"]
        assert found["best"]["break_even_speed"] > 1.0


class TestTheProgressAPagePolls:
    def test_state_is_readable_before_anything_has_run(self):
        found = bench.state()
        assert found["running"] is False and found["rows"] == []

    def test_state_never_hands_out_the_module_s_own_list(self):
        """The panel polls this while the sweep is appending to it. Handing over
        the live list would be a row half-written into a JSON response."""
        found = bench.state()
        found["rows"].append({"tampered": True})
        assert bench.state()["rows"] == []


# --------------------------------------------------------------------------- #
# The stub
# --------------------------------------------------------------------------- #


def _sweep_against(monkeypatch, *, voices=None, state="idle", interpreter="python",
                   threads=(4,), precisions=("full",), rate=0.8, fixed=400,
                   reported_threads=None, fail_precision="", rate_by_threads=None,
                   rate_by_precision=None):
    """Run a whole sweep with the subprocess replaced by arithmetic.

    Returns the module's state plus what the stub saw, so a test can assert on
    the environment and the requests as well as on the table.
    """
    import mc_voice_paths as paths
    import mc_voice_sopro as sopro
    import mc_voice_sopro_runtime as runtime

    seen = {"requests": [], "environs": [], "stopped": False}

    class Status:
        ready = True
        fingerprint = "abc123"

    monkeypatch.setattr(sopro, "status", lambda: Status())
    monkeypatch.setattr(sopro, "runtime_python", lambda: interpreter)
    monkeypatch.setattr(sopro, "worker_config", lambda: {
        "model_root": "/models/sopro", "model_id": "sopro-v2", "precision": "full",
        "steps": 2, "chunk_frames": 64, "fingerprint": "abc123",
        "voices": {"sopro:clone:one": {"root": "/voices/one", "fingerprint": "abc123"}}
        if voices is None else voices})
    monkeypatch.setattr(sopro, "worker_environment", lambda: {
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    monkeypatch.setattr(runtime, "engine", lambda: {"state": state})
    monkeypatch.setattr(runtime, "stop",
                        lambda reason="": seen.__setitem__("stopped", True))
    monkeypatch.setattr(paths, "sopro_worker_script", lambda: "worker.py")
    monkeypatch.setattr(paths, "extension_root", lambda: ".")

    class Finished:
        returncode = 0
        stderr = b""

        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, input=None, env=None, **_kwargs):  # noqa: A002
        request = input.decode("utf-8")
        seen["requests"].append(request)
        seen["environs"].append(dict(env or {}))
        asked = json.loads(request)
        want = asked["config"]["precision"]
        if want == fail_precision:
            return Finished(json.dumps(
                {"ok": False, "error": "this build has no int8 kernels"}).encode())
        count = int(env["MC_SOPRO_INTRAOP_THREADS"])
        slope = (rate_by_precision or {}).get(
            want, (rate_by_threads or {}).get(count, rate))
        runs = [{"audio_ms": length, "compute_ms": int(fixed + slope * length),
                 "first_audio_ms": 900, "chunks": 4, "chars": 10, "run": 0}
                for length in (2000, 12000)]
        return Finished(json.dumps({
            "ok": True, "runs": runs, "precision": want, "load_seconds": 1.2,
            "intraop_threads": reported_threads or count, "interop_threads": 1,
        }).encode())

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    bench.run(threads=threads, precisions=precisions, repeats=1)
    return dict(bench.state(), **seen)
