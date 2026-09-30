"""``tools/affected_tests.py`` and ``tools/run_tests.py``: which tests a change reaches.

Each case is a small repository written into ``tmp_path`` with the links this
one actually has -- a test that imports a helper from another test, a page
script read by path, a worker started by a path built from a constant, a
fixture that imports a module, a scan of every ``mc_*.py`` -- and the words
that must not count as links (the extension's own name in a logger).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str):
    """A tool, imported by path: ``tools/`` is never on the extension's path."""
    found = sys.modules.get(name)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


affected = _load("affected_tests")
runner = _load("run_tests")


REPOSITORY = {
    "mc_core.py": "VALUE = 1\n",
    "mc_feature.py": "import mc_core\n",
    "mc_other.py": "",
    "reset_only.py": "",
    "worker_paths.py": '''
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent
        WORKER_DIRNAME = "fake_worker"

        def script():
            return ROOT / WORKER_DIRNAME / "worker.py"
    ''',
    "fake_worker/__init__.py": "",
    "fake_worker/worker.py": "import json\n",
    "other_worker/worker.py": "",
    "javascript/page.js": "console.log('page');\n",
    "javascript/other.js": "",
    "style.css": "#page { color: red; }\n",
    "scripts/main_script.py": "import mc_feature\n",
    "README.md": "# A repository\n",
    "tests/conftest.py": '''
        import pytest

        def helper():
            import worker_paths
            return worker_paths

        @pytest.fixture
        def core_home(tmp_path):
            import mc_core
            return mc_core

        @pytest.fixture(autouse=True)
        def _reset():
            import mc_other
            yield

        @pytest.fixture(autouse=True)
        def _reset_more():
            import reset_only
            yield
    ''',
    "tests/test_feature.py": "import mc_feature\n\ndef test_it():\n    pass\n",
    "tests/test_core_fixture.py": "def test_it(core_home):\n    pass\n",
    "tests/test_page_js.py": '''
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent
        SCRIPT = ROOT / "javascript" / "page.js"

        def test_it():
            assert SCRIPT.read_text()
    ''',
    "tests/test_helper_user.py": "from test_page_js import SCRIPT\n\ndef test_it():\n    pass\n",
    "tests/test_scan.py": '''
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent

        def test_every_module():
            assert list(ROOT.glob("mc_*.py"))
    ''',
    "tests/test_by_name.py": '''
        import sys

        def test_it():
            assert sys.modules.get("mc_other") is None
    ''',
    "tests/test_logger.py": '''
        import logging

        def test_it():
            logging.getLogger("main_script").info("the extension's own name is no link")
    ''',
    "tests/test_worker_runner.py": "import worker_paths\n\ndef test_it():\n    pass\n",
    "tests/test_worker_import.py": "from fake_worker import worker\n\ndef test_it():\n    assert worker\n",
    "tests/test_embedded.py": '''
        SOURCE = """
        import mc_core
        print(mc_core.VALUE)
        """

        def test_it():
            assert SOURCE
    ''',
    "tests/test_conftest_helper.py": "from conftest import helper\n\ndef test_it():\n    helper()\n",
    "tests/test_style.py": '''
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent

        def test_it():
            assert (ROOT / "style.css").read_text()
    ''',
    "tests/test_unrelated.py": "def test_it():\n    pass\n",
    # A wildcard over a folder of its own, and a folder's name as a word:
    # neither is every page script.
    "tests/test_tmp_scan.py": "def test_it(tmp_path):\n    assert not list(tmp_path.glob(\"*\"))\n",
    "tests/test_word.py": "KIND = \"javascript\"\n\ndef test_it():\n    assert KIND\n",
    "tests/test_gone_user.py": "import mc_gone\n\ndef test_it():\n    pass\n",
}


@pytest.fixture
def repository(tmp_path):
    for name, text in REPOSITORY.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return tmp_path


def picked(repository, *changed):
    return set(affected.select(changed, repository, files=_files(repository)).tests)


def _files(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


class TestWhatAChangeReaches:
    def test_a_module_reaches_the_tests_that_import_it_however_they_do(self, repository):
        assert picked(repository, "mc_core.py") == {
            "tests/test_feature.py",        # through mc_feature
            "tests/test_core_fixture.py",   # through the fixture it asks for
            "tests/test_embedded.py",       # the Python source held in a string
            "tests/test_scan.py",           # ROOT.glob("mc_*.py")
        }

    def test_a_page_script_reaches_the_test_that_reads_it_and_the_tests_built_on_that(
            self, repository):
        assert picked(repository, "javascript/page.js") == {
            "tests/test_page_js.py", "tests/test_helper_user.py"}

    def test_a_script_nobody_reads_reaches_nothing(self, repository):
        found = affected.select(["javascript/other.js"], repository, files=_files(repository))

        assert found.tests == [] and found.unreached == ["javascript/other.js"]

    def test_a_worker_started_by_a_path_built_from_a_constant_is_that_worker_only(
            self, repository):
        expected = {"tests/test_worker_runner.py", "tests/test_conftest_helper.py",
                    "tests/test_worker_import.py"}

        assert picked(repository, "fake_worker/worker.py") == expected
        assert picked(repository, "other_worker/worker.py") == set()

    def test_a_module_imported_from_its_package_is_that_module(self, repository):
        """``from vibevoice_worker import worker`` is how the runtime loads its
        worker's protocol: the package's ``__init__`` alone is not the link."""
        assert "tests/test_worker_import.py" in picked(repository, "fake_worker/worker.py")
        assert picked(repository, "fake_worker/__init__.py") == {"tests/test_worker_import.py"}

    def test_a_module_named_in_sys_modules_is_a_link_and_a_logger_name_is_not(
            self, repository):
        assert picked(repository, "mc_other.py") == {"tests/test_by_name.py", "tests/test_scan.py"}
        assert "tests/test_logger.py" not in picked(repository, "scripts/main_script.py")

    def test_a_fixture_that_runs_for_every_test_is_not_a_link_from_every_test(self, repository):
        """``_reset`` imports ``mc_other`` for every test; any test picked runs
        it, so it is no reason to pick them all."""
        assert "tests/test_unrelated.py" not in picked(repository, "mc_other.py")

    def test_a_file_only_a_fixture_for_every_test_reaches_picks_every_test(self, repository):
        """Every test would break with it, and nothing else reaches it: picking
        nothing would say "no test reaches this" about a file every test runs."""
        found = affected.select(["reset_only.py"], repository, files=_files(repository))

        assert found.everything
        assert found.because.startswith("reset_only.py")
        assert not affected.select(["mc_other.py"], repository, files=_files(repository)).everything

    def test_a_helper_imported_from_conftest_brings_what_it_reaches(self, repository):
        assert "tests/test_conftest_helper.py" in picked(repository, "worker_paths.py")
        assert "tests/test_conftest_helper.py" not in picked(repository, "mc_core.py")

    def test_the_stylesheet_reaches_the_test_that_reads_it(self, repository):
        assert picked(repository, "style.css") == {"tests/test_style.py"}

    def test_a_changed_test_is_picked_and_so_is_a_test_that_imports_it(self, repository):
        assert picked(repository, "tests/test_unrelated.py") == {"tests/test_unrelated.py"}
        assert picked(repository, "tests/test_page_js.py") == {
            "tests/test_page_js.py", "tests/test_helper_user.py"}

    def test_a_new_module_reaches_the_test_that_scans_for_every_module(self, repository):
        (repository / "mc_new.py").write_text("", encoding="utf-8")

        assert picked(repository, "mc_new.py") == {"tests/test_scan.py"}

    def test_a_deleted_module_reaches_the_tests_that_still_import_it(self, repository):
        assert picked(repository, "mc_gone.py") == {"tests/test_gone_user.py"}

    def test_a_document_reaches_no_test(self, repository):
        found = affected.select(["README.md"], repository, files=_files(repository))

        assert found.tests == [] and found.unreached == ["README.md"]
        assert not found.everything

    @pytest.mark.parametrize("path", ["tests/conftest.py", "pytest.ini", "tools/affected_tests.py",
                                      "tools/run_tests.py"])
    def test_what_every_test_depends_on_picks_every_test(self, repository, path):
        found = affected.select([path, "README.md"], repository, files=_files(repository))

        assert found.everything and found.because == path

    def test_each_test_picked_says_which_chain_picked_it(self, repository):
        found = affected.select(["mc_core.py"], repository, files=_files(repository))

        assert found.reasons["tests/test_feature.py"] == [
            "tests/test_feature.py", "mc_feature.py", "mc_core.py"]


class TestWhatChanged:
    @pytest.fixture
    def repo(self, repository):
        def git(*args):
            subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("-c", "user.name=t", "-c", "user.email=t@example.com", "add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "base")
        git("checkout", "-q", "-b", "work")
        (repository / "mc_core.py").write_text("VALUE = 2\n", encoding="utf-8")
        git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-am", "one")
        (repository / "javascript" / "page.js").write_text("// edited\n", encoding="utf-8")
        (repository / "mc_brand_new.py").write_text("", encoding="utf-8")
        (repository / "mc_other.py").unlink()
        return repository

    def test_committed_edited_new_and_deleted_files_all_count(self, repo):
        assert affected.changed_files(repo) == [
            "javascript/page.js", "mc_brand_new.py", "mc_core.py", "mc_other.py"]

    def test_the_base_is_where_the_branch_left_main(self, repo):
        base = subprocess.run(["git", "rev-parse", "main"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()

        assert affected.default_base(repo) == base


class TestTheRunner:
    def test_the_whole_suite_runs_on_every_core_when_it_can(self):
        line = runner.command(None, serial=False, workers="auto", parallel=True, extra=[])

        assert line[1:] == ["-m", "pytest", "-q", "tests", "-n", "auto"]

    def test_without_pytest_xdist_or_when_asked_it_runs_on_one_core(self):
        assert "-n" not in runner.command(None, serial=False, workers="auto", parallel=False,
                                          extra=[])
        assert "-n" not in runner.command(None, serial=True, workers="auto", parallel=True,
                                          extra=[])

    def test_a_few_test_files_run_on_one_core_and_many_on_all_of_them(self):
        few = ["tests/test_a.py", "tests/test_b.py"]
        many = [f"tests/test_{n}.py" for n in range(runner.PARALLEL_FROM)]

        assert "-n" not in runner.command(few, serial=False, workers="auto", parallel=True,
                                          extra=[])
        assert runner.command(many, serial=False, workers="4", parallel=True,
                              extra=["-x"])[-3:] == ["-n", "4", "-x"]

    def test_the_list_names_each_test_with_its_chain_and_what_no_test_reaches(
            self, repository):
        found = affected.select(["mc_core.py", "README.md"], repository, files=_files(repository))

        lines = runner.describe(found)

        assert "tests/test_feature.py  <-  mc_feature.py  <-  mc_core.py" in lines
        assert lines[-1] == "(no test reaches README.md)"
