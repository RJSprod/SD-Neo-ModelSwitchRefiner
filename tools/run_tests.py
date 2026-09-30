"""Run the suite: all of it on every core, or only the tests a change can reach.

    python3 tools/run_tests.py                     every test, on every core
    python3 tools/run_tests.py --affected          only the tests a change reaches
    python3 tools/run_tests.py --affected --list   name them and why; run nothing
    python3 tools/run_tests.py --serial            every test on one core, in order
    python3 tools/run_tests.py -- -k sampling      anything after -- goes to pytest

The full run is the check before a push. ``--affected`` is for while you work:
it compares the working tree with where the branch left main (``--base`` names
another commit) and runs what ``affected_tests.py`` picks, which cannot see a
test that fails because of what another test left behind.

Every core needs pytest-xdist (``pip install pytest-xdist``); without it the
run is on one core and says so.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

try:
    import affected_tests
except ImportError:  # started from somewhere other than tools/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import affected_tests

ROOT = affected_tests.ROOT
# Below this many test files a run on one core is quicker than starting workers.
PARALLEL_FROM = 4


def parallel_available() -> bool:
    return importlib.util.find_spec("xdist") is not None


def command(tests: list | None, *, serial: bool, workers: str, parallel: bool,
            extra: list) -> list:
    """The pytest command line: ``tests`` None means the whole suite."""
    line = [sys.executable, "-m", "pytest", "-q"]
    line += tests if tests is not None else ["tests"]
    wanted = tests is None or len(tests) >= PARALLEL_FROM
    if parallel and not serial and wanted:
        line += ["-n", workers]
    return line + list(extra)


def describe(selection: affected_tests.Selection) -> list:
    """What ``--list`` says: each test picked, and the chain that picked it."""
    lines = []
    if selection.everything:
        return [f"every test: {selection.because} changed"]
    for test in selection.tests:
        chain = selection.reasons[test]
        lines.append(test if len(chain) == 1 else f"{test}  <-  " + "  <-  ".join(chain[1:]))
    for path in selection.unreached:
        lines.append(f"(no test reaches {path})")
    return lines


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        at = argv.index("--")
        argv, extra = argv[:at], argv[at + 1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--affected", action="store_true",
                        help="only the test files a change can reach")
    parser.add_argument("--base", help="compare with this commit instead of the fork from main")
    parser.add_argument("--list", action="store_true", help="name the tests picked; run nothing")
    parser.add_argument("--serial", action="store_true", help="one core, in file order")
    parser.add_argument("--workers", default="auto", help="pytest-xdist workers (default: auto)")
    options = parser.parse_args(argv)

    tests = None
    if options.affected or options.list:
        changed = affected_tests.changed_files(ROOT, options.base)
        selection = affected_tests.select(changed, ROOT)
        if options.list:
            print("\n".join(describe(selection)) or "nothing changed")
            return 0
        if not selection.everything:
            if not selection.tests:
                print("No test reaches what changed"
                      + (f" ({', '.join(selection.unreached)})" if selection.unreached else "")
                      + ". Nothing to run.")
                return 0
            tests = selection.tests
            print(f"{len(tests)} test files reach what changed "
                  f"({len(changed)} files); --list says why.")
        else:
            print(f"Every test: {selection.because} changed.")

    parallel = parallel_available()
    if not parallel and not options.serial:
        print("pytest-xdist is not installed, so this runs on one core "
              "(pip install pytest-xdist).")
    line = command(tests, serial=options.serial, workers=options.workers,
                   parallel=parallel, extra=extra)
    return subprocess.call(line, cwd=ROOT, env=dict(os.environ))


if __name__ == "__main__":
    sys.exit(main())
