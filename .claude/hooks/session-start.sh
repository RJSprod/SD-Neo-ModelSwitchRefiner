#!/bin/bash
# Claude Code on the web: make the test suite able to run on every core.
#
# tools/run_tests.py runs the suite with pytest-xdist when it is installed and
# on one core when it is not. Web sessions start from an image without it, so
# this installs it once; the container is cached after the hook, so later
# sessions find it already there.
#
# Two more, for the same reason, since handoff 43. httpx: prompt_master's
# downloader imports it at module level, so without it 135 LLM tests error
# before they run (handoffs 42 and 43 both lost time to it). Playwright's
# Python package: without it every real-browser test skips, and the browser
# tests are the ones that have caught what a node stub could not; the tests'
# own launcher falls back to the container's Chromium when Playwright's build
# of it is not there. Nothing else: the suite's other packages come with the
# image, and adding one (psutil, say) could change which code paths the tests
# take, where these two only decide whether they run at all.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

wanted=""
python3 -c "import xdist" >/dev/null 2>&1 || wanted="$wanted pytest-xdist>=3.6"
python3 -c "import httpx" >/dev/null 2>&1 || wanted="$wanted httpx"
python3 -c "import playwright" >/dev/null 2>&1 || wanted="$wanted playwright"

if [ -z "$wanted" ]; then
  exit 0
fi

# shellcheck disable=SC2086  # one argument per package, deliberately
python3 -m pip install --quiet --disable-pip-version-check --root-user-action=ignore $wanted
