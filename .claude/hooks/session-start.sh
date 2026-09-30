#!/bin/bash
# Claude Code on the web: make the test suite able to run on every core.
#
# tools/run_tests.py runs the suite with pytest-xdist when it is installed and
# on one core when it is not. Web sessions start from an image without it, so
# this installs it once; the container is cached after the hook, so later
# sessions find it already there. Nothing else is installed: the suite's other
# packages come with the image, and adding one (psutil, say) could change which
# code paths the tests take.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if python3 -c "import xdist" >/dev/null 2>&1; then
  exit 0
fi

python3 -m pip install --quiet --disable-pip-version-check --root-user-action=ignore "pytest-xdist>=3.6"
