"""The Image Pipeline's stage card, rendered by a real browser.

Every other test of this panel reads its source. That is what let three rounds
of this refactor ship broken: the card's layout is not decided by what the
stylesheet says, it is decided by what the stylesheet says *next to a theme's*,
in a DOM whose shape belongs to Gradio. Each round was measured against a
fixture built the way the previous bug suggested, and each round the real page
turned out to be a shape the fixture did not have.

So this one renders. It builds the hostile case -- a theme that marks every
declaration `!important`, hangs its own bullet and caret off the header, and
forces `nowrap`; an accordion with an extra child, so the header is *not* the
first one; and a wrapper element around the label text -- loads the real
``style.css`` and the real ``model_chain_pipeline.js``, and asks the browser
where things ended up.

What it asserts is the report, in the words it was reported in: two lines, the
name bigger than the description, and the text clear of the switch.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STYLE = ROOT / "style.css"
PIPELINE_JS = ROOT / "javascript" / "model_chain_pipeline.js"


def _browser() -> str | None:
    """Chromium, wherever this machine keeps it."""
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "/opt/pw-browsers"]
    for base in roots:
        for pattern in ("chromium-*/chrome-linux/chrome",
                        "chromium-*/chrome-linux64/chrome"):
            hits = sorted(glob.glob(str(Path(base) / pattern)))
            if hits:
                return hits[-1]
    return None


BROWSER = _browser()

pytestmark = pytest.mark.skipif(BROWSER is None,
                                reason="no chromium to render with")


PAGE = """<!doctype html><meta charset="utf-8">
<style>%(style)s</style>
<style>
body { font-size: 16px; font-family: system-ui, sans-serif; margin: 0; }
:root { --body-text-color:#eee; --body-text-color-subdued:#9a9aa2;
        --block-border-color:#3a3a42; --text-sm:0.88em; --text-lg:1.2em; }
.col { width: 420px; padding: 14px; }

/* A theme that fights: every declaration marked, its own furniture on the
   header, and the label text pinned to one line at one size. */
.gradio-container .label-wrap {
    display: flex !important; justify-content: space-between !important;
    align-items: center !important; white-space: nowrap !important;
    height: auto !important; padding: .55em .7em !important;
    font-weight: 700 !important; font-size: 1.05em !important;
}
.gradio-container .label-wrap::before { content: "\\25cf"; margin-right: .5em; }
.gradio-container .label-wrap::after { content: "\\25c0"; }
.gradio-container .label-wrap span {
    white-space: nowrap !important; font-weight: 700 !important;
    font-size: 15px !important;
}
</style>
<div class="gradio-container"><div class="col mc-pipeline-panel" id="col"></div></div>
<pre id="out"></pre>
<script>
const CARDS = %(cards)s;
for (const [id, label] of CARDS) {
  const stage = document.createElement('div');
  stage.className = 'mc-pipeline-stage';
  stage.id = 'mc-pipeline-stage-' + id;
  const ed = document.createElement('div');
  ed.className = 'mc-pipeline-editor mc-pipeline-drawer';
  ed.id = 'mc-editor-' + id;
  ed.appendChild(document.createElement('span'));   // the header is not first
  const head = document.createElement('button');
  head.className = 'label-wrap';
  const wrap = document.createElement('span');      // and it is wrapped
  wrap.appendChild(document.createTextNode(label));
  head.appendChild(wrap);
  head.appendChild(Object.assign(document.createElement('span'),
                                 { className: 'icon', textContent: '\\u25bc' }));
  const body = document.createElement('div');
  body.className = 'mc-pipeline-body'; body.style.display = 'none';
  ed.append(head, body);
  const sw = document.createElement('div');
  sw.className = 'mc-pipeline-switch';
  const lab = document.createElement('label');
  lab.append(Object.assign(document.createElement('input'), { type: 'checkbox' }),
             Object.assign(document.createElement('span'), { textContent: 'ON' }));
  sw.appendChild(lab);
  stage.append(ed, sw);
  document.getElementById('col').appendChild(stage);
}
window.gradioApp = () => document;
window.onUiLoaded = f => f();
window.onAfterUiUpdate = () => {};
</script>
<script>%(js)s</script>
<script>
const seen = {};
for (const [id] of CARDS) {
  const ed = document.getElementById('mc-editor-' + id);
  const stage = document.getElementById('mc-pipeline-stage-' + id);
  const name = ed.querySelector('.mc-pipeline-name');
  const said = ed.querySelector('.mc-pipeline-said');
  const lab = stage.querySelector('.mc-pipeline-switch label');
  if (!name || !said) { seen[id] = { split: false }; continue; }
  const n = name.getBoundingClientRect(), s = said.getBoundingClientRect();
  const sw = lab.getBoundingClientRect();
  seen[id] = {
    split: true,
    carded: stage.classList.contains('mc-pipeline-carded'),
    name: name.textContent,
    said: said.textContent,
    stacked: s.top >= n.bottom - 1,
    nameSize: parseFloat(getComputedStyle(name).fontSize),
    saidSize: parseFloat(getComputedStyle(said).fontSize),
    nameWeight: getComputedStyle(name).fontWeight,
    clearOfSwitch: Math.max(n.right, s.right) <= sw.left + 1,
    withinBand: s.bottom <= stage.getBoundingClientRect().bottom + 1,
    strays: (function () {
      const head = ed.querySelector('.mc-pipeline-card-head');
      if (!head) return -1;
      return [...head.children].filter(function (kid) {
        return !kid.classList.contains('mc-pipeline-label')
               && getComputedStyle(kid).display !== 'none';
      }).length;
    })(),
  };
}
document.getElementById('out').textContent = JSON.stringify(seen);
</script>
"""


CARDS = [["creative", "Creative\nBypassed — prompt as-is"],
         ["spatial", "Spatial\nBypassed — 4 regions"],
         ["stage2", "Stage 2\n1024 × 1024 in · Bypassed"]]
"""Exactly what mc_pipeline_panel.card_label() writes: one string, one newline."""


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    page = tmp_path_factory.mktemp("render") / "card.html"
    page.write_text(PAGE % {
        "style": STYLE.read_text(encoding="utf-8"),
        "js": PIPELINE_JS.read_text(encoding="utf-8"),
        "cards": json.dumps(CARDS),
    }, encoding="utf-8")

    done = subprocess.run(
        [BROWSER, "--headless", "--no-sandbox", "--disable-gpu",
         "--window-size=900,700", "--virtual-time-budget=2000",
         "--dump-dom", page.as_uri()],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]

    body = done.stdout
    assert '<pre id="out">' in body, "the page did not report"
    said = body.split('<pre id="out">', 1)[1].split("</pre>", 1)[0]
    said = said.replace("&quot;", '"').replace("&amp;", "&")
    said = said.replace("&lt;", "<").replace("&gt;", ">")
    return json.loads(said)


class TestTheCardRenders:
    def test_every_card_is_split_into_two_elements(self, rendered):
        """The newline is not styled any more, it is removed: the browser file
        finds the text Python wrote and makes it two elements of this
        extension's own. If that does not happen there is nothing to lay out."""
        for stage, found in rendered.items():
            assert found["split"], stage
            assert found["carded"], stage

    def test_the_two_lines_are_two_lines(self, rendered):
        """The report this exists for, in the words it was reported in: "the
        title and description live on the same line"."""
        for stage, found in rendered.items():
            assert found["stacked"], (stage, found)

    def test_the_name_reads_differently_than_the_description(self, rendered):
        """A theme that pins every piece of text in a header to one size is
        what made two lines look like one paragraph."""
        for stage, found in rendered.items():
            assert found["nameSize"] > found["saidSize"], (stage, found)
            assert int(found["nameWeight"]) >= 600, (stage, found)

    def test_nothing_runs_underneath_the_switch(self, rendered):
        """The lane the header's padding reserves, measured rather than
        assumed."""
        for stage, found in rendered.items():
            assert found["clearOfSwitch"], (stage, found)

    def test_the_description_is_inside_the_card(self, rendered):
        """A band that clips its own second line is the same bug as not having
        one."""
        for stage, found in rendered.items():
            assert found["withinBand"], (stage, found)

    def test_the_header_holds_nothing_but_the_two_lines(self, rendered):
        """A theme is free to hang a chevron, a bullet or a caret off its own
        accordion headers. This one has two lines' room and a switch painted
        over the end of it, so once the band became a block those markers each
        took a line of their own and pushed the description out of the card."""
        for stage, found in rendered.items():
            assert found["strays"] == 0, (stage, found)

    def test_the_names_are_the_names(self, rendered):
        assert rendered["creative"]["name"] == "Creative"
        assert rendered["spatial"]["name"] == "Spatial"
        assert rendered["stage2"]["name"] == "Stage 2"
        assert rendered["stage2"]["said"].startswith("1024")
