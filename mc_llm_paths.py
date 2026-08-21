"""Where LLM Studio keeps its data inside a WebUI installation.

The vendored ``prompt_master`` package discovers its own install root by
looking for the standalone app's ``app.py`` and falling back to a per-user
application-data directory. Neither answer is right inside an extension: the
first file does not exist here, and the second puts a user's characters and
chat threads somewhere unrelated to the WebUI that is showing them.

So the root is decided here instead and handed to ``AppPaths`` directly. The
order below is deliberate:

1. ``PROMPT_MASTER_ROOT``. Upstream's own escape hatch, honoured unchanged, so
   an existing standalone install can be pointed at and reused as-is --
   runtime, weights, characters, chats and all. That is the whole reason to
   resolve the environment variable first rather than last: somebody who
   already provisioned 20 GB of GGUF for the standalone app should not have to
   provision it again to use it here.
2. The ``model_chain_llm_root`` setting, for the same reuse without an
   environment variable, and for putting the weights on a different drive.
3. ``<WebUI data directory>/model_chain_llm``. The same convention presets
   already use: under the data directory rather than in the extension folder,
   so updating or reinstalling the extension does not throw away a user's
   threads, characters, or a 20 GB model.

Nothing here creates directories. Provisioning does that when it runs, and a
user who never opens LLM Studio should not find folders they did not ask for.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_ROOT = "model_chain_llm_root"

OPT_MODELS = "model_chain_llm_models_dir"
"""Where the tab's model chooser looks for GGUFs.

Separate from :data:`OPT_ROOT` because the two answer different questions. The
root is where this extension *keeps* things -- the runtime it starts, the
characters and chats it writes. A models folder is somewhere a user already
has twenty gigabytes of weights, very often on another drive and very often
shared with another front end, and asking them to move it in to be able to
pick from a list would be the wrong way round.
"""

DIRNAME = "model_chain_llm"

MODELS_DIRNAME = "models"
"""Where the provisioner puts a downloaded model, and so the default to scan."""

MANAGED_DIRNAME = "managed"
"""Where the managed catalogue's downloads live, under :data:`MODELS_DIRNAME`.

Under the LLM data root and never under :func:`models_root`, which is a
different question with a different answer. The models folder is somewhere a
user already keeps twenty gigabytes of weights -- very often another drive,
very often shared with another front end -- and writing eight gigabytes of
*our* download into it would be putting managed files in a directory this
extension does not own and cannot tidy up. The managed root is ours: nothing
outside the catalogue writes there, and nothing in it is ever a file somebody
put there by hand.
"""

STAGING_DIRNAME = ".downloads"
"""Where a managed bundle is assembled before it is anything.

A sibling of the installed bundles rather than a temporary directory
elsewhere, so the rename that promotes a verified download is a rename within
one filesystem -- which is what makes it atomic. The leading dot is not
decoration either: it keeps a half-downloaded model out of the folder listings
a user browses, and out of the scan that fills the model chooser.
"""

ROOT_ENV = "PROMPT_MASTER_ROOT"
"""Upstream's own name for this, from ``prompt_master.core.paths``.

Spelled out rather than imported so this module can answer before the vendored
package is importable, which matters when the answer is what makes it
importable.
"""


def data_root() -> Path:
    """The install root LLM Studio should use. See the module docstring."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()

    configured = _setting(OPT_ROOT)
    if configured:
        return Path(str(configured)).expanduser().resolve()

    return (_webui_data_path() / DIRNAME).resolve()


def models_root() -> Path:
    """The folder the tab's model chooser scans.

    The setting when there is one, and the folder provisioning downloads into
    otherwise -- so an installation that has only ever used the pinned model
    still opens the chooser on something rather than on nothing.
    """
    configured = _setting(OPT_MODELS)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return data_root() / MODELS_DIRNAME


def managed_models_root() -> Path:
    """Where downloaded managed backbones live: ``<LLM data root>/models/managed``.

    Deliberately not derived from :func:`models_root`. See
    :data:`MANAGED_DIRNAME` -- the models folder is a place a user keeps their
    own weights, and this is a place the extension keeps its own.

    Creates nothing, like everything else here. The download transaction makes
    the directories it needs at the point it needs them, so an installation
    that never opens the catalogue never grows a folder for it.
    """
    return data_root() / MODELS_DIRNAME / MANAGED_DIRNAME


def managed_staging_root() -> Path:
    """Where a managed bundle is downloaded to before it is promoted."""
    return managed_models_root() / STAGING_DIRNAME


def app_paths():
    """``AppPaths`` rooted at :func:`data_root`.

    Constructed rather than discovered: ``AppPaths.discover()`` would go
    looking for the standalone app's marker file and land in a per-user
    directory that has nothing to do with this WebUI.
    """
    from prompt_master.core.paths import AppPaths

    return AppPaths(data_root())


def configured() -> bool:
    """Whether a setup run has left usable state behind.

    Answered without importing the vendored package, so the tab can say "not
    set up yet" on an installation where the package's own dependencies are
    missing -- which is exactly the installation most likely to be asking.
    """
    try:
        return (data_root() / "data" / "setup-state.json").is_file()
    except OSError:
        return False


def _webui_data_path() -> Path:
    try:
        from modules import paths

        base = Path(paths.data_path)
    except Exception:
        # No host: the extension root is the only directory we can be sure
        # exists. Tests and standalone imports land here.
        base = Path(__file__).resolve().parent
    return base


def _setting(name: str):
    try:
        from modules import shared

        return getattr(shared.opts, name, None)
    except Exception:
        return None
