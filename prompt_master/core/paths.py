from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Written by the installer next to app.py. It holds one key, ``install_root``,
# which is where every large managed artifact lives: the llama.cpp runtime, the
# GGUF model, the vision projector, the download cache, the logs and the setup
# state. Keeping the marker beside app.py rather than inside the install root is
# what lets ``python app.py`` find a model that was installed on another drive.
MARKER = "install.json"

# Default install root when the installer has not recorded one. Under the
# checkout so a bare ``python app.py`` is self-contained, and named to match the
# convention the one-click installer uses for everything it manages.
DEFAULT_SUBDIR = "user_data"

# Escape hatch for tests, CI, and running several installs side by side.
ROOT_ENV = "PROMPT_MASTER_ROOT"


def application_dir() -> Path:
    """The checkout that owns this source tree — the directory holding app.py.

    ``src/prompt_master/core/paths.py`` sits three levels below it. When the
    package has instead been installed into site-packages that arithmetic lands
    somewhere meaningless, so the result is only accepted when app.py is really
    there; otherwise the per-user application data directory is used.
    """
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "app.py").is_file():
        return candidate
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "PromptMasterStandalone"


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @classmethod
    def discover(cls) -> "AppPaths":
        override = os.environ.get(ROOT_ENV)
        if override:
            return cls(Path(override).expanduser().resolve())
        base = application_dir()
        marker = base / MARKER
        if marker.exists():
            try:
                configured = json.loads(marker.read_text(encoding="utf-8"))["install_root"]
            except (OSError, ValueError, KeyError, TypeError):
                # A corrupt marker must not strand the app on a path it cannot
                # read; setup rewrites it atomically and can simply be re-run.
                return cls((base / DEFAULT_SUBDIR).resolve())
            return cls(Path(configured).expanduser().resolve())
        return cls((base / DEFAULT_SUBDIR).resolve())

    @classmethod
    def record(cls, root: Path) -> Path:
        """Point future launches at ``root``. Returns the marker path written."""
        from .config import atomic_write_json

        marker = application_dir() / MARKER
        atomic_write_json(marker, {"install_root": str(Path(root).resolve())})
        return marker

    @property
    def data(self) -> Path: return self.root / "data"
    # Conversation mode's two stores, kept beside the model rather than under
    # ``data`` because they are the user's own writing: characters are files
    # meant to be copied in and out — the folder name matches the one
    # oobabooga uses — and a chat is a document, not application state.
    @property
    def characters(self) -> Path: return self.root / "characters"
    @property
    def chats(self) -> Path: return self.root / "chats"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def cache(self) -> Path: return self.root / "cache"
    @property
    def state_file(self) -> Path: return self.data / "setup-state.json"

    @property
    def configured(self) -> bool:
        """True once a setup run has completed and left validated state behind."""
        return self.state_file.is_file()

    def create_managed_dirs(self) -> None:
        for path in (self.data, self.logs, self.cache / "downloads", self.cache / "temp-images", self.root / "models", self.root / "runtime", self.characters, self.chats):
            path.mkdir(parents=True, exist_ok=True)

    def contained(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root.resolve() and self.root.resolve() not in candidate.parents:
            raise ValueError("Path escapes installation root")
        return candidate

    def locate(self, recorded: str | Path) -> Path:
        """Where an artifact the state file names actually is.

        Everything setup installs is recorded relative to the install root and
        has to stay inside it — that is ``contained``, and it is what keeps a
        state file from pointing the launcher at an executable somewhere else.
        A model chosen by hand is the one thing that is not installed: it is
        already on the disk, it is 16-27 GiB, and moving it into the install
        root to satisfy a rule about tidiness is not a thing to do to somebody
        else's drive. So an absolute path is taken as given, and a relative one
        is contained exactly as before.

        This is for the weights and the projector — data the server reads.
        The runtime is still resolved with ``contained``, because that one is a
        program this application starts.
        """
        path = Path(recorded).expanduser()
        return path.resolve() if path.is_absolute() else self.contained(path)

    def record(self, path: str | Path) -> str:
        """``path`` as the state file should hold it — relative when it is ours.

        The inverse of ``locate``: a file under the install root is recorded
        relative to it, so the whole installation stays movable, and anything
        outside is recorded as the absolute path it was chosen at.
        """
        resolved = Path(path).expanduser().resolve()
        root = self.root.resolve()
        if resolved == root or root in resolved.parents:
            return resolved.relative_to(root).as_posix()
        return str(resolved)
