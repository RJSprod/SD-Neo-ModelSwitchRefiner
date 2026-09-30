"""The test files a change can reach, worked out from the repository alone.

``python3 tools/run_tests.py --affected`` runs what this picks. It is for while
you work. The full suite, on every core, stays the check before a push, because
some of what connects a change to a test is written nowhere this can read: the
state one test leaves for the next, and the order tests happen to run in.

What it follows, from every Python file in the repository, tests included:

* imports: ``import x``, ``from x import y``, relative imports, a module loaded
  by name (``importlib.import_module("x")``, ``sys.modules["x"]``, a
  ``patch("x.y")`` target), and the imports of Python source held in a string
  (the fake workers ``tests/conftest.py`` writes out and starts);
* repository files named in strings: ``"voice_box.js"``,
  ``ROOT / "javascript" / "voice_box.js"``, ``os.path.join(..., "worker.py")``,
  ``"vibevoice_worker.worker"``;
* folders a file lists: ``ROOT.glob("mc_voice_*.py")``,
  ``(ROOT / "scripts").glob("*.py")``, ``iterdir``, ``os.walk``;
* the fixtures of ``tests/conftest.py`` a test asks for by name, and the helpers
  it imports from there, with everything each of those reaches.

A test file is picked when a changed file can be reached from it through those
links, or when it changed itself. A few files pick every test: the fixtures that
run for every test live in ``tests/conftest.py``, and ``pytest.ini``, this file
and the runner decide what runs at all. It errs towards picking too much -- a
file named in a string is a link whether or not the code ever opens it -- but a
bare word is not a module and a folder's name is not every file in it, or the
extension's own name (``"model_chain"``, its logger and every setting's prefix)
would tie every test to every other.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
import subprocess
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CONFTEST = "tests/conftest.py"
# A change to one of these changes what every test sees, or which tests run.
EVERYTHING = (CONFTEST, "pytest.ini", "tools/affected_tests.py", "tools/run_tests.py")
# Folders whose modules are imported by their bare name: the extension root and
# scripts/ (the host puts both on sys.path), tests/ (pytest does), and tools/
# (whose scripts the tests load by path under their own names).
FLAT = ("", "scripts", "tests", "tools")
SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
        "model_chain_llm", "model_chain_voice"}

FILE_NAME = re.compile(r"^[\w.\-]+\.[A-Za-z0-9]{1,8}$")
GLOB = re.compile(r"^[\w.\-/\[\]*?]*[*?][\w.\-/\[\]*?]*$")
DOTTED = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
MODULE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
# Calls whose first string names a module to load.
IMPORT_CALLS = {"import_module", "__import__", "find_spec", "spec_from_file_location", "reload"}
# Calls whose first string is a dotted target inside a module (``patch("x.y")``).
TARGET_CALLS = {"patch", "setattr", "delattr"}
# Calls that list a whole folder, and whether they go down into its folders.
FOLDER_CALLS = {"listdir": False, "scandir": False, "walk": True, "copytree": True}
FIXTURE_CALLS = ("usefixtures", "getfixturevalue")


def is_test(path: str) -> bool:
    return path.startswith("tests/") and path.count("/") == 1 and path.endswith(".py") \
        and PurePosixPath(path).name.startswith("test_")


def repo_files(root: Path = ROOT) -> list:
    """Every file git tracks or would add, as POSIX paths from ``root``."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True).stdout
        names = {name for name in listed.decode("utf-8", "surrogateescape").split("\0") if name}
    except (OSError, subprocess.CalledProcessError):
        names = set()
        for folder, folders, leaves in os.walk(root):
            folders[:] = [name for name in folders if name not in SKIP]
            names.update(Path(folder, leaf).relative_to(root).as_posix() for leaf in leaves)
    return sorted(name for name in names if (root / name).is_file())


def module_names(path: str) -> list:
    """The names a Python file is imported by."""
    pure = PurePosixPath(path)
    parts = pure.with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    names = [".".join(parts)]
    if "/".join(pure.parts[:-1]) in FLAT and pure.name != "__init__.py":
        names.append(parts[-1])
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------- #
# Reading one Python file
# --------------------------------------------------------------------------- #


@dataclass
class Reading:
    """What one Python file, or one definition in it, names."""

    imports: set = field(default_factory=set)      # dotted names, with their prefixes
    strings: set = field(default_factory=set)      # string constants and joined paths
    globs: set = field(default_factory=set)        # (folder or "", pattern, recursive)
    names: set = field(default_factory=set)        # identifiers it uses
    requested: set = field(default_factory=set)    # fixture names it asks for
    from_conftest: set = field(default_factory=set)  # names it imports from conftest

    def merge(self, other: "Reading") -> None:
        for name in ("imports", "strings", "globs", "names", "requested", "from_conftest"):
            getattr(self, name).update(getattr(other, name))


def _prefixes(name: str) -> list:
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(len(parts), 0, -1)]


def _specific(pattern: str) -> bool:
    """Whether a glob names something beyond a wildcard and an extension."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", pattern)
    return re.search(r"[A-Za-z0-9_\-]{3,}", stem) is not None


def _path_parts(node, constants: dict | None = None) -> list | None:
    """The string parts at the end of a ``a / "b" / "c"`` chain, or None.

    A name bound to a string at the top of the file counts as that string
    (``root / PIPELINE_WORKER_DIRNAME / "worker.py"``); any other name ends the
    chain, so only the parts after it are known."""
    constants = constants or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Name) and node.id in constants:
        return [constants[node.id]]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        right = _path_parts(node.right, constants)
        if right is not None and len(right) == 1:
            return (_path_parts(node.left, constants) or []) + right
    return None


def _constants(tree) -> dict:
    """Names bound to a string at the top of a file."""
    found = {}
    for node in getattr(tree, "body", ()):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found[node.targets[0].id] = node.value.value
    return found


def _string_arg(node, index: int = 0) -> str | None:
    if len(node.args) > index:
        arg = node.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _is_sys_modules(node) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "modules"


def _docstrings(tree) -> set:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                found.add(id(body[0].value))
    return found


class _Reader(ast.NodeVisitor):
    def __init__(self, package: tuple, skip: set, constants: dict | None = None):
        self.package = package
        self.skip = skip
        self.constants = constants or {}
        self.found = Reading()

    def _module(self, name: str) -> None:
        if name and MODULE.match(name):
            self.found.imports.update(_prefixes(name))

    def visit_Import(self, node):
        for alias in node.names:
            self.found.imports.update(_prefixes(alias.name))

    def visit_ImportFrom(self, node):
        if node.level:
            keep = len(self.package) - (node.level - 1)
            base = list(self.package[:max(keep, 0)])
            if node.module:
                base.append(node.module)
            base = ".".join(base)
        else:
            base = node.module or ""
        if base == "conftest":
            self.found.from_conftest.update(alias.name for alias in node.names)
            return
        if base:
            self.found.imports.update(_prefixes(base))
        for alias in node.names:
            if alias.name != "*":
                self.found.imports.add(f"{base}.{alias.name}" if base else alias.name)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self.skip:
            self._string(node.value)

    def _string(self, text: str) -> None:
        self.found.strings.add(text)
        # Python source held in a string: a fake worker written out and run.
        if "\n" in text and "import " in text:
            try:
                tree = ast.parse(textwrap.dedent(text))
            except (SyntaxError, ValueError):
                return
            inner = _Reader(self.package, _docstrings(tree), _constants(tree))
            inner.visit(tree)
            self.found.merge(inner.found)

    def visit_BinOp(self, node):
        parts = _path_parts(node, self.constants)
        if parts and len(parts) > 1:
            # The whole path is the link; its last part alone ("worker.py")
            # would name every file of that name.
            self.found.strings.add("/".join(parts))
            chain = node
            while isinstance(chain, ast.BinOp):
                if isinstance(chain.right, ast.Constant):
                    self.skip.add(id(chain.right))
                chain = chain.left
            if isinstance(chain, ast.Constant):
                self.skip.add(id(chain))
        self.generic_visit(node)

    def visit_Name(self, node):
        self.found.names.add(node.id)

    def visit_Subscript(self, node):
        key = node.slice
        if _is_sys_modules(node.value) and isinstance(key, ast.Constant) \
                and isinstance(key.value, str):
            self._module(key.value)
        self.generic_visit(node)

    def visit_Compare(self, node):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) \
                and any(_is_sys_modules(item) for item in node.comparators):
            self._module(node.left.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = _string_arg(node)
        if name == "join":
            parts = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value)
                else:
                    parts = []
            if len(parts) > 1:
                self.found.strings.add("/".join(parts))
        if name in ("glob", "rglob") and isinstance(func, ast.Attribute) and first is not None:
            base = _path_parts(func.value, self.constants)
            self.found.globs.add(("/".join(base) if base else "", first, name == "rglob"))
            self.skip.add(id(node.args[0]))
        elif name == "iterdir" and isinstance(func, ast.Attribute):
            base = _path_parts(func.value, self.constants)
            if base:
                self.found.globs.add(("/".join(base), "*", False))
        elif name in FOLDER_CALLS and node.args:
            base = _path_parts(node.args[0], self.constants)
            if base:
                self.found.globs.add(("/".join(base), "*", FOLDER_CALLS[name]))
        if name in IMPORT_CALLS and first is not None:
            self._module(first)
        if isinstance(func, ast.Attribute) and _is_sys_modules(func.value) and first is not None:
            self._module(first)
        if name in ("setitem", "delitem") and node.args and _is_sys_modules(node.args[0]):
            self._module(_string_arg(node, 1) or "")
        if name in TARGET_CALLS and first is not None and DOTTED.match(first):
            self.found.imports.update(_prefixes(first))
        if name in FIXTURE_CALLS:
            self.found.requested.update(arg.value for arg in node.args
                                        if isinstance(arg, ast.Constant)
                                        and isinstance(arg.value, str))
        self.generic_visit(node)

    def _function(self, node):
        arguments = node.args
        for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
            self.found.requested.add(arg.arg)
        self.generic_visit(node)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function


def read_source(source: str, path: str) -> Reading:
    """What a Python file names. A file that does not parse names nothing."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return Reading()
    return _read_tree(tree, path)


def _read_tree(tree, path: str, skip: set | None = None, constants: dict | None = None) -> Reading:
    reader = _Reader(tuple(PurePosixPath(path).parent.parts),
                     skip if skip is not None else _docstrings(tree),
                     constants if constants is not None else _constants(tree))
    reader.visit(tree)
    return reader.found


def _fixture(node) -> tuple:
    """(is a fixture, runs for every test) for a function definition."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "fixture":
            every = isinstance(decorator, ast.Call) and any(
                keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True for keyword in decorator.keywords)
            return True, every
    return False, False


# --------------------------------------------------------------------------- #
# The repository as a graph of files
# --------------------------------------------------------------------------- #


class Graph:
    """Every file, and for each Python file the files it reaches directly."""

    def __init__(self, root: Path = ROOT, files: Iterable[str] | None = None):
        self.root = Path(root)
        self.files = sorted(files if files is not None else repo_files(self.root))
        self.present = set(self.files)
        self.by_module: dict = {}
        self.by_name: dict = {}
        self.folders: dict = {}
        for path in self.files:
            if path.endswith(".py"):
                for name in module_names(path):
                    self.by_module.setdefault(name, set()).add(path)
            self.by_name.setdefault(PurePosixPath(path).name, set()).add(path)
            parts = PurePosixPath(path).parts
            for end in range(1, len(parts)):
                self.folders.setdefault("/".join(parts[:end]), set()).add(path)
        self.readings: dict = {}
        self._definitions, self.fixtures, self.every_test = self._read_conftest()
        self._reached_by_definition: dict = {}
        self.edges = {path: self._reach(path) for path in self.files if path.endswith(".py")}
        # What the fixtures that run for every test reach, all the way down.
        self.every_test_reach = self._closure(
            set().union(*(self.definition_reach(name) for name in self.every_test)))

    def _source(self, path: str) -> str:
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def reading(self, path: str) -> Reading:
        if path not in self.readings:
            self.readings[path] = read_source(self._source(path), path)
        return self.readings[path]

    # -- tests/conftest.py, one definition at a time --------------------------- #

    def _read_conftest(self) -> tuple:
        """Each top-level definition's reading, which of them are fixtures, and
        which fixtures run for every test.

        A fixture that runs for every test (``autouse``) is never asked for by
        name, so it is never followed: any test picked runs it anyway, and
        following it would tie every test to what it imports."""
        if CONFTEST not in self.present:
            return {}, set(), set()
        try:
            tree = ast.parse(self._source(CONFTEST))
        except (SyntaxError, ValueError):
            return {}, set(), set()
        skip = _docstrings(tree)
        constants = _constants(tree)
        definitions: dict = {}
        fixtures: set = set()
        every_test: set = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [target.id for target in targets if isinstance(target, ast.Name)]
            else:
                continue
            reading = _read_tree(node, CONFTEST, skip, constants)
            for name in names:
                definitions[name] = reading
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_fixture, every = _fixture(node)
                if is_fixture:
                    fixtures.add(node.name)
                    if every:
                        every_test.add(node.name)
        return definitions, fixtures, every_test

    def _closure(self, start: set) -> set:
        found = set(start)
        queue = deque(start)
        while queue:
            for target in self.edges.get(queue.popleft(), ()):
                if target not in found:
                    found.add(target)
                    queue.append(target)
        return found

    def definition_reach(self, name: str) -> set:
        """What a conftest definition reaches, with the definitions and
        fixtures it uses in turn."""
        if name not in self._reached_by_definition:
            merged = Reading()
            seen: set = set()
            queue = deque([name])
            while queue:
                current = queue.popleft()
                if current in seen or current not in self._definitions:
                    continue
                seen.add(current)
                found = self._definitions[current]
                merged.imports |= found.imports
                merged.strings |= found.strings
                merged.globs |= found.globs
                queue.extend(found.names | found.requested)
            self._reached_by_definition[name] = self._resolve(merged, CONFTEST)
        return self._reached_by_definition[name]

    # -- resolving ------------------------------------------------------------ #

    def _reach(self, path: str) -> set:
        reading = self.reading(path)
        found = self._resolve(reading, path)
        for name in reading.from_conftest:
            found |= self.definition_reach(name)
        if is_test(path):
            for name in reading.requested & self.fixtures:
                found |= self.definition_reach(name)
        found.discard(path)
        if path != CONFTEST and reading.from_conftest:
            found.discard(CONFTEST)
        return found

    def _resolve(self, reading: Reading, path: str) -> set:
        found = set()
        folder = "/".join(PurePosixPath(path).parts[:-1])
        for name in reading.imports:
            found |= self.by_module.get(name, set())
            if folder:
                # A worker run as a script imports the modules beside it by bare name.
                local = f"{folder}/{name.replace('.', '/')}"
                found |= {candidate for candidate in (local + ".py", local + "/__init__.py")
                          if candidate in self.present}
        for text in reading.strings:
            found |= self.named(text)
        for base, pattern, recursive in reading.globs:
            found |= self.globbed(base, pattern, recursive)
        return found

    def named(self, text: str) -> set:
        """The repository files a string names, if any."""
        text = text.strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        text = text.rstrip("/")
        if not text or len(text) > 260 or "\n" in text or " " in text:
            return set()
        if text in self.present:
            return {text}
        if "/" in text:
            if text in self.folders:
                return set(self.folders[text])
            return {path for path in self.present if path.endswith("/" + text)}
        if FILE_NAME.match(text) and text in self.by_name:
            return set(self.by_name[text])
        if DOTTED.match(text) and text in self.by_module:
            return set(self.by_module[text])
        if GLOB.match(text):
            return self.globbed("", text, True)
        return set()

    def globbed(self, base: str, pattern: str, recursive: bool) -> set:
        """Files a glob can find: inside ``base`` when that is a repository
        folder, and otherwise anywhere -- but only for a pattern with a name in
        it (``mc_voice_*.py``), because ``*`` or ``*.json`` over a folder this
        cannot see is far more often a test's own temporary folder."""
        if base and base in self.folders:
            prefix = base + "/"
            found = set()
            for path in self.folders[base]:
                rest = path[len(prefix):]
                if not recursive and "/" in rest:
                    continue
                if fnmatch.fnmatch(rest, pattern) or fnmatch.fnmatch(PurePosixPath(path).name, pattern):
                    found.add(path)
            return found
        if not _specific(pattern):
            return set()
        return {path for path in self.present
                if fnmatch.fnmatch(PurePosixPath(path).name, pattern)
                or fnmatch.fnmatch(path, pattern)}

    # -- what a missing file was called --------------------------------------- #

    def mentions_of_missing(self, missing: str) -> set:
        """Files that still name a file that is gone: they break with it."""
        names = set(module_names(missing)) if missing.endswith(".py") else set()
        leaf = PurePosixPath(missing).name
        found = set()
        for path in self.edges:
            reading = self.reading(path)
            if names & reading.imports:
                found.add(path)
            elif any(text in (leaf, missing) or text.endswith("/" + missing)
                     or text.endswith("/" + leaf) for text in reading.strings):
                found.add(path)
        return found


# --------------------------------------------------------------------------- #
# Selecting
# --------------------------------------------------------------------------- #


@dataclass
class Selection:
    changed: list
    everything: bool = False
    because: str = ""
    tests: list = field(default_factory=list)
    # For each test picked, a chain of files from it to a changed file.
    reasons: dict = field(default_factory=dict)
    # Changed files no test reaches (documents, say).
    unreached: list = field(default_factory=list)


def select(changed: Iterable[str], root: Path = ROOT, files: Iterable[str] | None = None,
           graph: Graph | None = None) -> Selection:
    """The test files that can reach any of ``changed`` (POSIX paths from ``root``)."""
    changed = sorted({str(path).replace("\\", "/") for path in changed if str(path).strip()})
    selection = Selection(changed=changed)
    for path in changed:
        if path in EVERYTHING:
            selection.everything = True
            selection.because = path
            return selection
    graph = graph or Graph(root, files)
    backwards: dict = {}
    for source, targets in graph.edges.items():
        for target in targets:
            backwards.setdefault(target, set()).add(source)
    for path in changed:
        if path not in graph.present:
            backwards.setdefault(path, set()).update(graph.mentions_of_missing(path))
    came_from: dict = {path: None for path in changed}
    queue = deque(changed)
    while queue:
        current = queue.popleft()
        for source in sorted(backwards.get(current, ())):
            if source not in came_from:
                came_from[source] = current
                queue.append(source)
    for path in sorted(came_from):
        if is_test(path) and path in graph.present:
            chain = [path]
            while came_from[chain[-1]] is not None:
                chain.append(came_from[chain[-1]])
            selection.tests.append(path)
            selection.reasons[path] = chain
    selection.unreached = [path for path in changed
                           if not _reaches_a_test(path, backwards, graph)]
    # A file reached only through a fixture that runs for every test: every
    # test would break with it, and nothing above picked one. Pick them all.
    if not selection.tests:
        hidden = [path for path in changed if path in graph.every_test_reach]
        if hidden:
            selection.everything = True
            selection.because = f"{hidden[0]} (reached through a fixture every test runs)"
    return selection


def _reaches_a_test(path: str, backwards: dict, graph: Graph) -> bool:
    seen = {path}
    queue = deque([path])
    while queue:
        current = queue.popleft()
        if is_test(current) and current in graph.present:
            return True
        for source in backwards.get(current, ()):
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return False


# --------------------------------------------------------------------------- #
# What changed
# --------------------------------------------------------------------------- #


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def default_base(root: Path = ROOT) -> str:
    """Where the work began: the fork point from main, or the last commit."""
    for ref in ("origin/main", "main"):
        found = _git(root, "merge-base", "HEAD", ref)
        if found and found.strip():
            return found.strip()
    return "HEAD"


def changed_files(root: Path = ROOT, base: str | None = None) -> list:
    """Files changed since ``base``: committed, staged, unstaged or new."""
    base = base or default_base(root)
    listed = _git(root, "diff", "--name-only", "--no-renames", base, "--") or ""
    untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    return sorted({line.strip() for line in (listed + "\n" + untracked).splitlines()
                   if line.strip()})
