"""Runtime components this extension pins beyond the vendored release manifest.

``prompt_master/release-manifest.json`` is part of a byte-identical vendored
tree -- ``prompt_master/VENDORED_FROM.txt`` says so, and says where changes
belong instead: in the ``mc_llm_*`` modules layered on top. This is that
module for runtime archives. It pins the llama.cpp builds the vendored manifest
does not know about, in the same shape the manifest uses (``Component``, with
an HTTPS URL, a SHA-256 and a version), so the same verified downloader and the
same atomic extractor install them and nothing downstream can tell the two
sources apart.

One family today: llama.cpp's SYCL build for Intel graphics, pinned to the
same ``b10621`` release the vendored CUDA and CPU families come from. Keeping
every family on one release is deliberate -- the flags this extension probes
for, the log lines it reads back and the verbosity scale it asks for are all
facts about a build, and a machine running two releases side by side would be
two sets of facts to keep straight.

The other thing this module owns is *which* archives a device needs. The
vendored rule is "every non-CPU device wants a runtime archive and a cudart
archive beside it", which is true of every CUDA family and false of SYCL: the
SYCL zip carries its oneAPI runtime libraries inside it and has no companion.
So the choice is made by backend here, and the vendored rule is consulted only
for the devices it was written for.
"""

from __future__ import annotations

import logging

from prompt_master.provisioning.manifest import Component

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

LLAMA_RELEASE = "b10621"
"""The llama.cpp release every runtime family is pinned to. See the module docstring."""

SYCL_RUNTIME = "llama-runtime-sycl"
"""The component id, runtime-family marker and ``runtime_id`` of the SYCL build."""

COMPONENTS: dict[str, Component] = {
    SYCL_RUNTIME: Component(
        component_id=SYCL_RUNTIME,
        url=(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/"
             f"llama-{LLAMA_RELEASE}-bin-win-sycl-x64.zip"),
        destination=f"cache/downloads/llama-{LLAMA_RELEASE}-bin-win-sycl-x64.zip",
        size=None,
        # The official release's own figure, and verified against the archive
        # itself when this was written: 119,290,141 bytes, 72 entries,
        # llama-server.exe beside ggml-sycl.dll, sycl8.dll and the Level Zero
        # adapter DLLs, and no cudart anything.
        sha256="9744eb81396e5b52ecb85591c1c9a5e767de5ce8aedb741bd971d89e14295004",
        version=f"{LLAMA_RELEASE}-sycl",
    ),
}
"""Every runtime archive pinned here, by component id."""


def components() -> dict[str, Component]:
    """Every pinned component: the vendored manifest's, plus this module's.

    The vendored manifest wins a collision, deliberately. If a later upstream
    release ever pins a SYCL build of its own, that is the one to install --
    the pin here exists because upstream has none, not to override one.
    """
    from prompt_master.provisioning.installer import load_components

    found = dict(load_components())
    for key, component in COMPONENTS.items():
        component.validate()
        found.setdefault(key, component)
    return found


def runtime_component_id(device) -> str:
    """The runtime family ``device`` needs.

    SYCL for an Intel device, and the vendored rule -- pinned card, compute
    capability, model number -- for everything it was written for. Asked here
    rather than of the vendored function directly, because the vendored
    function has never heard of Intel and answers "cuda12" for anything it
    cannot place, which would record a CUDA build beside an Intel device and
    refuse ``--device SYCL0`` at every start.
    """
    import mc_llm_sycl

    if mc_llm_sycl.is_sycl_device(device):
        return SYCL_RUNTIME
    from prompt_master.inference.device_detection import runtime_component_id as vendored

    return vendored(device)


def runtime_component_ids(device) -> tuple[str, ...]:
    """The archives ``device`` needs, in download order.

    One for SYCL: the archive is self-contained. The vendored pairing of a
    runtime archive with its cudart companion holds for every CUDA family and
    for nothing else, so it is consulted only for those.
    """
    import mc_llm_sycl

    if mc_llm_sycl.is_sycl_device(device):
        return (SYCL_RUNTIME,)
    from prompt_master.provisioning.installer import runtime_component_ids as vendored

    return vendored(device)


def is_sycl_family(component_id: str) -> bool:
    """Whether a runtime-family marker names the SYCL build."""
    return str(component_id or "").strip().casefold() == SYCL_RUNTIME
