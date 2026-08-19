from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Component:
    component_id: str; url: str; destination: str; size: int | None; sha256: str; version: str

    def validate(self) -> None:
        if urlparse(self.url).scheme != "https": raise ValueError(f"{self.component_id}: HTTPS URL required")
        if self.size is not None and self.size <= 0: raise ValueError(f"{self.component_id}: invalid size")
        if len(self.sha256) != 64: raise ValueError(f"{self.component_id}: incomplete SHA-256")
        if "latest" in self.url.casefold(): raise ValueError(f"{self.component_id}: unpinned latest URL")


def load_manifest(path: Path) -> dict[str, Component]:
    raw = json.loads(path.read_text(encoding="utf-8")); result = {}
    for item in raw["components"]:
        component = Component(**item); component.validate(); result[component.component_id] = component
    return result
