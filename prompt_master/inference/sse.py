from __future__ import annotations

import json
from collections.abc import Iterable, Iterator


def assistant_chunks(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": return
        try: payload = json.loads(data)
        except json.JSONDecodeError: continue
        choices = payload.get("choices") or []
        if choices:
            content = (choices[0].get("delta") or {}).get("content")
            if isinstance(content, str): yield content
