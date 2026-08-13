from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted({name for name in _VARIABLE.findall(value) if name not in variables})
    if missing:
        names = ", ".join(missing)
        raise KeyError(f"Unset environment variable(s) in config: {names}")
    return _VARIABLE.sub(lambda match: variables[match.group(1)], value)


def _load_document(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive config include: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise TypeError(f"Config root must be a mapping: {path}")

    includes = document.pop("include", document.pop("includes", []))
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = _deep_update(merged, _load_document(include_path, (*stack, path)))
    return _deep_update(merged, document)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with recursive includes and environment expansion.

    ``REPO_ROOT`` is always available. Dataset paths intentionally require an
    explicit ``DATA_ROOT`` so a run cannot silently use a developer's machine.
    """

    config_path = Path(path).expanduser().resolve()
    variables = {**os.environ, "REPO_ROOT": str(REPO_ROOT)}
    config = _expand(_load_document(config_path, ()), variables)
    config["_config_path"] = str(config_path)
    return config
