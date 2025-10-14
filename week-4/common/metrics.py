"""Metrics logging helpers shared by Week 4 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


try:  # DDP utilities live alongside Example 03
    import ddp_utils  # type: ignore
except ImportError:  # pragma: no cover - available only when Example 03 is on path
    ddp_utils = None  # type: ignore


def _load_or_init(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"config": {}, "epochs": [], "test": {}}


def append_metrics_json(path: Path, record: Dict[str, Any]) -> None:
    """Merge ``record`` into a metrics JSON blob.

    The backing file maintains the structure ``{"config": {...}, "epochs": [...],
    "test": {...}}`` to match the original scripts.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_or_init(path)

    if "config" in record:
        data["config"].update(record["config"])

    if record.keys() - {"config", "test"} and "epoch" in record:
        data["epochs"].append({k: v for k, v in record.items() if k != "config"})
    elif "epoch" in record:
        data["epochs"].append(record)

    if "test" in record:
        data["test"].update(record["test"])

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def append_metrics_jsonl(
    path: Path,
    record: Dict[str, Any],
    *,
    master_only: bool = False,
    is_master_fn: Optional[callable] = None,
) -> None:
    """Append ``record`` as a JSON line to ``path``.

    Args:
        path: Destination JSONL file.
        record: Metrics payload to serialise.
        master_only: Skip writes on non-master ranks when ``True``.
        is_master_fn: Optional callback returning ``True`` when the caller is the
            designated writer. Defaults to ``ddp_utils.is_master`` if available.
    """

    if master_only:
        predicate = is_master_fn or (getattr(ddp_utils, "is_master", None) if ddp_utils else None)
        if predicate is not None and not predicate():
            return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
