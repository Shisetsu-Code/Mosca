from __future__ import annotations

import platform
import sys

TARGET_PYTHON = (3, 12, 14)


def runtime_status() -> dict[str, object]:
    current = sys.version_info[:3]
    return {
        "target": ".".join(map(str, TARGET_PYTHON)),
        "current": platform.python_version(),
        "exact": current == TARGET_PYTHON,
        "compatible_minor": current[:2] == TARGET_PYTHON[:2],
        "implementation": platform.python_implementation(),
    }


def require_target(*, exact: bool = False) -> None:
    status = runtime_status()
    ok = status["exact"] if exact else status["compatible_minor"]
    if not ok:
        mode = "exactly" if exact else "the 3.12 series required by"
        raise RuntimeError(
            f"Mosca targets CPython {status['target']}; running {status['current']}. "
            f"Use {mode} the pinned runtime (Dockerfile is provided)."
        )
