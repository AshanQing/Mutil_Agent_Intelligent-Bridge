from __future__ import annotations

import ctypes
import sys
from typing import Any


def enable_windows_high_dpi(
    *,
    platform_name: str | None = None,
    windll: Any = None,
) -> str:
    """Declare DPI awareness before the first Tk window is created.

    Per-Monitor V2 keeps text and controls rendered at the monitor's native
    resolution. Older Windows APIs remain as deterministic fallbacks.
    """
    platform_name = platform_name or sys.platform
    if platform_name != "win32":
        return "not_windows"
    windll = windll or ctypes.windll

    try:
        if windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per_monitor_v2"
    except (AttributeError, OSError):
        pass
    try:
        if windll.shcore.SetProcessDpiAwareness(2) == 0:
            return "per_monitor"
    except (AttributeError, OSError):
        pass
    try:
        if windll.user32.SetProcessDPIAware():
            return "system"
    except (AttributeError, OSError):
        pass
    return "unavailable"

