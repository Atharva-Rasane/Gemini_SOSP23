"""Apply measured GPU function durations to CPU-executed surrogate functions.

The wrapper never invents a duration and never shortens a slow CPU operation.
For a named function with measured target T and observed CPU runtime C, the
visible duration is max(C, T). Missing names are errors in strict mode.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class FunctionTimingProfile:
    source: str = "uncalibrated"
    functions: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Optional[str]) -> "FunctionTimingProfile":
        if not path:
            return cls()
        payload = json.loads(Path(path).read_text())
        values = {
            str(name): float(target_s)
            for name, target_s in payload.get("functions", {}).items()
        }
        return cls(source=str(payload.get("source", "unknown")), functions=values)

    def target(self, name: str) -> Optional[float]:
        return self.functions.get(str(name))


class GpuFunctionTiming:
    """Run original CPU-callable functions under exact measured GPU timing."""

    def __init__(self, profile: FunctionTimingProfile, *, strict: bool = True,
                 trace_path: Optional[str] = None):
        self.profile = profile
        self.strict = bool(strict)
        self.trace_path = trace_path
        self._trace = None

    def __enter__(self):
        if self.trace_path:
            path = Path(self.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = path.open("w", buffering=1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._trace is not None:
            self._trace.close()
            self._trace = None
        return False

    def _emit(self, row: Dict[str, Any]) -> None:
        if self._trace is None:
            return
        self._trace.write(json.dumps({
            "schema": "cpu_gpu_function_timing.v1",
            "t_wall": time.time(),
            **row,
        }, sort_keys=True) + "\n")
        self._trace.flush()

    def run(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        target = self.profile.target(name)
        if target is None:
            if self.strict:
                raise RuntimeError(
                    f"No measured GPU duration for function {name!r} "
                    f"(profile source={self.profile.source!r})"
                )
            target = 0.0

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        operation_s = time.perf_counter() - start
        residual_wait_s = max(0.0, target - operation_s)
        if residual_wait_s:
            time.sleep(residual_wait_s)

        self._emit({
            "event": "function",
            "name": str(name),
            "operation_s": operation_s,
            "target_s": target,
            "residual_wait_s": residual_wait_s,
            "actual_s": max(operation_s, target),
            "timing_source": self.profile.source,
        })
        return result


def timed_call(profile_path: str, name: str, fn: Callable[..., Any],
               *args, trace_path: Optional[str] = None, **kwargs) -> Any:
    """Convenience wrapper for one measured function call."""
    profile = FunctionTimingProfile.from_json(profile_path)
    with GpuFunctionTiming(profile, strict=True, trace_path=trace_path) as timer:
        return timer.run(name, fn, *args, **kwargs)
