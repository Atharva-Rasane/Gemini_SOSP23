"""Scoped monkeypatch helper for timing original functions without editing them."""
from __future__ import annotations

from contextlib import ExitStack
from functools import wraps
from unittest import mock

from cpu.function_timing import GpuFunctionTiming


class TimedFunctionOverrides:
    """Temporarily wrap existing callables with measured GPU timing.

    The original callable still runs. Only its externally visible duration is
    adjusted by GpuFunctionTiming. This is intended for unit-test/CPU harnesses
    that must leave the original training/checkpoint source untouched.
    """

    def __init__(self, timer: GpuFunctionTiming):
        self.timer = timer
        self._stack = None

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("TimedFunctionOverrides is not re-entrant")
        self._stack = ExitStack()
        return self

    def patch(self, target, attribute: str, *, timing_name: str | None = None):
        if self._stack is None:
            raise RuntimeError("enter TimedFunctionOverrides before patching")
        original = getattr(target, attribute)
        if not callable(original):
            raise TypeError(f"{target!r}.{attribute} is not callable")
        name = timing_name or attribute

        @wraps(original)
        def wrapped(*args, **kwargs):
            return self.timer.run(name, original, *args, **kwargs)

        self._stack.enter_context(mock.patch.object(target, attribute, wrapped))
        return original

    def __exit__(self, exc_type, exc, tb):
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        return False
