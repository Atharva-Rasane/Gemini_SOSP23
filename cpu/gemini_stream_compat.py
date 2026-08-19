"""Gemini-only extension for CUDA copy-stream behavior on CPU VMs.

Enter this context *inside* CpuGpuCompat and import snapshot_comm afterwards.
Only Tensor.copy_(..., non_blocking=True) issued while a non-default fake CUDA
stream is current is scheduled asynchronously. Blocking copies remain blocking.
This preserves the control-flow distinction Gemini relies on without changing
DeepSpeed's SnapshotOptimizer code.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch

from cpu.gpu_compat import FakeCudaEvent, _StreamContext


class GeminiAsyncStream:
    _counter = 1000
    _counter_lock = threading.Lock()

    def __init__(self, scheduler, *args, **kwargs):
        del args, kwargs
        self.scheduler = scheduler
        with self._counter_lock:
            type(self)._counter += 1
            self.stream_id = type(self)._counter

    def synchronize(self):
        self.scheduler.synchronize_stream(self)

    def wait_stream(self, stream):
        stream.synchronize()

    def wait_event(self, event):
        if hasattr(event, "wait"):
            event.wait(self)

    def record_event(self, event=None):
        self.synchronize()
        event = event or FakeCudaEvent()
        event.record(self)
        return event


class GeminiAsyncCopyCompat:
    """Schedule Gemini's nonblocking D2H copies off the caller thread."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini-copy")
        self._futures = {}
        self._lock = threading.Lock()
        self._stack = None

    def _register(self, stream, future):
        with self._lock:
            self._futures.setdefault(id(stream), []).append(future)

    def _drain(self, stream):
        with self._lock:
            futures = self._futures.pop(id(stream), [])
        for future in futures:
            future.result()

    def synchronize_stream(self, stream):
        self._drain(stream)

    def synchronize_all(self, *args, **kwargs):
        del args, kwargs
        while True:
            with self._lock:
                streams = list(self._futures.keys())
                pending = [f for values in self._futures.values() for f in values]
                self._futures.clear()
            if not pending:
                return
            for future in pending:
                future.result()
            if not streams:
                return

    def __enter__(self):
        stack = self._stack = __import__("contextlib").ExitStack()
        patched_copy = torch.Tensor.copy_

        def stream_factory(*args, **kwargs):
            return GeminiAsyncStream(self, *args, **kwargs)

        def async_aware_copy(dst, src, *args, **kwargs):
            non_blocking = bool(kwargs.get("non_blocking", False))
            stream = self.runtime.current_stream()
            is_async_stream = isinstance(stream, GeminiAsyncStream)
            if not (non_blocking and is_async_stream):
                return patched_copy(dst, src, *args, **kwargs)

            def perform():
                previous = getattr(self.runtime._tls, "stream", None)
                self.runtime._tls.stream = stream
                try:
                    patched_copy(dst, src, *args, **kwargs)
                finally:
                    self.runtime._tls.stream = previous

            future = self._executor.submit(perform)
            self._register(stream, future)
            return dst

        stack.enter_context(mock.patch.object(torch.Tensor, "copy_", async_aware_copy))
        stack.enter_context(mock.patch.object(torch.cuda, "Stream", stream_factory))
        stack.enter_context(mock.patch.object(torch.cuda, "synchronize", self.synchronize_all))
        stack.enter_context(
            mock.patch.object(torch.cuda, "stream", lambda s: _StreamContext(self.runtime, s))
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.synchronize_all()
        finally:
            if self._stack is not None:
                self._stack.close()
                self._stack = None
            self._executor.shutdown(wait=True)
        return False
