"""Scoped CUDA-to-CPU compatibility layer for checkpoint-system experiments.

Original checkpoint code keeps running; only accelerator-only primitives are
substituted. CUDA allocations become tagged CPU device-surrogate allocations,
GPU/host copies remain real copies between distinct buffers, NCCL process-group
initialization becomes Gloo, and exact-size measured timing targets can add
residual delay. Distributed send/recv/collective calls are never mocked.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import torch

CPU_COMPAT_ENV = "CPU_GPU_COMPAT"


def _is_cuda_device(device: Any) -> bool:
    if device is None:
        return False
    if isinstance(device, int):
        return True
    text = str(device)
    return text == "cuda" or text.startswith("cuda:")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


@dataclass
class TimingProfile:
    """Exact-size accelerator timing targets; no interpolation is performed."""
    source: str = "uncalibrated"
    targets: Dict[Tuple[str, int], float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Optional[str]) -> "TimingProfile":
        if not path:
            return cls()
        payload = json.loads(Path(path).read_text())
        targets: Dict[Tuple[str, int], float] = {}
        for kind, rows in payload.get("transfers", {}).items():
            for row in rows:
                targets[(str(kind), int(row["bytes"]))] = float(row["target_s"])
        return cls(source=str(payload.get("source", "unknown")), targets=targets)

    def target(self, kind: str, nbytes: int) -> Optional[float]:
        return self.targets.get((kind, int(nbytes)))


class FakeCudaEvent:
    def __init__(self, *args, **kwargs):
        self._t = None

    def record(self, stream=None):
        self._t = time.perf_counter()
        return self

    def synchronize(self):
        return None

    def wait(self, stream=None):
        return None

    def elapsed_time(self, end_event):
        if self._t is None or end_event._t is None:
            raise RuntimeError("elapsed_time() requires both events to be recorded")
        return (end_event._t - self._t) * 1000.0

    def query(self):
        return self._t is not None


class FakeCudaStream:
    """Synchronous stream API surrogate. Ordering is preserved; overlap is not."""
    _counter = 0
    _lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        with self._lock:
            type(self)._counter += 1
            self.stream_id = type(self)._counter

    def synchronize(self):
        return None

    def wait_stream(self, stream):
        return None

    def wait_event(self, event):
        if hasattr(event, "wait"):
            event.wait(self)
        return None

    def record_event(self, event=None):
        event = event or FakeCudaEvent()
        event.record(self)
        return event


class _StreamContext:
    def __init__(self, runtime, stream):
        self.runtime = runtime
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = getattr(self.runtime._tls, "stream", None)
        self.runtime._tls.stream = self.stream
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        self.runtime._tls.stream = self.previous
        return False


class FakeGradScaler:
    def __init__(self, *args, **kwargs):
        self.enabled = kwargs.get("enabled", True)

    def scale(self, loss): return loss
    def step(self, optimizer, *args, **kwargs): return optimizer.step(*args, **kwargs)
    def update(self, *args, **kwargs): return None
    def unscale_(self, optimizer): return None
    def state_dict(self): return {}
    def load_state_dict(self, state_dict): return None


class CpuGpuCompat:
    """Context manager that makes CUDA-oriented checkpoint code CPU-runnable."""
    def __init__(self, *, timing_profile=None, strict_timing=False, trace_path=None):
        self.profile = timing_profile or TimingProfile()
        self.strict_timing = bool(strict_timing)
        self.trace_path = trace_path
        self._device_storages = set()
        self._lock = threading.RLock()
        self._tls = threading.local()
        self.default_stream = FakeCudaStream()
        self._stack = None
        self._trace_fh = None

    def _mark_device(self, tensor):
        with self._lock:
            self._device_storages.add(_storage_ptr(tensor))
        return tensor

    def is_device_surrogate(self, tensor):
        if not torch.is_tensor(tensor):
            return False
        with self._lock:
            return _storage_ptr(tensor) in self._device_storages

    def current_stream(self, device=None):
        return getattr(self._tls, "stream", None) or self.default_stream

    def _trace(self, record):
        if self._trace_fh is None:
            return
        row = {"schema": "cpu_gpu_compat.v1", "t_wall": time.time(),
               "thread": threading.get_ident(), **record}
        self._trace_fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._trace_fh.flush()

    def _finish_transfer(self, kind, nbytes, op_s):
        target = self.profile.target(kind, nbytes)
        if target is None:
            if self.strict_timing:
                raise RuntimeError(
                    f"No measured {kind} target for exactly {nbytes} bytes "
                    f"(profile source={self.profile.source!r})")
            target = 0.0
        residual = max(0.0, target - op_s)
        if residual:
            time.sleep(residual)
        self._trace({"event": "transfer", "kind": kind, "bytes": int(nbytes),
                     "operation_s": op_s, "target_s": target,
                     "residual_wait_s": residual, "actual_s": max(op_s, target),
                     "timing_source": self.profile.source,
                     "stream": getattr(self.current_stream(), "stream_id", None)})

    @staticmethod
    def _rewrite_to_cpu(args, kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        if _is_cuda_device(kwargs.get("device")):
            kwargs["device"] = "cpu"
        for i, arg in enumerate(args):
            if isinstance(arg, (torch.dtype, torch.Tensor)):
                continue
            if _is_cuda_device(arg):
                args[i] = "cpu"
                break
        if kwargs.get("pin_memory"):
            kwargs["pin_memory"] = False
        return tuple(args), kwargs

    def _factory_wrapper(self, original):
        def wrapped(*args, **kwargs):
            explicit_device = kwargs.get("device") if "device" in kwargs else None
            requested_cuda = _is_cuda_device(explicit_device)
            inherited_fake = ("device" not in kwargs and bool(args)
                              and torch.is_tensor(args[0])
                              and self.is_device_surrogate(args[0]))
            new_kwargs = dict(kwargs)
            if requested_cuda:
                new_kwargs["device"] = "cpu"
            if new_kwargs.get("pin_memory"):
                new_kwargs["pin_memory"] = False
            out = original(*args, **new_kwargs)
            if (requested_cuda or inherited_fake) and torch.is_tensor(out):
                self._mark_device(out)
            return out
        return wrapped

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("CpuGpuCompat contexts are not re-entrant")
        self._stack = ExitStack()
        stack = self._stack
        if self.trace_path:
            Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(self.trace_path, "w", buffering=1)
        stack.enter_context(mock.patch.dict(os.environ, {CPU_COMPAT_ENV: "1"}))

        tensor_to = torch.Tensor.to
        tensor_cpu = torch.Tensor.cpu
        tensor_copy = torch.Tensor.copy_
        module_to = torch.nn.Module.to
        torch_load = torch.load

        def fake_tensor_cuda(tensor, *args, **kwargs):
            t0 = time.perf_counter()
            out = tensor_to(tensor, device="cpu", copy=True)
            self._mark_device(out)
            self._finish_transfer("h2d", _tensor_nbytes(out), time.perf_counter() - t0)
            return out

        def fake_tensor_to(tensor, *args, **kwargs):
            requested_cuda = _is_cuda_device(kwargs.get("device")) or any(
                _is_cuda_device(a) for a in args if not isinstance(a, (torch.dtype, torch.Tensor)))
            if not requested_cuda:
                return tensor_to(tensor, *args, **kwargs)
            new_args, new_kwargs = self._rewrite_to_cpu(args, kwargs)
            new_kwargs["copy"] = True
            t0 = time.perf_counter()
            out = tensor_to(tensor, *new_args, **new_kwargs)
            self._mark_device(out)
            self._finish_transfer("h2d", _tensor_nbytes(out), time.perf_counter() - t0)
            return out

        def fake_tensor_cpu(tensor, *args, **kwargs):
            if not self.is_device_surrogate(tensor):
                return tensor_cpu(tensor, *args, **kwargs)
            t0 = time.perf_counter()
            out = tensor_to(tensor, device="cpu", copy=True)
            self._finish_transfer("d2h", _tensor_nbytes(tensor), time.perf_counter() - t0)
            return out

        def fake_tensor_copy(dst, src, *args, **kwargs):
            src_fake = torch.is_tensor(src) and self.is_device_surrogate(src)
            dst_fake = self.is_device_surrogate(dst)
            t0 = time.perf_counter()
            out = tensor_copy(dst, src, *args, **kwargs)
            op_s = time.perf_counter() - t0
            if src_fake and not dst_fake:
                self._finish_transfer("d2h", _tensor_nbytes(dst), op_s)
            elif dst_fake and not src_fake:
                self._finish_transfer("h2d", _tensor_nbytes(dst), op_s)
            elif src_fake and dst_fake:
                self._finish_transfer("d2d", _tensor_nbytes(dst), op_s)
            return out

        def fake_module_cuda(module, *args, **kwargs):
            def move(t):
                out = tensor_to(t, device="cpu", copy=True)
                self._mark_device(out)
                return out
            return module._apply(move)

        def fake_module_to(module, *args, **kwargs):
            requested_cuda = _is_cuda_device(kwargs.get("device")) or any(
                _is_cuda_device(a) for a in args if not isinstance(a, (torch.dtype, torch.Tensor)))
            if not requested_cuda:
                return module_to(module, *args, **kwargs)
            new_args, new_kwargs = self._rewrite_to_cpu(args, kwargs)
            out = module_to(module, *new_args, **new_kwargs)
            for p in module.parameters(recurse=True): self._mark_device(p)
            for b in module.buffers(recurse=True): self._mark_device(b)
            return out

        def fake_pin_memory(tensor, device=None):
            return tensor.clone(memory_format=torch.preserve_format)

        def fake_torch_load(*args, **kwargs):
            if _is_cuda_device(kwargs.get("map_location")):
                kwargs = dict(kwargs); kwargs["map_location"] = "cpu"
            return torch_load(*args, **kwargs)

        stack.enter_context(mock.patch.object(torch.Tensor, "cuda", fake_tensor_cuda))
        stack.enter_context(mock.patch.object(torch.Tensor, "to", fake_tensor_to))
        stack.enter_context(mock.patch.object(torch.Tensor, "cpu", fake_tensor_cpu))
        stack.enter_context(mock.patch.object(torch.Tensor, "copy_", fake_tensor_copy))
        stack.enter_context(mock.patch.object(torch.Tensor, "pin_memory", fake_pin_memory))
        stack.enter_context(mock.patch.object(torch.nn.Module, "cuda", fake_module_cuda))
        stack.enter_context(mock.patch.object(torch.nn.Module, "to", fake_module_to))
        stack.enter_context(mock.patch.object(torch, "load", fake_torch_load))

        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like",
                     "full", "full_like", "rand", "rand_like", "randn", "randn_like",
                     "randint", "arange", "tensor", "as_tensor"):
            if hasattr(torch, name):
                stack.enter_context(mock.patch.object(torch, name, self._factory_wrapper(getattr(torch, name))))

        stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(mock.patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
        stack.enter_context(mock.patch.object(torch.cuda, "set_device", return_value=None))
        stack.enter_context(mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None))
        stack.enter_context(mock.patch.object(torch.cuda, "empty_cache", return_value=None))
        stack.enter_context(mock.patch.object(torch.cuda, "manual_seed_all", return_value=None))
        stack.enter_context(mock.patch.object(torch.cuda, "Stream", FakeCudaStream))
        stack.enter_context(mock.patch.object(torch.cuda, "Event", FakeCudaEvent))
        stack.enter_context(mock.patch.object(torch.cuda, "current_stream", self.current_stream))
        stack.enter_context(mock.patch.object(torch.cuda, "stream", lambda s: _StreamContext(self, s)))
        if hasattr(torch.cuda, "default_stream"):
            stack.enter_context(mock.patch.object(torch.cuda, "default_stream", lambda *a, **k: self.default_stream))
        if hasattr(torch.cuda, "IntTensor"):
            stack.enter_context(mock.patch.object(torch.cuda, "IntTensor", torch.IntTensor))
        if hasattr(torch.cuda, "amp"):
            stack.enter_context(mock.patch.object(torch.cuda.amp, "autocast", lambda *a, **k: nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda.amp, "GradScaler", FakeGradScaler))
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            stack.enter_context(mock.patch.object(torch.backends.cuda, "sdp_kernel", lambda *a, **k: nullcontext()))

        import torch.distributed as dist
        original_init = dist.init_process_group
        def cpu_init_process_group(*args, **kwargs):
            args = list(args)
            if args and str(args[0]).lower() == "nccl": args[0] = "gloo"
            if str(kwargs.get("backend", "")).lower() == "nccl":
                kwargs = dict(kwargs); kwargs["backend"] = "gloo"
            return original_init(*args, **kwargs)
        stack.enter_context(mock.patch.object(dist, "init_process_group", cpu_init_process_group))
        self._trace({"event": "compat_enter", "timing_source": self.profile.source,
                     "strict_timing": self.strict_timing})
        return self

    def __exit__(self, exc_type, exc, tb):
        self._trace({"event": "compat_exit", "exception": repr(exc) if exc else None})
        try:
            if self._stack is not None: self._stack.close()
        finally:
            self._stack = None
            if self._trace_fh is not None:
                self._trace_fh.close(); self._trace_fh = None
        return False


@contextmanager
def cpu_gpu_compat(*, profile_path=None, strict_timing=False, trace_path=None):
    runtime = CpuGpuCompat(timing_profile=TimingProfile.from_json(profile_path),
                           strict_timing=strict_timing, trace_path=trace_path)
    with runtime:
        yield runtime
