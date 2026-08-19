"""Common contract tests; this file is identical in all three repositories."""
from __future__ import annotations

import time

import torch

from cpu.function_overrides import TimedFunctionOverrides
from cpu.function_timing import FunctionTimingProfile, GpuFunctionTiming
from cpu.gpu_compat import CpuGpuCompat


def _storage_ptr(tensor):
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


def test_cuda_redirect_preserves_separate_storage_and_copy():
    values = torch.arange(1024, dtype=torch.float32)
    with CpuGpuCompat() as runtime:
        device_surrogate = values.cuda()
        host_checkpoint = torch.empty(1024, dtype=torch.float32, device="cpu")

        assert runtime.is_device_surrogate(device_surrogate)
        assert not runtime.is_device_surrogate(host_checkpoint)
        assert _storage_ptr(device_surrogate) != _storage_ptr(host_checkpoint)

        host_checkpoint.copy_(device_surrogate)
        assert torch.equal(host_checkpoint, values)


def test_missing_function_timing_is_an_error():
    timer = GpuFunctionTiming(
        FunctionTimingProfile(source="deliberately-empty"), strict=True
    )
    try:
        timer.run("forward", lambda: None)
    except RuntimeError as exc:
        assert "forward" in str(exc)
    else:
        raise AssertionError("missing measured GPU timing must fail in strict mode")


def test_function_timing_never_shortens_cpu_operation():
    profile = FunctionTimingProfile(
        source="synthetic-contract-test", functions={"phase": 0.001}
    )
    timer = GpuFunctionTiming(profile, strict=True)

    def slower_operation():
        time.sleep(0.003)

    start = time.perf_counter()
    timer.run("phase", slower_operation)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.003


def test_scoped_override_executes_original_callable_and_restores_it():
    class Workload:
        def step(self, value):
            return value + 1

    workload = Workload()
    original = workload.step
    timer = GpuFunctionTiming(
        FunctionTimingProfile(
            source="synthetic-contract-test", functions={"optimizer_step": 0.001}
        ),
        strict=True,
    )

    with TimedFunctionOverrides(timer) as overrides:
        overrides.patch(workload, "step", timing_name="optimizer_step")
        assert workload.step(4) == 5

    assert workload.step(4) == 5
    assert workload.step.__func__ is original.__func__
