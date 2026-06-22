"""
Optional fake CUDA support for example launch scripts.

Python imports this file automatically when the repo root is on PYTHONPATH.  The
hook is intentionally inert unless --fakegpu is present in this process or an
ancestor launch command, or PCCHECK_FAKEGPU=1 is set.

Fake mode avoids expensive model math by returning shape-correct garbage tensors,
but keeps DeepSpeed/distributed/checkpoint orchestration on the real code paths.
"""

import os
import sys
import importlib
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock


FAKEGPU_ARG = "--fakegpu"
FAKEGPU_ENV = "PCCHECK_FAKEGPU"
FAKEGPU_EXPORT_ENV = "PYTHONPCCHECK_FAKEGPU"

_FAKEGPU_STACK = None


def _strip_fakegpu_arg():
    found = False
    cleaned = []
    for arg in sys.argv:
        if arg == FAKEGPU_ARG:
            found = True
            continue
        cleaned.append(arg)

    if found:
        sys.argv[:] = cleaned
    return found


def _parent_cmdlines():
    try:
        import psutil

        process = psutil.Process()
        for parent in process.parents():
            try:
                yield parent.cmdline()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return
    except Exception:
        pass

    if os.name != "posix":
        return

    pid = os.getppid()
    seen = set()
    while pid and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                raw_cmdline = handle.read()
            if raw_cmdline:
                yield [
                    part.decode(errors="replace")
                    for part in raw_cmdline.rstrip(b"\0").split(b"\0")
                ]

            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
                stat = handle.read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:
            break


def _ancestor_requested_fakegpu():
    for cmdline in _parent_cmdlines() or ():
        if FAKEGPU_ARG in cmdline:
            return True
    return False


def _fakegpu_requested():
    requested = _strip_fakegpu_arg()
    requested = requested or os.environ.get(FAKEGPU_ENV) == "1"
    requested = requested or os.environ.get(FAKEGPU_EXPORT_ENV) == "1"
    requested = requested or _ancestor_requested_fakegpu()
    if requested:
        os.environ[FAKEGPU_ENV] = "1"
        os.environ[FAKEGPU_EXPORT_ENV] = "1"
    return requested


def _install_fakegpu():
    global _FAKEGPU_STACK
    if _FAKEGPU_STACK is not None:
        return

    import torch

    stack = ExitStack()

    tensor_cuda = torch.Tensor.cuda
    tensor_to = torch.Tensor.to
    tensor_type = torch.Tensor.type
    module_cuda = torch.nn.Module.cuda
    module_to = torch.nn.Module.to
    torch_load = torch.load

    orig_empty = torch.empty
    orig_empty_like = torch.empty_like
    dist_module = torch.distributed if hasattr(torch, "distributed") else None
    orig_dist_init_process_group = (
        dist_module.init_process_group if dist_module is not None else None
    )
    orig_dist_all_gather = dist_module.all_gather if dist_module is not None else None
    orig_dist_all_reduce = dist_module.all_reduce if dist_module is not None else None
    orig_dist_get_rank = dist_module.get_rank if dist_module is not None else None
    orig_dist_get_world_size = (
        dist_module.get_world_size if dist_module is not None else None
    )

    class FakeWork:
        def wait(self, *args, **kwargs):
            return True

        def is_completed(self):
            return True

        def is_success(self):
            return True

        def exception(self):
            return None

        def get_future(self):
            return None

    class FakeCudaStream:
        def __init__(self, *args, **kwargs):
            pass

        def wait_stream(self, stream):
            return None

        def wait_event(self, event):
            return None

        def record_event(self, event=None):
            return event if event is not None else FakeCudaEvent()

        def synchronize(self):
            return None

        def query(self):
            return True

    class FakeCudaEvent:
        def __init__(self, *args, **kwargs):
            self._time = None

        def record(self, stream=None):
            import time

            self._time = time.time()
            return None

        def wait(self, stream=None):
            return None

        def query(self):
            return True

        def synchronize(self):
            return None

        def elapsed_time(self, end_event):
            if self._time is None or getattr(end_event, "_time", None) is None:
                return 0.0
            return max(0.0, (end_event._time - self._time) * 1000.0)

    class FakeGradScaler:
        def __init__(self, *args, **kwargs):
            self.enabled = kwargs.get("enabled", True)

        def scale(self, loss):
            return loss

        def step(self, optimizer, *args, **kwargs):
            return optimizer.step(*args, **kwargs)

        def update(self, *args, **kwargs):
            return None

        def unscale_(self, optimizer):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return None

    def _identity_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(fn):
            return fn

        return decorate

    def _rank():
        if (
            dist_module is not None
            and dist_module.is_available()
            and dist_module.is_initialized()
        ):
            return orig_dist_get_rank()
        return int(os.environ.get("RANK", "0"))

    def _world_size():
        if (
            dist_module is not None
            and dist_module.is_available()
            and dist_module.is_initialized()
        ):
            return orig_dist_get_world_size()
        return int(os.environ.get("WORLD_SIZE", "1"))

    def _is_fake_device(device):
        if device is None:
            return False
        if isinstance(device, int):
            return True
        if isinstance(device, torch.device):
            return device.type == "cuda"
        text = str(device)
        return text == "cuda" or text.startswith("cuda:")

    def _extract_to_dtype(args, kwargs, fallback):
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            return kwargs["dtype"]
        for arg in args:
            if isinstance(arg, torch.dtype):
                return arg
            if isinstance(arg, torch.Tensor):
                return arg.dtype
        return fallback

    def _to_requests_fake_device(args, kwargs):
        if _is_fake_device(kwargs.get("device")):
            return True
        for arg in args:
            if isinstance(arg, torch.dtype):
                continue
            if isinstance(arg, torch.Tensor):
                return _is_fake_device(arg.device)
            if _is_fake_device(arg):
                return True
        return False

    def _empty_like_garbage(tensor, dtype=None):
        kwargs = {
            "dtype": dtype or tensor.dtype,
            "device": "cpu",
            "requires_grad": tensor.requires_grad,
        }
        try:
            return orig_empty_like(tensor, memory_format=torch.preserve_format, **kwargs)
        except TypeError:
            return orig_empty_like(tensor, **kwargs)

    def _fake_device_tensor(tensor, dtype=None):
        return _empty_like_garbage(tensor, dtype or tensor.dtype)

    def _clean_factory_kwargs(kwargs):
        cleaned = dict(kwargs)
        if _is_fake_device(cleaned.get("device")):
            cleaned["device"] = "cpu"
        if cleaned.get("pin_memory"):
            cleaned["pin_memory"] = False
        return cleaned

    def _factory_requests_fake_device(args, kwargs):
        return _is_fake_device(kwargs.get("device"))

    def _factory_wrapper(original):
        def wrapped(*args, **kwargs):
            fake_device = _factory_requests_fake_device(args, kwargs)
            result = original(*args, **_clean_factory_kwargs(kwargs))
            if fake_device and torch.is_tensor(result):
                return _empty_like_garbage(result)
            return result

        return wrapped

    def _fake_tensor_cuda(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        return _fake_device_tensor(self, dtype=dtype)

    def _fake_tensor_to(self, *args, **kwargs):
        if _to_requests_fake_device(args, kwargs):
            return _fake_device_tensor(
                self,
                dtype=_extract_to_dtype(args, kwargs, self.dtype),
            )
        return tensor_to(self, *args, **kwargs)

    def _fake_tensor_type(self, *args, **kwargs):
        if not args and not kwargs:
            type_name = tensor_type(self)
            return {
                "torch.HalfTensor": "torch.cuda.HalfTensor",
                "torch.FloatTensor": "torch.cuda.FloatTensor",
                "torch.DoubleTensor": "torch.cuda.DoubleTensor",
                "torch.BFloat16Tensor": "torch.cuda.BFloat16Tensor",
                "torch.ByteTensor": "torch.cuda.ByteTensor",
                "torch.CharTensor": "torch.cuda.CharTensor",
                "torch.ShortTensor": "torch.cuda.ShortTensor",
                "torch.IntTensor": "torch.cuda.IntTensor",
                "torch.LongTensor": "torch.cuda.LongTensor",
                "torch.BoolTensor": "torch.cuda.BoolTensor",
            }.get(type_name, type_name)

        if args and isinstance(args[0], str) and args[0].startswith("torch.cuda."):
            dtype_name = args[0].split("torch.cuda.", 1)[1]
            dtype = {
                "HalfTensor": torch.float16,
                "FloatTensor": torch.float32,
                "DoubleTensor": torch.float64,
                "BFloat16Tensor": torch.bfloat16,
                "ByteTensor": torch.uint8,
                "CharTensor": torch.int8,
                "ShortTensor": torch.int16,
                "IntTensor": torch.int32,
                "LongTensor": torch.int64,
                "BoolTensor": torch.bool,
            }.get(dtype_name)
            if dtype is not None:
                return _fake_device_tensor(self, dtype=dtype)
            args = ("torch." + dtype_name,) + args[1:]
        return tensor_type(self, *args, **kwargs)

    def _fake_tensor_record_stream(self, stream):
        return None

    def _fake_tensor_pin_memory(self, *args, **kwargs):
        return self

    def _fake_module_cuda(self, *args, **kwargs):
        return self._apply(lambda tensor: _fake_device_tensor(tensor))

    def _fake_module_to(self, *args, **kwargs):
        if _to_requests_fake_device(args, kwargs):
            dtype = _extract_to_dtype(args, kwargs, None)
            return self._apply(lambda tensor: _fake_device_tensor(tensor, dtype=dtype))
        return module_to(self, *args, **kwargs)

    def _fake_torch_load(*args, **kwargs):
        map_location = kwargs.get("map_location")
        if _is_fake_device(map_location):
            kwargs = dict(kwargs)
            kwargs["map_location"] = "cpu"
        return torch_load(*args, **kwargs)

    def _copy_tensor(dst, src):
        if not torch.is_tensor(dst) or not torch.is_tensor(src):
            return
        with torch.no_grad():
            src = src.detach()
            if src.dtype != dst.dtype:
                src = tensor_to(src, dtype=dst.dtype)
            try:
                dst.copy_(src.reshape(dst.shape))
                return
            except Exception:
                pass

            dst_flat = dst.reshape(-1)
            src_flat = src.reshape(-1)
            count = min(dst_flat.numel(), src_flat.numel())
            if count:
                dst_flat.narrow(0, 0, count).copy_(src_flat.narrow(0, 0, count))

    def _maybe_work(async_op):
        return FakeWork() if async_op else None

    class FakeFusedAdamCuda:
        def multi_tensor_adam(self, chunk_size, noop_flag_buffer, tensor_lists, *args):
            # Keep FusedAdam.step() and ZeRO optimizer-state plumbing intact, but
            # replace the CUDA Adam math with in-place garbage updates.
            for tensor_group in tensor_lists[1:]:
                for tensor in tensor_group:
                    if torch.is_tensor(tensor):
                        _copy_tensor(tensor, orig_empty_like(tensor))
            return None

    def _fake_cpp_extension_load(original_load):
        def wrapped(name, *args, **kwargs):
            if name == "fused_adam":
                return FakeFusedAdamCuda()
            return original_load(name, *args, **kwargs)

        return wrapped

    def _fake_import_module(original_import_module):
        def wrapped(name, package=None):
            if name.endswith(".fused_adam_op"):
                return FakeFusedAdamCuda()
            return original_import_module(name, package)

        return wrapped

    def _cpu_dist_backend(backend):
        if backend is None:
            return "gloo"
        text = str(backend).lower()
        if text == "nccl" or text.endswith(".nccl"):
            return "gloo"
        return backend

    def _fake_init_process_group(backend=None, *args, **kwargs):
        if "backend" in kwargs:
            kwargs = dict(kwargs)
            kwargs["backend"] = _cpu_dist_backend(kwargs["backend"])
            return orig_dist_init_process_group(*args, **kwargs)
        return orig_dist_init_process_group(_cpu_dist_backend(backend), *args, **kwargs)

    def _fake_get_global_rank(group, group_rank):
        if hasattr(dist_module, "get_global_rank"):
            return dist_module.get_global_rank(group, group_rank)
        if group is None:
            return group_rank
        return group_rank

    def _real_world_size(group=None):
        if (
            dist_module is not None
            and dist_module.is_available()
            and dist_module.is_initialized()
        ):
            return orig_dist_get_world_size(group=group)
        return _world_size()

    def _real_rank(group=None):
        if (
            dist_module is not None
            and dist_module.is_available()
            and dist_module.is_initialized()
        ):
            return orig_dist_get_rank(group=group)
        return _rank()

    def _fake_all_gather_base(output_tensor, input_tensor, group=None, async_op=False, **kwargs):
        world_size = _real_world_size(group)
        flat_output = output_tensor.reshape(-1)
        flat_input = input_tensor.reshape(-1)
        gathered = [orig_empty_like(flat_input) for _ in range(world_size)]
        orig_dist_all_gather(gathered, flat_input, group=group, async_op=False)
        offset = 0
        for tensor in gathered:
            count = min(tensor.numel(), flat_output.numel() - offset)
            if count > 0:
                _copy_tensor(
                    flat_output.narrow(0, offset, count),
                    tensor.narrow(0, 0, count),
                )
                offset += count
        return _maybe_work(async_op)

    def _fake_reduce_scatter(output, input_list, op=None, group=None, async_op=False, **kwargs):
        world_size = _real_world_size(group)
        rank = _real_rank(group)
        stacked = torch.stack([tensor.reshape(-1) for tensor in input_list])
        if op is None:
            op = dist_module.ReduceOp.SUM
        orig_dist_all_reduce(stacked, op=op, group=group, async_op=False)
        index = max(0, min(rank, world_size - 1, stacked.shape[0] - 1))
        _copy_tensor(output.reshape(-1), stacked[index].narrow(0, 0, output.numel()))
        return _maybe_work(async_op)

    def _fake_reduce_scatter_base(output, input, op=None, group=None, async_op=False, **kwargs):
        world_size = _real_world_size(group)
        rank = _real_rank(group)
        flat_input = input.reshape(-1).clone()
        if op is None:
            op = dist_module.ReduceOp.SUM
        orig_dist_all_reduce(flat_input, op=op, group=group, async_op=False)
        chunk = output.numel()
        start = max(0, min(rank, world_size - 1)) * chunk
        _copy_tensor(output.reshape(-1), flat_input.narrow(0, start, chunk))
        return _maybe_work(async_op)

    def _fake_all_to_all_single(
        output,
        input,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
        **kwargs,
    ):
        world_size = _real_world_size(group)
        rank = _real_rank(group)
        gathered = [orig_empty_like(input) for _ in range(world_size)]
        orig_dist_all_gather(gathered, input, group=group, async_op=False)

        if input_split_sizes is None:
            input_split_sizes = [input.shape[0] // world_size] * world_size
        if output_split_sizes is None:
            output_split_sizes = [output.shape[0] // world_size] * world_size

        output_offset = 0
        for src_rank, src_input in enumerate(gathered):
            input_offset = sum(input_split_sizes[:rank])
            count = min(input_split_sizes[rank], output_split_sizes[src_rank])
            if count > 0:
                _copy_tensor(
                    output.narrow(0, output_offset, count),
                    src_input.narrow(0, input_offset, count),
                )
            output_offset += output_split_sizes[src_rank]
        return _maybe_work(async_op)

    def _fake_cuda_tensor_constructor(dtype):
        def construct(*args, **kwargs):
            kwargs = _clean_factory_kwargs(kwargs)
            kwargs.setdefault("dtype", dtype)
            return _empty_like_garbage(torch.tensor(*args, **kwargs))

        return construct

    def _fake_zero_dependency(*tensors):
        zero = None
        for tensor in tensors:
            if torch.is_tensor(tensor) and tensor.requires_grad and tensor.numel() > 0:
                dep = tensor.reshape(-1).narrow(0, 0, 1).sum() * 0
                zero = dep if zero is None else zero + dep
        return zero

    def _fake_empty_with_dependency(shape, dtype, *deps):
        result = orig_empty(tuple(shape), dtype=dtype, device="cpu")
        zero = _fake_zero_dependency(*deps)
        if zero is not None:
            result = result + zero
        elif result.is_floating_point():
            result.requires_grad_(any(torch.is_tensor(dep) and dep.requires_grad for dep in deps))
        return result

    def _fake_embedding(input, weight, *args, **kwargs):
        return _fake_empty_with_dependency(
            tuple(input.shape) + (weight.shape[-1],),
            weight.dtype,
            weight,
        )

    def _fake_linear(input, weight, bias=None):
        return _fake_empty_with_dependency(
            tuple(input.shape[:-1]) + (weight.shape[0],),
            input.dtype if input.is_floating_point() else weight.dtype,
            input,
            weight,
            bias,
        )

    def _fake_layer_norm(input, normalized_shape, weight=None, bias=None, *args, **kwargs):
        return _fake_empty_with_dependency(input.shape, input.dtype, input, weight, bias)

    def _fake_activation_like(input, *args, **kwargs):
        return _fake_empty_with_dependency(input.shape, input.dtype, input)

    def _fake_dropout(input, *args, **kwargs):
        return _fake_empty_with_dependency(input.shape, input.dtype, input)

    def _fake_cross_entropy(input, target, *args, **kwargs):
        reduction = kwargs.get("reduction", "mean")
        if len(args) >= 5:
            reduction = args[4]
        shape = target.shape if reduction == "none" else ()
        return _fake_empty_with_dependency(shape, input.dtype, input)

    def _fake_memory_stats(*args, **kwargs):
        return {
            "num_alloc_retries": 0,
            "allocated_bytes.all.current": 0,
            "allocated_bytes.all.peak": 0,
            "reserved_bytes.all.current": 0,
            "reserved_bytes.all.peak": 0,
        }

    def _fake_get_rng_state(device=None):
        return torch.random.get_rng_state()

    def _fake_set_rng_state(new_state, device=None):
        return torch.random.set_rng_state(new_state.cpu())

    fake_stream = FakeCudaStream()

    stack.enter_context(mock.patch.dict(os.environ, {FAKEGPU_ENV: "1"}))

    stack.enter_context(mock.patch.object(torch.Tensor, "cuda", _fake_tensor_cuda))
    stack.enter_context(mock.patch.object(torch.Tensor, "to", _fake_tensor_to))
    stack.enter_context(mock.patch.object(torch.Tensor, "type", _fake_tensor_type))
    stack.enter_context(mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)))
    stack.enter_context(mock.patch.object(torch.Tensor, "record_stream", _fake_tensor_record_stream))
    stack.enter_context(mock.patch.object(torch.Tensor, "pin_memory", _fake_tensor_pin_memory))
    stack.enter_context(mock.patch.object(torch.nn.Module, "cuda", _fake_module_cuda))
    stack.enter_context(mock.patch.object(torch.nn.Module, "to", _fake_module_to))
    stack.enter_context(mock.patch.object(torch, "load", _fake_torch_load))
    stack.enter_context(
        mock.patch.object(
            importlib,
            "import_module",
            _fake_import_module(importlib.import_module),
        )
    )

    try:
        import torch.utils.cpp_extension as cpp_extension

        stack.enter_context(
            mock.patch.object(
                cpp_extension,
                "load",
                _fake_cpp_extension_load(cpp_extension.load),
            )
        )
    except Exception:
        pass

    stack.enter_context(mock.patch.object(torch.nn.functional, "embedding", _fake_embedding))
    stack.enter_context(mock.patch.object(torch.nn.functional, "linear", _fake_linear))
    stack.enter_context(mock.patch.object(torch.nn.functional, "layer_norm", _fake_layer_norm))
    stack.enter_context(mock.patch.object(torch.nn.functional, "gelu", _fake_activation_like))
    stack.enter_context(mock.patch.object(torch.nn.functional, "relu", _fake_activation_like))
    stack.enter_context(mock.patch.object(torch.nn.functional, "silu", _fake_activation_like))
    stack.enter_context(mock.patch.object(torch.nn.functional, "softmax", _fake_activation_like))
    stack.enter_context(mock.patch.object(torch.nn.functional, "dropout", _fake_dropout))
    stack.enter_context(mock.patch.object(torch.nn.functional, "cross_entropy", _fake_cross_entropy))

    for name in [
        "empty",
        "empty_like",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
        "arange",
        "tensor",
        "rand",
        "randn",
        "randint",
        "randperm",
        "linspace",
        "logspace",
        "eye",
    ]:
        if hasattr(torch, name):
            stack.enter_context(mock.patch.object(torch, name, _factory_wrapper(getattr(torch, name))))

    stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
    stack.enter_context(mock.patch.object(torch.cuda, "device_count", return_value=1))
    stack.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
    stack.enter_context(mock.patch.object(torch.cuda, "set_device", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "synchronize", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "empty_cache", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "manual_seed", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "manual_seed_all", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "initial_seed", return_value=0))
    stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state", _fake_get_rng_state))
    stack.enter_context(mock.patch.object(torch.cuda, "set_rng_state", _fake_set_rng_state))
    stack.enter_context(mock.patch.object(torch.cuda, "default_generators", (torch.default_generator,)))
    stack.enter_context(mock.patch.object(torch.cuda, "Stream", FakeCudaStream))
    stack.enter_context(mock.patch.object(torch.cuda, "Event", FakeCudaEvent))
    stack.enter_context(mock.patch.object(torch.cuda, "current_stream", return_value=fake_stream))
    stack.enter_context(mock.patch.object(torch.cuda, "default_stream", return_value=fake_stream))
    stack.enter_context(mock.patch.object(torch.cuda, "stream", lambda stream: nullcontext()))
    stack.enter_context(mock.patch.object(torch.cuda, "device", lambda device=None: nullcontext()))
    stack.enter_context(mock.patch.object(torch.cuda, "_lazy_call", lambda fn, *a, **k: fn()))
    stack.enter_context(
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=80 * 1024**3, name="Fake CUDA GPU"),
        )
    )
    stack.enter_context(mock.patch.object(torch.cuda, "get_device_name", return_value="Fake CUDA GPU"))
    stack.enter_context(mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)))
    stack.enter_context(mock.patch.object(torch.cuda, "memory_stats", _fake_memory_stats))

    for name in [
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
        "memory_cached",
        "max_memory_cached",
        "reset_peak_memory_stats",
        "reset_max_memory_allocated",
        "reset_max_memory_cached",
    ]:
        if hasattr(torch.cuda, name):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=0))

    for name, dtype in [
        ("ByteTensor", torch.uint8),
        ("CharTensor", torch.int8),
        ("ShortTensor", torch.int16),
        ("IntTensor", torch.int32),
        ("LongTensor", torch.int64),
        ("HalfTensor", torch.float16),
        ("FloatTensor", torch.float32),
        ("DoubleTensor", torch.float64),
        ("BoolTensor", torch.bool),
        ("BFloat16Tensor", torch.bfloat16),
    ]:
        stack.enter_context(mock.patch.object(torch.cuda, name, _fake_cuda_tensor_constructor(dtype), create=True))

    if hasattr(torch.cuda, "amp"):
        stack.enter_context(mock.patch.object(torch.cuda.amp, "autocast", lambda *a, **k: nullcontext()))
        stack.enter_context(mock.patch.object(torch.cuda.amp, "GradScaler", FakeGradScaler))
        stack.enter_context(mock.patch.object(torch.cuda.amp, "custom_fwd", _identity_decorator))
        stack.enter_context(mock.patch.object(torch.cuda.amp, "custom_bwd", _identity_decorator))

    if hasattr(torch.cuda, "nvtx"):
        stack.enter_context(mock.patch.object(torch.cuda.nvtx, "range", lambda *a, **k: nullcontext(), create=True))

    if hasattr(torch.backends, "cuda"):
        stack.enter_context(
            mock.patch.object(torch.backends.cuda, "sdp_kernel", lambda *a, **k: nullcontext())
        )

    if hasattr(torch, "distributed"):
        dist = torch.distributed
        stack.enter_context(mock.patch.object(dist, "init_process_group", _fake_init_process_group))
        stack.enter_context(mock.patch.object(dist, "_all_gather_base", _fake_all_gather_base, create=True))
        stack.enter_context(mock.patch.object(dist, "reduce_scatter", _fake_reduce_scatter))
        stack.enter_context(mock.patch.object(dist, "_reduce_scatter_base", _fake_reduce_scatter_base, create=True))
        stack.enter_context(mock.patch.object(dist, "all_to_all_single", _fake_all_to_all_single, create=True))
        if not hasattr(dist, "get_global_rank"):
            stack.enter_context(mock.patch.object(dist, "get_global_rank", _fake_get_global_rank, create=True))

        if hasattr(dist, "distributed_c10d"):
            c10d = dist.distributed_c10d
            stack.enter_context(mock.patch.object(c10d, "_all_gather_base", _fake_all_gather_base, create=True))
            if not hasattr(c10d, "_get_global_rank"):
                stack.enter_context(mock.patch.object(c10d, "_get_global_rank", _fake_get_global_rank, create=True))

    _FAKEGPU_STACK = stack

    if os.environ.get("PCCHECK_FAKEGPU_VERBOSE") == "1":
        print("[fakegpu] enabled fake CUDA hooks", file=sys.stderr)


if _fakegpu_requested():
    _install_fakegpu()
