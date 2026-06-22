"""
Optional fake CUDA support for example launch scripts.

Python imports this file automatically when the repository root is on
PYTHONPATH. The hook is intentionally inert unless ``--fakegpu`` is present in
this process or an ancestor launch command, or one of these variables is set:

    PCCHECK_FAKEGPU=1
    PYTHONPCCHECK_FAKEGPU=1

The fake backend keeps tensors physically on CPU and redirects NCCL process
groups to Gloo. CUDA-facing APIs are spoofed so existing DeepSpeed code can run
without a GPU. Expensive model math is replaced with cheap, shape-correct zero
placeholders that retain an autograd dependency on their inputs.

Important design rule:
    Device transfers, constructors, factories, collectives, ranks, sizes,
    masks, IDs, checkpoint data, and other control tensors preserve values.
    Only expensive numerical kernels may return placeholder values.
"""

import importlib
import os
import sys
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

    # Save original implementations before monkey-patching.
    tensor_cuda = torch.Tensor.cuda
    tensor_to = torch.Tensor.to
    tensor_type = torch.Tensor.type
    module_cuda = torch.nn.Module.cuda
    module_to = torch.nn.Module.to
    torch_load = torch.load

    orig_empty = torch.empty
    orig_empty_like = torch.empty_like
    orig_where = torch.where
    orig_pow = torch.pow
    orig_clamp = torch.clamp

    dist_module = torch.distributed if hasattr(torch, "distributed") else None
    orig_dist_init_process_group = (
        dist_module.init_process_group if dist_module is not None else None
    )
    orig_dist_all_gather = (
        dist_module.all_gather if dist_module is not None else None
    )
    orig_dist_all_reduce = (
        dist_module.all_reduce if dist_module is not None else None
    )
    orig_dist_get_rank = (
        dist_module.get_rank if dist_module is not None else None
    )
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

    def _replace_fake_device_with_cpu(args, kwargs):
        """Rewrite CUDA device arguments to CPU while preserving all values."""
        rewritten_args = list(args)
        rewritten_kwargs = dict(kwargs)

        if rewritten_args and _is_fake_device(rewritten_args[0]):
            rewritten_args[0] = torch.device("cpu")

        if _is_fake_device(rewritten_kwargs.get("device")):
            rewritten_kwargs["device"] = torch.device("cpu")

        return tuple(rewritten_args), rewritten_kwargs

    def _placeholder_like(tensor, dtype=None):
        """Create a deterministic zero placeholder with the same shape."""
        kwargs = {
            "dtype": dtype or tensor.dtype,
            "device": "cpu",
            "requires_grad": False,
        }
        try:
            result = orig_empty_like(
                tensor,
                memory_format=torch.preserve_format,
                **kwargs,
            )
        except TypeError:
            result = orig_empty_like(tensor, **kwargs)
        result.zero_()
        return result

    def _fake_device_tensor(tensor, dtype=None):
        """
        Fake a CUDA transfer without discarding tensor contents.

        The tensor remains physically on CPU. If a dtype conversion was
        requested, use the saved real Tensor.to implementation.
        """
        target_dtype = dtype or tensor.dtype
        if target_dtype != tensor.dtype:
            return tensor_to(tensor, dtype=target_dtype)
        return tensor

    def _clean_factory_kwargs(kwargs):
        cleaned = dict(kwargs)
        if _is_fake_device(cleaned.get("device")):
            cleaned["device"] = "cpu"
        if cleaned.get("pin_memory"):
            cleaned["pin_memory"] = False
        return cleaned

    def _factory_wrapper(original):
        def wrapped(*args, **kwargs):
            # Constructors/factories may produce control tensors. Redirect the
            # allocation to CPU but preserve the factory's actual values.
            return original(*args, **_clean_factory_kwargs(kwargs))

        return wrapped

    def _fake_tensor_cuda(self, *args, **kwargs):
        # Tensor.cuda() does not perform numerical work. Preserve values.
        memory_format = kwargs.get("memory_format", torch.preserve_format)
        try:
            return tensor_to(self, device="cpu", memory_format=memory_format)
        except TypeError:
            return self

    def _fake_tensor_to(self, *args, **kwargs):
        rewritten_args, rewritten_kwargs = _replace_fake_device_with_cpu(args, kwargs)
        return tensor_to(self, *rewritten_args, **rewritten_kwargs)

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
        # Parameters and buffers are already CPU-backed. Do not replace them
        # with uninitialized tensors.
        return self

    def _fake_module_to(self, *args, **kwargs):
        rewritten_args, rewritten_kwargs = _replace_fake_device_with_cpu(args, kwargs)
        return module_to(self, *rewritten_args, **rewritten_kwargs)

    def _fake_torch_load(*args, **kwargs):
        rewritten_args = list(args)
        rewritten_kwargs = dict(kwargs)

        # torch.load(f, map_location, ...)
        if len(rewritten_args) >= 2 and _is_fake_device(rewritten_args[1]):
            rewritten_args[1] = "cpu"

        if _is_fake_device(rewritten_kwargs.get("map_location")):
            rewritten_kwargs["map_location"] = "cpu"

        return torch_load(*rewritten_args, **rewritten_kwargs)

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
            # Keep FusedAdam.step() and ZeRO optimizer-state plumbing intact,
            # but replace CUDA Adam math with cheap deterministic zero updates.
            for tensor_group in tensor_lists[1:]:
                for tensor in tensor_group:
                    if torch.is_tensor(tensor):
                        _copy_tensor(tensor, _placeholder_like(tensor))
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

    def _fake_all_gather_base(
        output_tensor,
        input_tensor,
        group=None,
        async_op=False,
        **kwargs,
    ):
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

    def _fake_reduce_scatter(
        output,
        input_list,
        op=None,
        group=None,
        async_op=False,
        **kwargs,
    ):
        world_size = _real_world_size(group)
        rank = _real_rank(group)
        stacked = torch.stack([tensor.reshape(-1) for tensor in input_list])
        if op is None:
            op = dist_module.ReduceOp.SUM
        orig_dist_all_reduce(stacked, op=op, group=group, async_op=False)
        index = max(0, min(rank, world_size - 1, stacked.shape[0] - 1))
        _copy_tensor(
            output.reshape(-1),
            stacked[index].narrow(0, 0, output.numel()),
        )
        return _maybe_work(async_op)

    def _fake_reduce_scatter_base(
        output,
        input,
        op=None,
        group=None,
        async_op=False,
        **kwargs,
    ):
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

            # Legacy constructors treat no arguments as an empty length-0
            # tensor, and all-integer positional arguments as a requested size.
            if not args:
                return orig_empty((0,), **kwargs)
            if all(isinstance(arg, int) for arg in args):
                return orig_empty(tuple(args), **kwargs)

            # Data-bearing construction must preserve the supplied values.
            return torch.tensor(*args, **kwargs)

        return construct

    def _fake_zero_dependency(*tensors):
        zero = None
        for tensor in tensors:
            if torch.is_tensor(tensor) and tensor.requires_grad and tensor.numel() > 0:
                dep = tensor.reshape(-1).narrow(0, 0, 1).sum() * 0
                zero = dep if zero is None else zero + dep
        return zero

    def _fake_empty_with_dependency(shape, dtype, *deps):
        # Deterministic zero placeholders avoid random NaNs/infinities while
        # retaining shape, dtype, and a minimal autograd dependency.
        result = orig_empty(tuple(shape), dtype=dtype, device="cpu")
        result.zero_()

        zero = _fake_zero_dependency(*deps)
        if zero is not None:
            result = result + zero
        elif result.is_floating_point():
            requires_grad = any(
                torch.is_tensor(dep) and dep.requires_grad for dep in deps
            )
            result.requires_grad_(requires_grad)
        return result

    def _shape_of(value):
        return tuple(value.shape) if torch.is_tensor(value) else ()

    def _broadcast_shape(*shapes):
        shapes = [tuple(shape) for shape in shapes if shape is not None]
        if not shapes:
            return ()
        if hasattr(torch, "broadcast_shapes"):
            return tuple(torch.broadcast_shapes(*shapes))

        result = []
        max_ndim = max(len(shape) for shape in shapes)
        for offset in range(1, max_ndim + 1):
            dim = 1
            for shape in shapes:
                candidate = shape[-offset] if offset <= len(shape) else 1
                if candidate != 1:
                    if dim not in (1, candidate):
                        raise RuntimeError(f"shape mismatch: {shapes}")
                    dim = candidate
            result.append(dim)
        return tuple(reversed(result))

    def _dtype_from_values(*values, fallback=torch.float32):
        for value in values:
            if torch.is_tensor(value):
                return value.dtype
        return fallback

    def _return_with_optional_out(result, out):
        if out is not None:
            _copy_tensor(out, result)
            return out
        return result

    def _fake_addmm(input, mat1, mat2, *args, **kwargs):
        out = kwargs.get("out")
        shape = (mat1.shape[0], mat2.shape[1])
        result = _fake_empty_with_dependency(
            shape,
            _dtype_from_values(input, mat1, mat2),
            input,
            mat1,
            mat2,
        )
        return _return_with_optional_out(result, out)

    def _fake_mm(input, mat2, *args, **kwargs):
        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            (input.shape[0], mat2.shape[1]),
            _dtype_from_values(input, mat2),
            input,
            mat2,
        )
        return _return_with_optional_out(result, out)

    def _matmul_shape(input, other):
        left = tuple(input.shape)
        right = tuple(other.shape)
        left_ndim = len(left)
        right_ndim = len(right)

        if left_ndim == 1 and right_ndim == 1:
            return ()
        if left_ndim == 1:
            return _broadcast_shape((), right[:-2]) + (right[-1],)
        if right_ndim == 1:
            return _broadcast_shape(left[:-2], ()) + (left[-2],)
        return _broadcast_shape(left[:-2], right[:-2]) + (left[-2], right[-1])

    def _fake_matmul(input, other, *args, **kwargs):
        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            _matmul_shape(input, other),
            _dtype_from_values(input, other),
            input,
            other,
        )
        return _return_with_optional_out(result, out)

    def _fake_bmm(input, mat2, *args, **kwargs):
        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            (input.shape[0], input.shape[1], mat2.shape[2]),
            _dtype_from_values(input, mat2),
            input,
            mat2,
        )
        return _return_with_optional_out(result, out)

    def _fake_baddbmm(input, batch1, batch2, *args, **kwargs):
        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            (batch1.shape[0], batch1.shape[1], batch2.shape[2]),
            _dtype_from_values(input, batch1, batch2),
            input,
            batch1,
            batch2,
        )
        return _return_with_optional_out(result, out)

    def _fake_addbmm(input, batch1, batch2, *args, **kwargs):
        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            (batch1.shape[1], batch2.shape[2]),
            _dtype_from_values(input, batch1, batch2),
            input,
            batch1,
            batch2,
        )
        return _return_with_optional_out(result, out)

    def _fake_where(condition, input=None, other=None, *args, **kwargs):
        if input is None and other is None:
            return orig_where(condition)

        out = kwargs.get("out")
        result = _fake_empty_with_dependency(
            _broadcast_shape(
                _shape_of(condition),
                _shape_of(input),
                _shape_of(other),
            ),
            _dtype_from_values(input, other),
            input,
            other,
        )
        return _return_with_optional_out(result, out)

    def _fake_unary_torch_op(original):
        def wrapped(input, *args, **kwargs):
            if not torch.is_tensor(input):
                return original(input, *args, **kwargs)
            result = _fake_empty_with_dependency(input.shape, input.dtype, input)
            return _return_with_optional_out(result, kwargs.get("out"))

        return wrapped

    def _fake_pow(input, exponent, *args, **kwargs):
        if not torch.is_tensor(input) and not torch.is_tensor(exponent):
            return orig_pow(input, exponent, *args, **kwargs)
        shape = _broadcast_shape(_shape_of(input), _shape_of(exponent))
        dtype = _dtype_from_values(input, exponent)
        result = _fake_empty_with_dependency(shape, dtype, input, exponent)
        return _return_with_optional_out(result, kwargs.get("out"))

    def _fake_clamp(input, *args, **kwargs):
        if not torch.is_tensor(input):
            return orig_clamp(input, *args, **kwargs)
        result = _fake_empty_with_dependency(input.shape, input.dtype, input)
        return _return_with_optional_out(result, kwargs.get("out"))

    def _fake_tensor_addmm(self, mat1, mat2, *args, **kwargs):
        return _fake_addmm(self, mat1, mat2, *args, **kwargs)

    def _fake_tensor_mm(self, mat2, *args, **kwargs):
        return _fake_mm(self, mat2, *args, **kwargs)

    def _fake_tensor_matmul(self, other, *args, **kwargs):
        return _fake_matmul(self, other, *args, **kwargs)

    def _fake_tensor_bmm(self, mat2, *args, **kwargs):
        return _fake_bmm(self, mat2, *args, **kwargs)

    def _fake_tensor_baddbmm(self, batch1, batch2, *args, **kwargs):
        return _fake_baddbmm(self, batch1, batch2, *args, **kwargs)

    def _fake_tensor_pow(self, exponent, *args, **kwargs):
        return _fake_pow(self, exponent, *args, **kwargs)

    def _fake_tensor_clamp(self, *args, **kwargs):
        return _fake_clamp(self, *args, **kwargs)

    def _fake_tensor_masked_fill(self, mask, value):
        return _fake_empty_with_dependency(self.shape, self.dtype, self)

    def _fake_tensor_masked_fill_(self, mask, value):
        with torch.no_grad():
            self.zero_()
        return self

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

    def _fake_layer_norm(
        input,
        normalized_shape,
        weight=None,
        bias=None,
        *args,
        **kwargs,
    ):
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
    stack.enter_context(
        mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True))
    )
    stack.enter_context(
        mock.patch.object(torch.Tensor, "record_stream", _fake_tensor_record_stream)
    )
    stack.enter_context(
        mock.patch.object(torch.Tensor, "pin_memory", _fake_tensor_pin_memory)
    )
    stack.enter_context(mock.patch.object(torch.Tensor, "addmm", _fake_tensor_addmm))
    stack.enter_context(mock.patch.object(torch.Tensor, "mm", _fake_tensor_mm))
    stack.enter_context(mock.patch.object(torch.Tensor, "matmul", _fake_tensor_matmul))
    stack.enter_context(mock.patch.object(torch.Tensor, "bmm", _fake_tensor_bmm))
    stack.enter_context(
        mock.patch.object(torch.Tensor, "baddbmm", _fake_tensor_baddbmm)
    )
    stack.enter_context(mock.patch.object(torch.Tensor, "pow", _fake_tensor_pow))
    stack.enter_context(mock.patch.object(torch.Tensor, "clamp", _fake_tensor_clamp))
    stack.enter_context(
        mock.patch.object(torch.Tensor, "masked_fill", _fake_tensor_masked_fill)
    )
    stack.enter_context(
        mock.patch.object(torch.Tensor, "masked_fill_", _fake_tensor_masked_fill_)
    )
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

    stack.enter_context(
        mock.patch.object(torch.nn.functional, "embedding", _fake_embedding)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "linear", _fake_linear)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "layer_norm", _fake_layer_norm)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "gelu", _fake_activation_like)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "relu", _fake_activation_like)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "silu", _fake_activation_like)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "softmax", _fake_activation_like)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "dropout", _fake_dropout)
    )
    stack.enter_context(
        mock.patch.object(torch.nn.functional, "cross_entropy", _fake_cross_entropy)
    )
    for name in ["log_softmax", "softplus", "tanh", "sigmoid"]:
        if hasattr(torch.nn.functional, name):
            stack.enter_context(
                mock.patch.object(torch.nn.functional, name, _fake_activation_like)
            )

    for name, fake in [
        ("addmm", _fake_addmm),
        ("mm", _fake_mm),
        ("matmul", _fake_matmul),
        ("bmm", _fake_bmm),
        ("baddbmm", _fake_baddbmm),
        ("addbmm", _fake_addbmm),
        ("where", _fake_where),
        ("pow", _fake_pow),
        ("clamp", _fake_clamp),
        ("clip", _fake_clamp),
    ]:
        if hasattr(torch, name):
            stack.enter_context(mock.patch.object(torch, name, fake))

    for name in [
        "tanh",
        "sigmoid",
        "exp",
        "erf",
        "sqrt",
        "rsqrt",
        "log",
        "reciprocal",
        "square",
        "abs",
        "neg",
    ]:
        if hasattr(torch, name):
            stack.enter_context(
                mock.patch.object(
                    torch,
                    name,
                    _fake_unary_torch_op(getattr(torch, name)),
                )
            )

    # Preserve factory semantics and values; only redirect CUDA allocations to CPU.
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
            stack.enter_context(
                mock.patch.object(torch, name, _factory_wrapper(getattr(torch, name)))
            )

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
    stack.enter_context(
        mock.patch.object(
            torch.cuda,
            "default_generators",
            (torch.default_generator,),
        )
    )
    stack.enter_context(mock.patch.object(torch.cuda, "Stream", FakeCudaStream))
    stack.enter_context(mock.patch.object(torch.cuda, "Event", FakeCudaEvent))
    stack.enter_context(
        mock.patch.object(torch.cuda, "current_stream", return_value=fake_stream)
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "default_stream", return_value=fake_stream)
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "stream", lambda stream: nullcontext())
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "device", lambda device=None: nullcontext())
    )
    if hasattr(torch.cuda, "device_of"):
        stack.enter_context(
            mock.patch.object(torch.cuda, "device_of", lambda obj: nullcontext())
        )
    stack.enter_context(
        mock.patch.object(torch.cuda, "_lazy_call", lambda fn, *a, **k: fn())
    )
    stack.enter_context(
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(
                total_memory=80 * 1024**3,
                name="Fake CUDA GPU",
                major=8,
                minor=0,
                multi_processor_count=108,
            ),
        )
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "get_device_name", return_value="Fake CUDA GPU")
    )
    stack.enter_context(
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0))
    )
    stack.enter_context(mock.patch.object(torch.cuda, "memory_stats", _fake_memory_stats))

    if hasattr(torch.cuda, "is_initialized"):
        stack.enter_context(
            mock.patch.object(torch.cuda, "is_initialized", return_value=True)
        )
    if hasattr(torch.cuda, "is_bf16_supported"):
        stack.enter_context(
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True)
        )
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=False,
            )
        )
    if hasattr(torch.cuda, "mem_get_info"):
        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(80 * 1024**3, 80 * 1024**3),
            )
        )

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
        stack.enter_context(
            mock.patch.object(
                torch.cuda,
                name,
                _fake_cuda_tensor_constructor(dtype),
                create=True,
            )
        )

    if hasattr(torch.cuda, "amp"):
        stack.enter_context(
            mock.patch.object(
                torch.cuda.amp,
                "autocast",
                lambda *a, **k: nullcontext(),
            )
        )
        stack.enter_context(
            mock.patch.object(torch.cuda.amp, "GradScaler", FakeGradScaler)
        )
        if hasattr(torch.cuda.amp, "custom_fwd"):
            stack.enter_context(
                mock.patch.object(torch.cuda.amp, "custom_fwd", _identity_decorator)
            )
        if hasattr(torch.cuda.amp, "custom_bwd"):
            stack.enter_context(
                mock.patch.object(torch.cuda.amp, "custom_bwd", _identity_decorator)
            )

    if hasattr(torch, "amp"):
        if hasattr(torch.amp, "autocast"):
            stack.enter_context(
                mock.patch.object(
                    torch.amp,
                    "autocast",
                    lambda *a, **k: nullcontext(),
                )
            )
        if hasattr(torch.amp, "GradScaler"):
            stack.enter_context(
                mock.patch.object(torch.amp, "GradScaler", FakeGradScaler)
            )

    if hasattr(torch.cuda, "nvtx"):
        stack.enter_context(
            mock.patch.object(
                torch.cuda.nvtx,
                "range",
                lambda *a, **k: nullcontext(),
                create=True,
            )
        )

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        stack.enter_context(
            mock.patch.object(
                torch.backends.cuda,
                "sdp_kernel",
                lambda *a, **k: nullcontext(),
            )
        )

    if hasattr(torch, "distributed"):
        dist = torch.distributed
        stack.enter_context(
            mock.patch.object(dist, "init_process_group", _fake_init_process_group)
        )
        stack.enter_context(
            mock.patch.object(
                dist,
                "_all_gather_base",
                _fake_all_gather_base,
                create=True,
            )
        )
        if hasattr(dist, "all_gather_into_tensor"):
            stack.enter_context(
                mock.patch.object(
                    dist,
                    "all_gather_into_tensor",
                    _fake_all_gather_base,
                )
            )
        stack.enter_context(
            mock.patch.object(dist, "reduce_scatter", _fake_reduce_scatter)
        )
        stack.enter_context(
            mock.patch.object(
                dist,
                "_reduce_scatter_base",
                _fake_reduce_scatter_base,
                create=True,
            )
        )
        if hasattr(dist, "reduce_scatter_tensor"):
            stack.enter_context(
                mock.patch.object(
                    dist,
                    "reduce_scatter_tensor",
                    _fake_reduce_scatter_base,
                )
            )
        stack.enter_context(
            mock.patch.object(
                dist,
                "all_to_all_single",
                _fake_all_to_all_single,
                create=True,
            )
        )
        if not hasattr(dist, "get_global_rank"):
            stack.enter_context(
                mock.patch.object(
                    dist,
                    "get_global_rank",
                    _fake_get_global_rank,
                    create=True,
                )
            )

        if hasattr(dist, "distributed_c10d"):
            c10d = dist.distributed_c10d
            stack.enter_context(
                mock.patch.object(
                    c10d,
                    "_all_gather_base",
                    _fake_all_gather_base,
                    create=True,
                )
            )
            if not hasattr(c10d, "_get_global_rank"):
                stack.enter_context(
                    mock.patch.object(
                        c10d,
                        "_get_global_rank",
                        _fake_get_global_rank,
                        create=True,
                    )
                )

    _FAKEGPU_STACK = stack

    if os.environ.get("PCCHECK_FAKEGPU_VERBOSE") == "1":
        print("[fakegpu] enabled fake CUDA hooks", file=sys.stderr)


if _fakegpu_requested():
    _install_fakegpu()
