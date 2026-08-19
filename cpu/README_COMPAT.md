# Gemini CPU compatibility: what is preserved and why

## Objective

Execute Gemini's original snapshot code on CPU-only VMs while preserving its
checkpoint topology: blockwise peer transfer followed by placement of the
received block into the peer checkpoint's host-DRAM buffer.

The `cpu/` folder does not reimplement Gemini. It changes hardware primitives
beneath the original code.

## Import ordering is part of the compatibility design

Gemini's `snapshot_comm.py` contains:

```python
from torch.cuda import Stream
```

That symbol is bound when the module is imported. Therefore the compatibility
layer must be installed **before** importing
`deepspeed.runtime.snapshot.snapshot_comm`. `cpu/run_unit_test.py` does this so
Gemini's existing `Stream()` construction receives the CPU stream surrogate.

## Exact mapping to the original Gemini code

### 1. Device buffers

Calls such as:

```python
torch.zeros(..., device=torch.cuda.current_device())
```

still execute. The compatibility layer returns a CPU allocation with identical
shape/dtype/bytes and tags its storage as a device surrogate. `*_like` factories
inherit that tag when the original call would have inherited the CUDA device.

### 2. Peer checkpoint transfer stays real

The original `_snapshot_torch_p2p_fn()` builds:

```text
P2POp(isend, input_tensor, peer)
P2POp(irecv, output_tensor, peer)
batch_isend_irecv(...)
req.wait()
```

Those calls are not mocked. On CPU buffers the unit test initializes a real
Gloo process group, so the checkpoint block physically moves between the two
processes.

The hardware substitution is only:

```text
GPU run:  CUDA/NCCL-capable tensor storage + GPU communication backend
CPU run:  tagged CPU device-surrogate storage + Gloo backend
```

The peer selected by Gemini, block size, number of blocks, P2P call structure,
and wait points remain Gemini logic.

### 3. Remote GPU receive buffer -> remote host DRAM

The original Gemini path is:

```text
SnapshotOptimizer.snapshot()
  -> _snapshot_torch_p2p_fn()
  -> with torch.cuda.stream(checkpoint_copy_stream)
  -> move_to_cpu()
  -> tensor_on_cpu.copy_(received_tensor, non_blocking=True)
```

`move_to_cpu()` is unchanged. The received tensor is backed by tagged
CPU device-surrogate storage; `tensor_on_cpu` is a distinct ordinary CPU
allocation representing Gemini's host checkpoint DRAM. Therefore the original
`copy_` still moves the complete received block into a second buffer.

For an exact measured D2H target:

```text
emulated D2H time = max(real CPU copy time, measured GPU D2H time)
```

Only the residual is slept. Missing sizes are not extrapolated.

### 4. Pinned CPU checkpoint buffers

Gemini pins its CPU optimizer-state/checkpoint tensors to support CUDA transfer.
On CPU-only hardware the shim creates a distinct ordinary CPU allocation.
This preserves the state-placement boundary and memory footprint without
claiming CUDA page-locking exists.

### 5. NCCL -> Gloo

The compatibility layer translates only process-group initialization requesting
`nccl` to `gloo`. It does not replace send/recv/collective/barrier functions.
This is why network traffic remains observable on CPU VMs.

The focused unit test directly initializes Gloo. It patches only the
DeepSpeed-comm wrapper's rank query because that wrapper is not initialized by
the small test; the actual Gemini P2P transfer still uses its original
`torch.distributed.P2POp` and `batch_isend_irecv` calls.

## Streams and the important remaining limitation

`torch.cuda.Stream`, `torch.cuda.stream`, events, and synchronization points are
provided with compatible CPU-side objects so the original control flow can
execute.

At present these stream surrogates are synchronous. This preserves ordering,
but it does **not** physically reproduce overlap among CUDA kernels, NCCL, and
GPU copy engines.

That matters more for Gemini than for the other schemes because Gemini's main
optimization is interleaving checkpoint blocks into communication gaps. The
algorithmic decisions about which gap gets which block remain Gemini's own
code, but exact hardware overlap is a remaining calibration/emulation problem.
It is explicitly not hidden by this patch.

## What the unit test proves

`cpu/run_unit_test.py` starts two processes and calls the original
`SnapshotOptimizer.snapshot()` on each rank. Success requires:

1. a real Gloo P2P transfer of the requested block size,
2. the peer's actual values to arrive,
3. a real second copy into Gemini's separate host checkpoint buffer,
4. a D2H transfer trace for the full block.

Run:

```bash
python cpu/run_unit_test.py --size-mb 1
```

## Timing profile

The example profile is marked `synthetic-unit-test-only`. It exists to show the
exact-size schema. Replace it with measured GPU captures before timing results
are treated as GPU-equivalent.

## Remaining fidelity limits

1. Gloo preserves real byte movement and ordering but is not NCCL's protocol,
   GPU-direct path, or contention behavior.
2. CUDA stream/copy-engine overlap is currently synchronous on CPU.
3. GPU kernel execution is not reproduced by this focused checkpoint test.
4. Function-level forward/backward/optimizer waits require real GPU capture;
   none are invented here.
