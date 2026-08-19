# CPU VM system requirements

## Focused one-physical-node unit test

Gemini cannot be meaningfully tested with `world_size=1`: its snapshot group
size is 2 and a remote checkpoint requires a peer rank. The focused CPU test
therefore uses one physical VM with two local processes/ranks communicating over
loopback Gloo.

No NVIDIA driver, CUDA toolkit, or NCCL is required for that CPU test. The
required non-Python capabilities are only ordinary Linux process/socket support.

The repository itself is an older DeepSpeed tree and its pinned Python
requirements matter: `requirements/requirements.txt` pins `torch==1.13.0`,
`numpy==1.23.5`, and other era-matched packages. Use an environment compatible
with those pins rather than assuming a current system Python/PyTorch will import
the complete DeepSpeed package unchanged.

The Linux setup defaults `DS_BUILD_OPS=0`, so the focused snapshot test should
not need precompiled CUDA extensions. Do not enable `DS_BUILD_OPS=1` on a
CPU-only VM.

## Multi-node experiment

For two or more physical VMs, the ranks must be able to reach each other on the
TCP address/port used by the PyTorch process-group store and Gloo. No InfiniBand
or NCCL configuration is needed for the CPU emulation itself; those are replaced
by real Gloo communication and measured GPU/NCCL timing only at explicitly
calibrated boundaries.
