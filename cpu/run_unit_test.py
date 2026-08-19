#!/usr/bin/env python3
"""Two-rank CPU test of Gemini's original remote snapshot path."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as torch_dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpu.gemini_stream_compat import GeminiAsyncCopyCompat
from cpu.gpu_compat import CpuGpuCompat, TimingProfile


class PeerGroup:
    def __init__(self, peer_rank):
        self.peer_rank = int(peer_rank)

    def get_peer_rank(self):
        return self.peer_rank


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def worker(rank, port, nbytes, blocks, trace_dir, queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        total_numel = nbytes // 4
        if total_numel % blocks:
            raise ValueError("float32 element count must divide evenly into blocks")
        block_numel = total_numel // blocks
        block_bytes = block_numel * 4

        profile = TimingProfile(
            source="synthetic-unit-test-only",
            targets={
                ("h2d", nbytes): 0.0,
                ("h2d", block_bytes): 0.0,
                ("d2h", block_bytes): 0.0,
            },
        )
        trace = str(Path(trace_dir) / f"rank{rank}.jsonl")
        with CpuGpuCompat(
            timing_profile=profile,
            strict_timing=True,
            trace_path=trace,
        ) as runtime:
            # Gemini depends on a nonblocking D2H copy stream. Install the
            # Gemini-only async copy surrogate before importing snapshot_comm,
            # because snapshot_comm captures `Stream` at import time.
            with GeminiAsyncCopyCompat(runtime):
                torch_dist.init_process_group(
                    backend="gloo", rank=rank, world_size=2
                )

                from deepspeed.runtime.snapshot import snapshot_comm as sc

                peer = 1 - rank
                local_state = torch.full(
                    (total_numel,), float(rank + 1), dtype=torch.float32
                ).cuda()
                remote_host = torch.empty(
                    total_numel, dtype=torch.float32, device="cpu"
                )
                recv_buffers = [
                    torch.empty(block_numel, dtype=torch.float32, device="cuda")
                    for _ in range(2)
                ]

                assert runtime.is_device_surrogate(local_state)
                assert all(runtime.is_device_surrogate(x) for x in recv_buffers)
                assert not runtime.is_device_surrogate(remote_host)

                opt = sc.SnapshotOptimizer(group_size=2, dtype=torch.float32)
                opt.snapshot_group = PeerGroup(peer)
                opt.snapshot_gpu_buffers = recv_buffers
                opt.snapshot_gpu_versions = len(recv_buffers)
                opt.snapshot_gpu_buffer_id = 0
                opt.cur_block_id = 0
                opt.snapshot_current_version = 0
                opt.snapshot_versions = 1
                opt.snapshot_blocks = [
                    local_state[i * block_numel:(i + 1) * block_numel]
                    for i in range(blocks)
                ]
                opt.total_blocks = blocks
                opt.block_sizes = [[block_numel for _ in range(blocks)]]
                opt.tensor_blocks = [[
                    remote_host[i * block_numel:(i + 1) * block_numel]
                    for i in range(blocks)
                ]]
                opt.comm_gap_id = 0
                opt.allgather_gap_num = -1
                training_stream = torch.cuda.Stream()

                # DeepSpeed's comm wrapper is not initialized by this focused
                # test. Only its rank/profile queries are patched. Gemini's
                # original torch.distributed batch_isend_irecv stays real Gloo.
                with mock.patch.object(sc.dist, "get_rank", return_value=rank), \
                     mock.patch.object(sc.snapshot_settings, "is_profile_mode", return_value=False):
                    opt.remote_snapshot_blocks(blocks, event=None, stream=training_stream)

                assert opt.cur_block_id == blocks

                # The D2H copy stream is intentionally asynchronous. A consumer
                # of the remote DRAM checkpoint must synchronize before reading.
                torch.cuda.synchronize()
                expected = torch.full(
                    (total_numel,), float(peer + 1), dtype=torch.float32
                )
                assert torch.equal(remote_host, expected)
                torch_dist.barrier()
                torch_dist.destroy_process_group()

        queue.put({"rank": rank, "ok": True})
    except BaseException as exc:
        try:
            if torch_dist.is_initialized():
                torch_dist.destroy_process_group()
        except Exception:
            pass
        queue.put({
            "rank": rank,
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })


def transfer_bytes(path, kind):
    total = 0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "transfer" and row.get("kind") == kind:
            total += int(row.get("bytes", 0))
    return total


def run_test(size_mb, blocks):
    nbytes = int(size_mb) * 1024 * 1024
    blocks = int(blocks)
    if nbytes <= 0 or nbytes % 4:
        raise ValueError("size must be positive and float32 aligned")
    if blocks <= 0 or (nbytes // 4) % blocks:
        raise ValueError("blocks must divide the float32 element count exactly")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()

    with tempfile.TemporaryDirectory(prefix="gemini-cpu-") as td:
        processes = [
            ctx.Process(target=worker, args=(rank, port, nbytes, blocks, td, queue))
            for rank in range(2)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join(120)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
                raise RuntimeError("Gemini CPU unit test timed out")

        results = [queue.get(timeout=5) for _ in range(2)]
        failures = [r for r in results if not r.get("ok")]
        if failures:
            raise RuntimeError("\n\n".join(
                r.get("traceback", r.get("error", "unknown"))
                for r in failures
            ))

        for rank in range(2):
            assert transfer_bytes(Path(td) / f"rank{rank}.jsonl", "d2h") >= nbytes

        print("PASS Gemini CPU compatibility")
        print("ranks=2 backend=gloo")
        print(f"checkpoint_bytes={nbytes}")
        print(f"blocks={blocks} block_bytes={nbytes // blocks}")
        print("original_path=SnapshotOptimizer.remote_snapshot_blocks -> snapshot_block -> snapshot")
        print(f"real_p2p_payload_bytes_per_rank={nbytes}")
        print(f"real_d2h_surrogate_bytes_per_rank={nbytes}")
        print("d2h_copy_semantics=nonblocking fake copy stream")
        print("remote_checkpoint_location=separate CPU DRAM buffer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    run_test(args.size_mb, args.blocks)


if __name__ == "__main__":
    main()
