#!/usr/bin/env python3
"""Two-rank CPU test of Gemini's original SnapshotOptimizer.snapshot() path."""
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


def worker(rank, port, nbytes, trace_dir, queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        numel = nbytes // 4
        profile = TimingProfile(
            source="synthetic-unit-test-only",
            targets={("h2d", nbytes): 0.0, ("d2h", nbytes): 0.0},
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
                    (numel,), float(rank + 1), dtype=torch.float32
                ).cuda()
                recv_device = torch.empty(
                    numel, dtype=torch.float32, device="cuda"
                )
                remote_host = torch.empty(
                    numel, dtype=torch.float32, device="cpu"
                )

                assert runtime.is_device_surrogate(local_state)
                assert runtime.is_device_surrogate(recv_device)
                assert not runtime.is_device_surrogate(remote_host)

                opt = sc.SnapshotOptimizer(group_size=2, dtype=torch.float32)
                opt.snapshot_group = PeerGroup(peer)
                opt.snapshot_gpu_buffers = [recv_device]
                opt.snapshot_gpu_versions = 1
                opt.snapshot_gpu_buffer_id = 0
                opt.cur_block_id = 0
                opt.snapshot_current_version = 0
                opt.snapshot_versions = 1
                opt.block_sizes = [[numel]]
                opt.tensor_blocks = [[remote_host]]

                # DeepSpeed's comm wrapper is not initialized by this focused
                # test. Only its rank query is patched; Gemini's original
                # torch.distributed batch_isend_irecv remains real Gloo P2P.
                with mock.patch.object(sc.dist, "get_rank", return_value=rank):
                    opt.snapshot(local_state)

                # The original Gemini move_to_cpu(..., non_blocking=True) may
                # return before its copy stream completes. Synchronize exactly
                # where a consumer needs the remote DRAM checkpoint contents.
                torch.cuda.synchronize()
                expected = torch.full(
                    (numel,), float(peer + 1), dtype=torch.float32
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


def d2h_bytes(path):
    total = 0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "transfer" and row.get("kind") == "d2h":
            total += int(row.get("bytes", 0))
    return total


def run_test(size_mb):
    nbytes = int(size_mb) * 1024 * 1024
    if nbytes <= 0 or nbytes % 4:
        raise ValueError("size must be positive and float32 aligned")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()

    with tempfile.TemporaryDirectory(prefix="gemini-cpu-") as td:
        processes = [
            ctx.Process(target=worker, args=(rank, port, nbytes, td, queue))
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
            assert d2h_bytes(Path(td) / f"rank{rank}.jsonl") >= nbytes

        print("PASS Gemini CPU compatibility")
        print("ranks=2 backend=gloo")
        print(f"checkpoint_block_bytes={nbytes}")
        print("original_path=SnapshotOptimizer.snapshot")
        print(f"real_p2p_bytes_per_rank={nbytes}")
        print(f"real_d2h_surrogate_bytes_per_rank={nbytes}")
        print("d2h_copy_semantics=nonblocking fake copy stream")
        print("remote_checkpoint_location=separate CPU DRAM buffer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=1)
    args = parser.parse_args()
    run_test(args.size_mb)


if __name__ == "__main__":
    main()
