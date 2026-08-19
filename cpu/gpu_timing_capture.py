"""Collect measured CUDA-event durations for named GPU compute phases."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import torch


class GpuTimingCapture:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU timing capture requires CUDA")
        self.samples = {}

    def measure(self, name, operation):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        self.samples.setdefault(str(name), []).append(
            float(start.elapsed_time(end)) / 1000.0
        )
        return result

    def write_profile(self, path, source):
        profile = {
            "source": str(source),
            "units": "seconds",
            "reducer": "median",
            "functions": {
                name: median(values)
                for name, values in sorted(self.samples.items())
                if values
            },
            "sample_counts": {
                name: len(values)
                for name, values in sorted(self.samples.items())
            },
        }
        Path(path).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
