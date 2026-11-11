from __future__ import annotations

import os
import platform


def _mem_total_gb() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / 1024 / 1024, 2)
    except OSError:
        return None
    return None


def describe() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb_memtotal": _mem_total_gb(),
        "gpu": None,
        "cuda": False,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch is installed in this project
        info["torch_error"] = str(exc)
    return info
