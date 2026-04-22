from __future__ import annotations

import os
import sys
import time
from multiprocessing import get_context
from typing import Dict, List

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from state import build_hard_state, hard_to_full_placement  # noqa: E402
from worker import WorkerConfig, worker_entry  # noqa: E402


class FastMCMCPlacer:
    """
    High-Speed MCMC Simulated Annealing placer (fast_mcmc).

    - Data-oriented hard macro state (NumPy arrays)
    - Numba-compiled overlap + HPWL deltas
    - Multiprocessing across seeds
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        st, plc = build_hard_state(benchmark)

        # Config: default to near-timeout, but allow env override for local dev.
        time_limit_s = float(os.getenv("FAST_MCMC_TIME_LIMIT_S", str(58 * 60)))
        n_workers = int(os.getenv("FAST_MCMC_WORKERS", "16"))

        # Spatial grid configuration
        rows = int(os.getenv("FAST_MCMC_GRID_ROWS", str(max(16, benchmark.grid_rows))))
        cols = int(os.getenv("FAST_MCMC_GRID_COLS", str(max(16, benchmark.grid_cols))))
        max_per_bin = int(os.getenv("FAST_MCMC_MAX_PER_BIN", "96"))
        gap = float(os.getenv("FAST_MCMC_GAP", "0.05"))
        cfg = {
            "n_workers": n_workers,
            "rows": rows,
            "cols": cols,
            "max_per_bin": max_per_bin,
            "gap": gap,
            "time_limit_s": time_limit_s,
        }

        # Wall-clock: give a small cushion for result collection
        start = time.time()
        per_worker = max(1.0, float(cfg["time_limit_s"]) - 5.0)

        seeds = [12345 + 1009 * i for i in range(int(cfg["n_workers"]))]
        worker_cfgs = [
            WorkerConfig(
                seed=s,
                time_limit_s=per_worker,
                rows=int(cfg["rows"]),
                cols=int(cfg["cols"]),
                max_per_bin=int(cfg["max_per_bin"]),
                gap=float(cfg["gap"]),
            )
            for s in seeds
        ]

        # Multiprocessing: fork is fastest on macOS; use spawn-safe ctx anyway
        ctx = get_context("fork" if os.name != "nt" else "spawn")
        results: List[Dict[str, object]] = []

        with ctx.Pool(processes=int(cfg["n_workers"])) as pool:
            for r in pool.imap_unordered(worker_entry, [(st, wc) for wc in worker_cfgs], chunksize=1):
                results.append(r)
                if time.time() - start > float(cfg["time_limit_s"]):
                    break

        # Pick best by true proxy cost (including density + congestion), as cursor.md requires.
        best_cost = float("inf")
        best_full: torch.Tensor | None = None

        for r in results:
            hard_xy = r["pos"]
            placement = hard_to_full_placement(benchmark, hard_xy)
            is_valid, _ = validate_placement(placement, benchmark)
            if not is_valid:
                continue
            costs = compute_proxy_cost(placement, benchmark, plc)
            if costs["overlap_count"] != 0:
                continue
            proxy = float(costs["proxy_cost"])
            if proxy < best_cost:
                best_cost = proxy
                best_full = placement

        if best_full is None:
            # Fallback: return the benchmark initial placement (always valid per dataset)
            return benchmark.macro_positions.clone()

        return best_full

