"""
fast_mcmc.main
==============

Orchestrator + CLI for the high-performance MCMC macro placer.

This module is the *only* file the challenge evaluator interacts with:

.. code-block:: bash

   uv run evaluate submissions/fast_mcmc/main.py -b ibm01

The evaluator's contract is dead-simple – it loads this file, picks the
first class that has a ``place(self, benchmark) -> torch.Tensor`` method,
instantiates it with no arguments, then hands it a
``macro_place.benchmark.Benchmark`` and expects an ``(N, 2)`` tensor of
*center* coordinates in return.  Everything else – multiprocessing pool
management, benchmark fanout to workers, MCMC time budget, result
aggregation, and final layout selection – is the responsibility of this
module.

Architecture (cursor.md §C + §G)
--------------------------------
1. **Immediate forking.** ``main.py`` does *only* the parse-and-fanout
   work on the parent process.  It never builds a ``PlacementState`` or
   a ``spatial_grid`` on the parent – that work happens entirely inside
   each forked worker, so the inter-process payload stays small.  The
   torch-backed ``Benchmark`` is first converted *once* on the parent
   into a pure-numpy :class:`_BenchmarkBundle` ("only raw data, canvas
   dimensions, and a unique random seed" per cursor.md §C), which
   pickles cleanly through ``multiprocessing`` with no transitive
   torch / macro_place / absl dependency on the workers.

2. **multiprocessing.Pool with N forked workers.**  Each worker receives
   a unique random seed so the parallel ensemble explores structurally
   different starting layouts (delegated to
   :func:`fast_mcmc.initialization.grasp_initialize`) and follows
   independent Metropolis trajectories (delegated to
   :func:`fast_mcmc.worker.run_worker`).  The fork context is used when
   available (Linux / macOS), which gets us copy-on-write benchmark
   sharing and avoids re-pickling the netlist into every worker.

3. **Aggregation.**  Workers stream back :class:`WorkerResult` objects.
   We discard any layout that the worker's own validity sweep flagged
   ``invalid`` (per cursor.md §F.4), then pick the survivor with the
   lowest proxy cost.  If *no* worker returned a valid layout we fall
   back to the best-scoring invalid one (and warn loudly) so the
   evaluator's downstream metrics at least have something to compute.

4. **Output.**  The winning ``PlacementState.macro_coords`` (bottom-left)
   is converted back to the evaluator's ``(N, 2)`` *center* convention
   and returned as a ``torch.Tensor`` matching ``benchmark.macro_positions``'s
   dtype and device.

CLI overrides (cursor.md §G)
----------------------------
The class reads three knobs from environment variables so callers can
override the defaults without modifying the class signature (which has
to stay ``__init__()``-with-no-arguments for the evaluator)::

    FAST_MCMC_TIMEOUT   wall-clock budget per benchmark, seconds  [60]
    FAST_MCMC_WORKERS   pool size; 0 ⇒ cpu_count, 1 ⇒ serial      [0]
    FAST_MCMC_SEED      seed of the *first* worker; others get +i [0]

Running ``python submissions/fast_mcmc/main.py`` directly bypasses the
evaluator harness and exposes ``argparse`` flags ``-b/-w/-t/-s``.  This
is the verification path that does *not* require ``macro_place`` /
``absl`` to be importable – it loads the cached ``.pt`` snapshot
directly.
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# The challenge harness (``macro_place.evaluate._load_placer``) loads
# submission files via ``importlib.util.spec_from_file_location`` +
# ``exec_module`` *without* first registering the module in
# ``sys.modules``.  That confuses several stdlib facilities that walk
# ``sys.modules[cls.__module__]`` to locate the defining module – most
# notably ``pickle``'s ``whichmodule`` (which the multiprocessing
# forking pickler uses to pickle classes between parent and worker).
# Plant a stand-in module object whose ``__dict__`` reflects our
# globals so those lookups succeed.  When the loader *does* register us
# properly (e.g. ``import main`` from a unit test), this is a no-op.
if __name__ not in sys.modules:
    _self_stub = types.ModuleType(__name__)
    # ``ModuleType.__dict__`` is a read-only descriptor at the C level,
    # so we cannot literally alias it to ``globals()``.  Snapshotting
    # before any class definitions is also wrong (subsequent symbols
    # wouldn't be reachable through the stub).  Instead, point the
    # stub's mapping at our real globals via ``vars(...)``-style
    # copying *after* the module body finishes – but that is too late.
    # The reliable, lightweight workaround is to expose a ``__getattr__``
    # that proxies into the live globals.  ``pickle.whichmodule`` only
    # needs to find the class by its qualified name on this object.
    _self_globals = globals()
    def _stub_getattr(name, _g=_self_globals):
        try:
            return _g[name]
        except KeyError as _exc:
            raise AttributeError(name) from _exc
    _self_stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
    sys.modules[__name__] = _self_stub
    del _self_stub, _self_globals, _stub_getattr

# Local package modules – kept as flat (non-relative) imports because
# ``submissions/fast_mcmc/`` is not a Python package (no ``__init__.py``).
# The challenge harness's ``_load_placer`` (``macro_place/evaluate.py``)
# loads us via ``importlib.util.spec_from_file_location`` + ``exec_module``,
# which does NOT add the placer file's directory to ``sys.path``.  We
# therefore self-bootstrap it here so ``import fast_eval``,
# ``from state import …``, etc. resolve correctly.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# #region agent log
try:
    import json as _dbg_json, os as _dbg_os, time as _dbg_time
    _dbg_main_dir = _dbg_os.path.dirname(_dbg_os.path.abspath(__file__))
    with open('/Users/lucasolmeta/Desktop/Projects/macro-place-challenge-2026/.cursor/debug-ad94f7.log', 'a') as _dbg_f:
        _dbg_f.write(_dbg_json.dumps({
            "sessionId": "ad94f7",
            "hypothesisId": "H1+H2+H3",
            "runId": "pre-fix",
            "location": "submissions/fast_mcmc/main.py:120 (just before `import fast_eval`)",
            "message": "sys.path / cwd / file-location snapshot at the failing import",
            "data": {
                "cwd": _dbg_os.getcwd(),
                "__file__": __file__,
                "__name__": __name__,
                "main_dir": _dbg_main_dir,
                "main_dir_on_sys_path": _dbg_main_dir in sys.path,
                "fast_eval_exists_in_main_dir": _dbg_os.path.exists(_dbg_os.path.join(_dbg_main_dir, "fast_eval.py")),
                "state_exists_in_main_dir": _dbg_os.path.exists(_dbg_os.path.join(_dbg_main_dir, "state.py")),
                "sys_path": list(sys.path),
                "PYTHONPATH_env": _dbg_os.environ.get("PYTHONPATH", "<unset>"),
            },
            "timestamp": int(_dbg_time.time() * 1000),
        }) + "\n")
    del _dbg_json, _dbg_os, _dbg_time, _dbg_main_dir, _dbg_f
except Exception:
    pass
# #endregion

import fast_eval as fe
from initialization import GraspReport
from state import (
    PlacementState,
    bottom_left_to_centers,
    build_state,
)
from worker import WorkerConfig, WorkerResult, run_worker


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 0. Defaults & environment helpers                                      ║
# ╚════════════════════════════════════════════════════════════════════════╝

DEFAULT_TIMEOUT_SECONDS: float = 3000.0
DEFAULT_NUM_WORKERS: int = 0   # 0 ⇒ os.cpu_count() (capped at 16)
DEFAULT_SEED: int = 0
MAX_POOL_SIZE: int = 16        # cursor.md §C: "spawns 16 independent worker processes"

# Per-worker safety margin: if a single worker overruns its own budget by
# more than this many seconds we ``terminate()`` the whole pool so the
# evaluator's outer timer never starves the aggregation phase.
WORKER_HARD_OVERRUN_SECONDS: float = 5.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"[fast_mcmc] WARNING: ${name}={raw!r} is not a number; using default {default}",
              file=sys.stderr)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print(f"[fast_mcmc] WARNING: ${name}={raw!r} is not an integer; using default {default}",
              file=sys.stderr)
        return int(default)


def _resolve_num_workers(requested: int) -> int:
    """Translate ``requested`` (``0`` ⇒ auto) into a concrete pool size."""
    if requested > 0:
        return max(1, requested)
    cpu = os.cpu_count() or 1
    # cursor.md §C asks for 16; never go higher (diminishing returns,
    # and respect the spec's pool-size hint).
    return max(1, min(MAX_POOL_SIZE, cpu))


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 1. Numpy-only benchmark bundle (zero-dependency IPC payload)           ║
# ╚════════════════════════════════════════════════════════════════════════╝
#
# cursor.md §C asks us to ship "only raw data, canvas dimensions, and a
# unique random seed" to each worker.  Pickling a torch-backed
# ``macro_place.benchmark.Benchmark`` works fine in the production path
# (where ``macro_place`` is importable), but it requires the workers to
# carry the full ``macro_place`` + ``absl`` + ``torch`` import surface.
#
# We instead convert the benchmark *once* on the parent process into a
# pure-numpy ``_BenchmarkBundle``.  The bundle quacks like a
# ``Benchmark`` (via ``_NpTensor`` shims) so ``build_state`` can consume
# it unchanged, and it pickles cleanly through ``multiprocessing`` with
# no extra modules required.

class _NpTensor:
    """Duck-typed view that mirrors the subset of ``torch.Tensor`` that
    :func:`fast_mcmc.state.build_state` actually calls.

    Carrying numpy through the IPC channel avoids importing torch on the
    workers entirely; the same shim lets the benchmark consumer ignore
    whether the array came from torch or numpy in the first place.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = np.ascontiguousarray(arr)

    def detach(self) -> "_NpTensor":
        return self

    def cpu(self) -> "_NpTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def numel(self) -> int:
        return int(self._arr.size)

    def clone(self) -> "_NpTensor":
        return _NpTensor(self._arr.copy())

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    def sum(self):
        return self._arr.sum()

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


@dataclass
class _BenchmarkBundle:
    """Picklable, numpy-only mirror of ``macro_place.benchmark.Benchmark``.

    Only contains the fields :func:`build_state` reads.  Constructed on
    the parent via :meth:`from_benchmark`, then handed verbatim to every
    worker through the multiprocessing pool.
    """

    name: str
    canvas_width:  float
    canvas_height: float
    num_macros:      int
    num_hard_macros: int
    num_soft_macros: int
    num_nets:        int

    macro_positions: _NpTensor   # (N, 2) centres
    macro_sizes:     _NpTensor   # (N, 2)
    macro_fixed:     _NpTensor   # (N,) bool
    port_positions:  _NpTensor   # (P, 2) or empty
    net_weights:     _NpTensor   # (num_nets,) or empty
    macro_names:     List[str]
    net_nodes:       List[np.ndarray]

    grid_rows: int = 0
    grid_cols: int = 0

    @classmethod
    def from_benchmark(cls, b: Any) -> "_BenchmarkBundle":
        def _arr(field_name: str) -> _NpTensor:
            t = getattr(b, field_name)
            if isinstance(t, _NpTensor):
                return t
            # torch.Tensor → numpy.
            return _NpTensor(t.detach().cpu().numpy())

        net_nodes_np: List[np.ndarray] = []
        for nn in getattr(b, "net_nodes", []) or []:
            if isinstance(nn, np.ndarray):
                net_nodes_np.append(np.ascontiguousarray(nn))
            else:
                # torch.Tensor or sequence.
                net_nodes_np.append(np.asarray(nn))

        return cls(
            name=str(b.name),
            canvas_width =float(b.canvas_width),
            canvas_height=float(b.canvas_height),
            num_macros     =int(b.num_macros),
            num_hard_macros=int(b.num_hard_macros),
            num_soft_macros=int(b.num_soft_macros),
            num_nets       =int(b.num_nets),
            macro_positions=_arr("macro_positions"),
            macro_sizes    =_arr("macro_sizes"),
            macro_fixed    =_arr("macro_fixed"),
            port_positions =_arr("port_positions"),
            net_weights    =_arr("net_weights"),
            macro_names    =list(b.macro_names),
            net_nodes      =net_nodes_np,
            grid_rows=int(getattr(b, "grid_rows", 0) or 0),
            grid_cols=int(getattr(b, "grid_cols", 0) or 0),
        )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 2. Worker-task dataclass + top-level pool entry point                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class _WorkerTask:
    """Per-worker payload sent through the multiprocessing pool.

    Kept as a small dataclass (not a closure) so it pickles cleanly
    under both *fork* and *spawn* multiprocessing start methods.
    ``bundle`` is a numpy-only mirror of the source benchmark so workers
    don't transitively need ``macro_place`` / ``absl`` / ``torch``.
    """
    bundle: _BenchmarkBundle
    config: WorkerConfig
    worker_index: int = 0     # 0..N-1, for diagnostics


def _worker_entry(task: _WorkerTask) -> WorkerResult:
    """Top-level entry point executed on each pool worker.

    Builds a per-worker :class:`PlacementState` from the raw bundle
    (cursor.md §C: "Do not construct grid states on the main thread")
    and runs the full MCMC pipeline.  Errors are caught and re-packaged
    as an invalid :class:`WorkerResult` so a crashing worker does not
    bring down the entire pool.
    """
    cfg = task.config
    try:
        # All heavy state construction (spatial grid allocation, CSR
        # netlist, ``net_bbox`` initialisation) happens inside the worker
        # process, never in the parent.
        state = build_state(task.bundle, stamp_hard_macros=False)
        result = run_worker(state, cfg, copy_state=False)
        return result
    except Exception as exc:  # pragma: no cover – defensive
        traceback.print_exc()
        empty_grasp = GraspReport(
            seed=int(cfg.seed), num_movable_hard=0, num_placed_clean=0,
            num_placed_via_blf=0, num_placed_with_overlap=0,
            sum_partial_hpwl=0.0, movable_order=np.zeros(0, dtype=np.int64),
        )
        try:
            state = build_state(task.bundle, stamp_hard_macros=False)
        except Exception:
            # If we cannot even build a state we have nothing to return;
            # the caller will pick the *least worst* surviving worker.
            raise
        return WorkerResult(
            seed=int(cfg.seed),
            state=state,
            grasp_report=empty_grasp,
            valid=False,
            violations=[f"worker crashed: {type(exc).__name__}: {exc}"],
            cost_proxy=float("inf"),
        )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 2. Numba pre-warm (parent-side, fork-only optimisation)                ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _prewarm_numba(bundle: _BenchmarkBundle) -> None:
    """JIT-compile the hot kernels on the parent before forking workers.

    On Unix fork contexts the compiled function objects are inherited by
    every child via copy-on-write, so 16 workers do *not* each pay the
    Numba compilation tax.  On spawn / Windows contexts this is a no-op
    because the child re-imports everything from scratch.

    The pre-warm runs each kernel on a tiny synthetic input so the call
    sites compile but do not waste any real benchmark time.
    """
    if not fe.NUMBA_AVAILABLE:
        return
    try:
        # Compile the spatial-grid + HPWL + density kernels by running a
        # one-shot computation on a 2-macro / 1-net synthetic state.
        state = build_state(bundle, stamp_hard_macros=True)
        # HPWL ground truth – exercises net_bbox_with_override + co.
        _ = fe.compute_total_hpwl_njit(
            state.net_pin_owners, state.net_pin_offsets,
            state.net_weights,
            state.macro_coords, state.macro_dims,
            state.port_coords, state.num_macros,
            state.num_nets,
        )
        # Density grid construction.
        density_grid = np.zeros(
            (state.grid_num_rows, state.grid_num_cols), dtype=np.float64,
        )
        fe.compute_density_grid_njit(
            state.macro_coords, state.macro_dims, state.num_macros,
            state.grid_bin_width, state.grid_bin_height,
            state.grid_num_rows, state.grid_num_cols, density_grid,
        )
        # Shift / density-shift / commit kernels.
        if state.num_macros > 0:
            cap = max(1, fe.max_cells_per_swap(
                state.macro_dims, state.grid_bin_width, state.grid_bin_height,
            ))
            cap_nets = max(1, 2 * fe.max_nets_per_macro(state.macro_net_offsets))
            aff = np.zeros(cap_nets, dtype=np.int32)
            nbb = np.zeros((cap_nets, 4), dtype=np.float64)
            crs = np.zeros(cap, dtype=np.int32)
            ccs = np.zeros(cap, dtype=np.int32)
            nds = np.zeros(cap, dtype=np.float64)
            _ = fe.hpwl_delta_for_shift_njit(
                0, state.macro_coords[0, 0], state.macro_coords[0, 1],
                state.macro_coords, state.macro_dims,
                state.port_coords, state.num_macros,
                state.macro_net_ids, state.macro_net_offsets,
                state.net_pin_owners, state.net_pin_offsets,
                state.net_weights, state.net_bbox,
                aff, nbb,
            )
            _ = fe.density_grid_shift_delta_njit(
                0, state.macro_coords[0, 0], state.macro_coords[0, 1],
                state.macro_coords, state.macro_dims,
                state.grid_bin_width, state.grid_bin_height,
                state.grid_num_rows, state.grid_num_cols,
                density_grid, crs, ccs, nds,
            )
    except Exception as exc:  # pragma: no cover – defensive
        # Pre-warm failures are non-fatal; workers will JIT lazily.
        print(f"[fast_mcmc] Numba pre-warm skipped: {exc}", file=sys.stderr)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 3. Pool orchestration                                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class _PoolOutcome:
    """Outcome of one ``_run_pool`` invocation, returned to the placer."""
    results: List[WorkerResult] = field(default_factory=list)
    best_index: int = -1            # index into ``results`` (any layout)
    best_valid_index: int = -1      # index into ``results`` (valid only)
    elapsed_seconds: float = 0.0
    num_workers: int = 0
    pool_start_method: str = "fork"


def _build_worker_configs(
    num_workers: int,
    seed_base: int,
    timeout_seconds: float,
    overrides: Optional[WorkerConfig] = None,
) -> List[WorkerConfig]:
    """Produce one :class:`WorkerConfig` per pool slot.

    All workers share the same time budget and cost weights; only the
    random seed differs (each worker gets ``seed_base + i``).  The
    optional ``overrides`` template lets the CLI override anything else
    in one place (e.g. disable GRASP for ablations).
    """
    template = overrides or WorkerConfig()
    out: List[WorkerConfig] = []
    for i in range(num_workers):
        cfg = WorkerConfig(**vars(template))
        cfg.seed = int(seed_base) + i
        cfg.time_budget_seconds = float(timeout_seconds)
        # The biased selector and grid refresh are per-worker independent.
        out.append(cfg)
    return out


def _pick_start_method() -> str:
    """Choose ``fork`` if the platform supports it, else ``spawn``.

    ``fork`` is preferred because it gives us free copy-on-write sharing
    of the benchmark and inherits any Numba JIT cache pre-warmed on the
    parent process.
    """
    if "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


def _run_pool(
    benchmark: Any,
    num_workers: int,
    seed_base: int,
    timeout_seconds: float,
    overrides: Optional[WorkerConfig] = None,
    *,
    verbose: bool = False,
    prewarm: bool = True,
) -> _PoolOutcome:
    """Fan ``num_workers`` MCMC workers out onto a process pool.

    Returns aggregated :class:`_PoolOutcome` with best-by-cost indices.
    """
    num_workers = max(1, int(num_workers))
    configs = _build_worker_configs(num_workers, seed_base, timeout_seconds, overrides)

    # Convert the benchmark to a numpy-only bundle *once* on the parent.
    # Workers receive only flat numpy arrays + scalars + lists – no torch
    # tensors, no ``macro_place`` re-import, no JIT cost from each child.
    bundle = (
        benchmark
        if isinstance(benchmark, _BenchmarkBundle)
        else _BenchmarkBundle.from_benchmark(benchmark)
    )

    if prewarm:
        _prewarm_numba(bundle)

    tasks = [
        _WorkerTask(bundle=bundle, config=cfg, worker_index=i)
        for i, cfg in enumerate(configs)
    ]

    start_method = _pick_start_method()
    ctx = mp.get_context(start_method)

    t0 = time.perf_counter()

    if num_workers == 1:
        # Serial path – useful for step-through debugging (cursor.md §G).
        results = [_worker_entry(tasks[0])]
    else:
        # Hard timeout for the pool ``.get()`` call so a runaway worker
        # can never block aggregation beyond the user's budget.
        wait_budget = float(timeout_seconds) + WORKER_HARD_OVERRUN_SECONDS

        with ctx.Pool(processes=num_workers) as pool:
            async_results = pool.map_async(_worker_entry, tasks)
            try:
                results = async_results.get(timeout=wait_budget + WORKER_HARD_OVERRUN_SECONDS)
            except mp.TimeoutError:
                pool.terminate()
                pool.join()
                if verbose:
                    print(
                        f"[fast_mcmc] pool timed out after {wait_budget:.1f}s; "
                        "no usable results from this run",
                        file=sys.stderr,
                    )
                return _PoolOutcome(
                    results=[],
                    elapsed_seconds=time.perf_counter() - t0,
                    num_workers=num_workers,
                    pool_start_method=start_method,
                )

    elapsed = time.perf_counter() - t0

    # ── Find the best-by-cost layouts ────────────────────────────────────
    best_idx = -1
    best_valid_idx = -1
    best_cost = math.inf
    best_valid_cost = math.inf
    for i, r in enumerate(results):
        c = r.cost_proxy
        if math.isfinite(c) and c < best_cost:
            best_cost = c
            best_idx = i
        if r.valid and math.isfinite(c) and c < best_valid_cost:
            best_valid_cost = c
            best_valid_idx = i

    outcome = _PoolOutcome(
        results=list(results),
        best_index=best_idx,
        best_valid_index=best_valid_idx,
        elapsed_seconds=elapsed,
        num_workers=num_workers,
        pool_start_method=start_method,
    )

    if verbose:
        _print_pool_summary(outcome, benchmark_name=getattr(benchmark, "name", "?"))

    return outcome


def _print_pool_summary(outcome: _PoolOutcome, *, benchmark_name: str = "?") -> None:
    """Pretty-print a one-line-per-worker summary of the pool run."""
    print()
    print(f"=== fast_mcmc pool summary :: benchmark={benchmark_name} ===")
    print(f"  workers={outcome.num_workers}  start_method={outcome.pool_start_method}  "
          f"elapsed={outcome.elapsed_seconds:.2f}s")
    header = (
        f"  {'seed':>5} {'valid':>5} {'iters':>10} {'iter/s':>8} "
        f"{'cost₀':>10} {'cost':>10} {'WL':>8} {'D':>8} {'C':>8} "
        f"{'acc(s/w/r)':>14} {'rej_lg':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in outcome.results:
        ips = r.iterations / max(r.elapsed_seconds, 1e-9)
        print(
            f"  {r.seed:>5} {('Y' if r.valid else 'N'):>5} "
            f"{r.iterations:>10,} {ips:>8,.0f} "
            f"{r.initial_cost_proxy:>10.3f} {r.cost_proxy:>10.3f} "
            f"{r.cost_wirelength:>8.3f} {r.cost_density:>8.3f} {r.cost_congestion:>8.3f} "
            f"{r.accepts_shift:>4}/{r.accepts_swap:>4}/{r.accepts_reshape:>4} "
            f"{r.rejects_legalization:>7,}"
        )
    if outcome.best_valid_index >= 0:
        best = outcome.results[outcome.best_valid_index]
        print(f"  → best VALID worker: seed={best.seed} cost={best.cost_proxy:.4f}")
    elif outcome.best_index >= 0:
        best = outcome.results[outcome.best_index]
        print(f"  → no valid worker; best-effort: seed={best.seed} "
              f"cost={best.cost_proxy:.4f}  violations={best.violations}")
    else:
        print("  → no workers returned a usable result")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 4. Placer class (evaluator-facing entry point)                         ║
# ╚════════════════════════════════════════════════════════════════════════╝

class FastMCMCPlacer:
    """High-performance parallel MCMC macro placer.

    Constructed with **no arguments** so it slots into the challenge
    evaluator's ``_load_placer`` reflection (see
    ``macro_place/evaluate.py``).  Runtime knobs are read from
    environment variables on construction:

    +-----------------------+--------------------------------+---------+
    | Env var               | Meaning                        | Default |
    +=======================+================================+=========+
    | ``FAST_MCMC_TIMEOUT`` | Wall-clock budget, seconds     | 60      |
    | ``FAST_MCMC_WORKERS`` | Pool size; 0 → auto, 1 →serial | 0       |
    | ``FAST_MCMC_SEED``    | Seed of the first worker       | 0       |
    | ``FAST_MCMC_VERBOSE`` | Print per-worker summary       | 0       |
    +-----------------------+--------------------------------+---------+

    ``place(benchmark)`` is the only public method the evaluator calls;
    it forks the worker pool, waits for all (or times out), discards
    invalid layouts, and returns the best valid layout as a torch
    ``(N, 2)`` *center* tensor.
    """

    def __init__(self) -> None:
        self.timeout_seconds: float = _env_float("FAST_MCMC_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        self.num_workers: int = _resolve_num_workers(
            _env_int("FAST_MCMC_WORKERS", DEFAULT_NUM_WORKERS),
        )
        self.seed_base: int = _env_int("FAST_MCMC_SEED", DEFAULT_SEED)
        self.verbose: bool = bool(_env_int("FAST_MCMC_VERBOSE", 0))
        self.last_outcome: Optional[_PoolOutcome] = None
        # ``overrides`` is an optional ``WorkerConfig`` the CLI driver
        # populates when the user wants to ablate something (e.g.
        # ``--no-grasp``).  The evaluator path leaves it ``None``.
        self.overrides: Optional[WorkerConfig] = None

    # The signature MUST match the evaluator's expectation; do not change.
    def place(self, benchmark) -> "Any":
        """Run the parallel MCMC pipeline on ``benchmark``.

        Returns
        -------
        torch.Tensor
            ``(num_macros, 2)`` float tensor of *center* coordinates,
            matching the dtype / device of ``benchmark.macro_positions``.
            Fixed macros and any macro the worker pool failed to improve
            are returned at the benchmark's initial positions.
        """
        # Lazily import torch so the module can be imported on a torch-less
        # interpreter (the worker code path uses pure numpy throughout).
        import torch

        # ── 1. Fan out workers ──────────────────────────────────────────
        outcome = _run_pool(
            benchmark,
            num_workers=self.num_workers,
            seed_base=self.seed_base,
            timeout_seconds=self.timeout_seconds,
            overrides=self.overrides,
            verbose=self.verbose,
            prewarm=True,
        )
        self.last_outcome = outcome

        # ── 2. Pick the winning layout ──────────────────────────────────
        winner_state, winner_is_valid, winner_seed = self._select_winner(
            outcome, benchmark,
        )

        # ── 3. Convert bottom-left → center tensor ──────────────────────
        # The evaluator works in *centre* coordinates.  Soft macros that
        # were reshape-mutated during MCMC have their original size
        # restored at output time so the eval's downstream metrics use
        # consistent ``(centre, size)`` pairs.
        return self._state_to_center_tensor(winner_state, benchmark, torch)

    # ── helpers ────────────────────────────────────────────────────────

    def _select_winner(
        self, outcome: _PoolOutcome, benchmark,
    ) -> Tuple[PlacementState, bool, int]:
        """Pick the best layout from ``outcome.results``.

        Strategy (cursor.md §C / §F.4):
        * If at least one worker returned ``valid=True``, pick the
          valid result with the lowest proxy cost.
        * Otherwise warn and fall back to the lowest-cost result we
          have – the evaluator's own validator will then mark it
          invalid downstream, but at least the run produces output.
        * If the pool returned nothing at all (e.g. it timed out), we
          re-build a fresh ``PlacementState`` from the benchmark and
          return that – an honest "no-op" placement.
        """
        if outcome.best_valid_index >= 0:
            winner = outcome.results[outcome.best_valid_index]
            return winner.state, True, winner.seed

        if outcome.best_index >= 0:
            winner = outcome.results[outcome.best_index]
            print(
                f"[fast_mcmc] WARNING: no worker produced a valid layout; "
                f"returning best-effort layout from seed={winner.seed} "
                f"(cost_proxy={winner.cost_proxy:.4f}, violations="
                f"{winner.violations})",
                file=sys.stderr,
            )
            return winner.state, False, winner.seed

        print(
            "[fast_mcmc] WARNING: worker pool returned no results "
            "(likely timed out before any worker finished). "
            "Returning the benchmark's initial placement unchanged.",
            file=sys.stderr,
        )
        bundle = (
            benchmark
            if isinstance(benchmark, _BenchmarkBundle)
            else _BenchmarkBundle.from_benchmark(benchmark)
        )
        fallback = build_state(bundle, stamp_hard_macros=False)
        return fallback, False, -1

    def _state_to_center_tensor(
        self, state: PlacementState, benchmark, torch,
    ) -> "Any":
        """Convert ``state.macro_coords`` to a torch tensor of centres.

        Output dtype / device match ``benchmark.macro_positions`` so the
        evaluator sees a drop-in replacement.

        Subtle point about reshaped soft macros: the worker's reshape
        mutation preserves the macro *centre* but changes ``(w, h)``.
        The evaluator downstream uses ``benchmark.macro_sizes`` (the
        *original* dimensions) together with our reported centres, so a
        soft macro whose optimised centre sits within the canvas under
        its post-reshape dims may still extend past the canvas under
        the original dims.  We therefore clamp each soft macro's centre
        to ``[orig_w/2, canvas_w - orig_w/2]`` (and similarly on Y) so
        the final placement is guaranteed inside the canvas perimeter
        irrespective of any intermediate reshape.

        For *fixed* macros we deliberately echo back the benchmark's
        original centres so a buggy worker that accidentally moved one
        cannot break the validity check on that account.
        """
        N = int(benchmark.num_macros)
        if state.num_macros != N:
            raise RuntimeError(
                f"winner state has num_macros={state.num_macros}, "
                f"benchmark expects {N}"
            )

        # State-side optimised centres (uses state's possibly-reshaped dims).
        centers = bottom_left_to_centers(state.macro_coords, state.macro_dims)

        # Clamp every macro centre against its *original* benchmark size
        # so the canvas-containment property holds at evaluation time.
        orig_dims = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
        if orig_dims.shape == (N, 2):
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            half_w = 0.5 * orig_dims[:, 0]
            half_h = 0.5 * orig_dims[:, 1]
            np.clip(centers[:, 0], half_w, np.maximum(half_w, cw - half_w),
                    out=centers[:, 0])
            np.clip(centers[:, 1], half_h, np.maximum(half_h, ch - half_h),
                    out=centers[:, 1])

        # Overwrite fixed-macro centres with the benchmark's originals.
        if benchmark.macro_fixed is not None:
            fixed_np = benchmark.macro_fixed.detach().cpu().numpy().astype(bool, copy=False)
            if fixed_np.shape == (N,):
                orig_centers = benchmark.macro_positions.detach().cpu().numpy()
                centers[fixed_np] = orig_centers[fixed_np]

        ref = benchmark.macro_positions
        out = torch.tensor(
            np.ascontiguousarray(centers, dtype=np.float64),
            dtype=ref.dtype, device=ref.device,
        )
        return out


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 5. Standalone CLI (cursor.md §G)                                       ║
# ╚════════════════════════════════════════════════════════════════════════╝
#
# This block runs only when ``python submissions/fast_mcmc/main.py`` is
# invoked *directly* (not through the evaluator harness).  It exposes the
# three required flags and emits an ``evaluate``-style summary line so
# CI scripts can grep the result.

def _install_macro_place_stub() -> "Any":
    """Make ``macro_place.benchmark`` importable without triggering the
    ``absl``-heavy ``macro_place/__init__.py``.

    The ``Benchmark`` dataclass pickles to ``(macro_place.benchmark,
    Benchmark)`` – so we have to register the *exact* canonical names in
    ``sys.modules`` for pool workers to be able to unpickle the object
    they receive over the IPC channel.  We install a no-op stub for the
    parent package ``macro_place`` and load only the leaf
    ``benchmark.py`` module from disk.  Fork-mode child processes
    inherit this ``sys.modules`` entry transparently; spawn-mode
    children would need the same stub re-installed (this CLI path
    targets fork-mode platforms).
    """
    import sys as _sys
    import types as _types
    import importlib.util as _ilu

    canonical = "macro_place.benchmark"
    if canonical in _sys.modules:
        return _sys.modules[canonical]

    # Stub the parent package so ``macro_place.benchmark`` resolves
    # without triggering the heavy __init__.py.  ``__path__`` must point
    # at the real directory so :func:`importlib.import_module` can find
    # the submodule when re-resolving the pickled class on either side.
    mp_dir = Path(__file__).resolve().parents[2] / "macro_place"
    pkg = _sys.modules.get("macro_place")
    if pkg is None:
        pkg = _types.ModuleType("macro_place")
        pkg.__path__ = [str(mp_dir)]
        _sys.modules["macro_place"] = pkg

    bm_path = mp_dir / "benchmark.py"
    spec = _ilu.spec_from_file_location(
        canonical, str(bm_path),
        submodule_search_locations=None,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {canonical} from {bm_path}")
    mod = _ilu.module_from_spec(spec)
    # Register *before* exec so any "from macro_place.benchmark import …"
    # triggered transitively finds the in-flight module.
    _sys.modules[canonical] = mod
    spec.loader.exec_module(mod)
    pkg.benchmark = mod  # type: ignore[attr-defined]

    # Sanity-check pickle resolution: pickle's ``whichmodule`` will look
    # up ``sys.modules[module_name]`` and verify the class is reachable
    # by its dotted name.  Force-register the class attribute now.
    if not hasattr(mod, "Benchmark"):
        raise RuntimeError(
            f"loaded {canonical} from {bm_path} but no Benchmark class "
            "was defined – is the file corrupted?"
        )
    return mod


def _cli_load_benchmark(name: str):
    """Try the live ``macro_place.loader`` first, fall back to the
    cached ``.pt`` snapshot so the CLI works in slim environments that
    lack ``absl-py`` (the loader's dependency).
    """
    # Live loader path (preferred when available).
    try:
        from macro_place.loader import load_benchmark_from_dir  # type: ignore
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if root.exists():
            bench, _plc = load_benchmark_from_dir(str(root))
            return bench
    except Exception:
        pass

    # Cached snapshot path – register the canonical module name first
    # so the returned ``Benchmark`` instance picks up the correct
    # ``__module__`` for downstream pickling across process boundaries.
    cached = Path(__file__).resolve().parents[2] / "benchmarks" / "processed" / "public" / f"{name}.pt"
    if not cached.exists():
        raise FileNotFoundError(
            f"cannot find benchmark {name!r} – tried the live loader and "
            f"{cached}"
        )
    mod = _install_macro_place_stub()
    return mod.Benchmark.load(str(cached))


def _format_cli_summary(
    name: str, placement, benchmark, runtime: float, outcome: _PoolOutcome,
) -> str:
    """Compose a one-line ``evaluate``-style result summary."""
    # Approximate per-axis canvas overshoot count for a quick sanity check.
    try:
        import torch  # noqa: F401
        coords = placement.detach().cpu().numpy()
    except Exception:
        coords = np.asarray(placement)
    dims = benchmark.macro_sizes.detach().cpu().numpy()
    x_lo = coords[:, 0] - 0.5 * dims[:, 0]
    y_lo = coords[:, 1] - 0.5 * dims[:, 1]
    x_hi = x_lo + dims[:, 0]
    y_hi = y_lo + dims[:, 1]
    n_oob = int(np.count_nonzero(
        (x_lo < -1e-6) | (y_lo < -1e-6)
        | (x_hi > benchmark.canvas_width  + 1e-6)
        | (y_hi > benchmark.canvas_height + 1e-6)
    ))

    if outcome.best_valid_index >= 0:
        win = outcome.results[outcome.best_valid_index]
        status = "VALID"
    elif outcome.best_index >= 0:
        win = outcome.results[outcome.best_index]
        status = f"INVALID ({', '.join(win.violations) if win.violations else 'see worker'})"
    else:
        win = None
        status = "NO RESULT"

    cost_str = f"cost={win.cost_proxy:.4f}" if win else "cost=—"
    wl_str   = f"wl={win.cost_wirelength:.3f}"  if win else "wl=—"
    d_str    = f"den={win.cost_density:.3f}"   if win else "den=—"
    c_str    = f"cong={win.cost_congestion:.3f}" if win else "cong=—"
    iters_str = f"iters={win.iterations:,}" if win else "iters=—"

    return (
        f"  {name}  {cost_str}  ({wl_str} {d_str} {c_str})  "
        f"{iters_str}  oob={n_oob}  {status}  [{runtime:.2f}s, "
        f"{outcome.num_workers}w]"
    )


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fast_mcmc",
        description=(
            "Standalone driver for the fast_mcmc parallel macro placer. "
            "Loads a benchmark, fans MCMC workers out onto a multiprocessing "
            "pool, picks the lowest-cost valid layout, and prints an "
            "evaluate-style summary line."
        ),
    )
    parser.add_argument(
        "--benchmark", "-b", type=str, default="ibm01",
        help="Benchmark name (e.g. ibm01, ibm03).  Default: ibm01.",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_NUM_WORKERS,
        help=(
            "Maximum worker pool size.  0 ⇒ min(cpu_count, 16).  "
            "1 ⇒ fully serial (useful for step-through debugging)."
        ),
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Per-worker wall-clock budget in seconds.  cursor.md §G calls "
            "out 30 s for verification runs and 3600 s for tournaments."
        ),
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=DEFAULT_SEED,
        help="Seed of the first worker; worker i uses seed + i.",
    )
    parser.add_argument(
        "--no-grasp", action="store_true",
        help="Skip GRASP construction (workers start from benchmark layout).",
    )
    parser.add_argument(
        "--no-prewarm", action="store_true",
        help="Skip parent-side Numba pre-warm (useful for cold-cache timings).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-worker summary table after the pool finishes.",
    )
    args = parser.parse_args(argv)

    # Load benchmark on the parent.
    try:
        benchmark = _cli_load_benchmark(args.benchmark)
    except Exception as exc:
        print(f"[fast_mcmc] failed to load benchmark {args.benchmark!r}: {exc}",
              file=sys.stderr)
        return 2

    n_workers = _resolve_num_workers(args.workers)
    overrides = WorkerConfig(enable_grasp=not args.no_grasp)

    print("=" * 80)
    print(f"fast_mcmc · benchmark={args.benchmark} workers={n_workers} "
          f"timeout={args.timeout:.1f}s seed_base={args.seed}")
    print("=" * 80)

    t0 = time.perf_counter()
    outcome = _run_pool(
        benchmark,
        num_workers=n_workers,
        seed_base=args.seed,
        timeout_seconds=args.timeout,
        overrides=overrides,
        verbose=args.verbose,
        prewarm=not args.no_prewarm,
    )
    runtime = time.perf_counter() - t0

    # Build the final placement tensor exactly the way the evaluator
    # would consume it.  Going through ``FastMCMCPlacer`` is more
    # honest than poking ``outcome.results`` directly because it
    # exercises the fixed-macro echo-back and dtype handling.
    placer = FastMCMCPlacer.__new__(FastMCMCPlacer)
    placer.timeout_seconds = args.timeout
    placer.num_workers     = n_workers
    placer.seed_base       = args.seed
    placer.verbose         = args.verbose
    placer.overrides       = overrides
    placer.last_outcome    = outcome
    winner_state, _winner_valid, _winner_seed = placer._select_winner(outcome, benchmark)
    try:
        import torch
        placement = placer._state_to_center_tensor(winner_state, benchmark, torch)
    except Exception:
        # torch may be unavailable in extra-slim envs; fall back to numpy
        # for the printout (we still complain about the missing dep).
        centers = bottom_left_to_centers(winner_state.macro_coords, winner_state.macro_dims)
        placement = centers

    print(_format_cli_summary(args.benchmark, placement, benchmark, runtime, outcome))

    if outcome.best_valid_index >= 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_cli_main())
