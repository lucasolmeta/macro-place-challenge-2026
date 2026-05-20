"""
modified_annealing.state
===============

Data-oriented memory architect for the high-performance MCMC macro placer.

This module is the *single source of truth* for the in-memory layout consumed
by every other file in the submission (``fast_eval``, ``initialization``,
``worker``, ``main``).  It deliberately avoids object-oriented modelling of
individual macros, nets, or grid cells.  Instead, all state is packed into
flat, contiguous ``numpy`` arrays so that every downstream routine can be
expressed as tight, vectorised, JIT-friendly loops with zero Python pointer
chasing.

Mathematical conventions
------------------------
Let

* :math:`N`  = total number of macros (``num_macros``)
* :math:`N_h`= number of hard macros (``num_hard_macros``)
* :math:`N_s`= number of soft macros (``num_soft_macros``), with
  :math:`N = N_h + N_s`
* :math:`P`  = number of fixed I/O ports
* :math:`M`  = number of nets in the hypergraph netlist

Hard macros occupy indices :math:`[0, N_h)`, soft macros occupy
:math:`[N_h, N)`, and I/O ports occupy the *virtual* index range
:math:`[N, N+P)` so that any pin in a net is identified by a single
``int32`` ``node_id``.

Macro positions
~~~~~~~~~~~~~~~
The external harness (``macro_place.benchmark.Benchmark``) stores macro
positions as **centres** :math:`(x_c, y_c)`.  The MCMC engine, however,
operates on **bottom-left** coordinates :math:`(x_\ell, y_\ell)` so that
bounding-box intersection tests collapse to simple ``min``/``max`` comparisons.
The conversion is

.. math::

    x_\ell = x_c - \tfrac{w}{2}, \qquad y_\ell = y_c - \tfrac{h}{2}

and is inverted when emitting the final placement back to the evaluator.

Netlist Compressed Sparse Row (CSR)
-----------------------------------
The hypergraph netlist is stored twice, in both directions of traversal,
because the inner Metropolis loop needs both:

1. **Net → pins** (used by HPWL recomputation when *any* pin of a net moves)

   * ``net_pin_owners``  : ``int32[total_pins]``      flat node ids
   * ``net_pin_offsets`` : ``int32[M + 1]``           cumulative offsets

   The pins of net :math:`k` are
   ``net_pin_owners[net_pin_offsets[k] : net_pin_offsets[k+1]]``.

2. **Macro → nets** (used by a mutation to find the affected nets in
   :math:`\mathcal{O}(\deg(\text{macro}))`)

   * ``macro_net_ids``     : ``int32[total_macro_pins]``
   * ``macro_net_offsets`` : ``int32[N + 1]``

   The nets touched by macro :math:`i` are
   ``macro_net_ids[macro_net_offsets[i] : macro_net_offsets[i+1]]``.

The reverse map deduplicates within a single net so each macro appears at
most once per net (matching the Benchmark.net_nodes semantics).

Spatial collision grid
----------------------
``spatial_grid`` is a 2-D ``int32`` matrix of shape
``(grid_num_rows, grid_num_cols)`` that partitions the canvas into uniform
rectangular bins of size
``(grid_bin_width, grid_bin_height)``.  Every cell stores the integer
``Macro_ID`` currently occupying that spatial zone, or the sentinel
``EMPTY_CELL = -1`` for empty space.

Given a macro with bottom-left :math:`(x_\ell, y_\ell)` and dimensions
:math:`(w, h)`, the bin range it covers is

.. math::

    c_\text{lo} = \lfloor x_\ell / b_w \rfloor, \qquad
    c_\text{hi} = \lceil (x_\ell + w) / b_w \rceil - 1
.. math::

    r_\text{lo} = \lfloor y_\ell / b_h \rfloor, \qquad
    r_\text{hi} = \lceil (y_\ell + h) / b_h \rceil - 1

where :math:`b_w, b_h` are the bin dimensions.  Collision detection is then
a constant-cost lookup over the contiguous sub-matrix
``spatial_grid[r_lo:r_hi+1, c_lo:c_hi+1]``: if any cell holds a value other
than ``EMPTY_CELL`` or the moving macro's own id, the candidate move
overlaps an existing macro and is rejected.

This file performs *only* parsing and memory allocation.  No MCMC or cost
mathematics live here; those belong in ``fast_eval.py`` and ``worker.py``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ─── Public constants ────────────────────────────────────────────────────────

EMPTY_CELL: np.int32 = np.int32(-1)
"""Sentinel value written into every cell of ``spatial_grid`` to denote
unoccupied space.  Chosen as ``-1`` because all valid macro ids are
non-negative and an unsigned comparison ``cell != EMPTY_CELL`` therefore
also detects collisions in a single CPU instruction."""

DEFAULT_GRID_MAX_DIM: int = 1024
"""Hard cap on the number of bins along either axis of the spatial grid.
Prevents pathological memory blow-up on benchmarks with very small macros
or very large canvases.  Tuned so the grid fits comfortably in L2 cache
(``1024 * 1024 * 4`` B = 4 MiB)."""

DEFAULT_BIN_OVERSAMPLE: float = 1.0
"""Multiplicative factor applied to the *smallest hard macro dimension*
when auto-selecting the default bin size.  A value of ``1.0`` makes every
hard macro span at least one bin; values ``< 1.0`` make the grid finer
(more accurate but more memory and slower paint/clear)."""

EPS: float = 1e-9
"""Numerical safety margin used when rounding bin indices to avoid
spuriously claiming an extra bin on the high edge of a macro."""


# ─── PlacementState container ────────────────────────────────────────────────

@dataclass
class PlacementState:
    """Flat, data-oriented representation of one placement instance.

    Every attribute is a plain Python scalar, a Python ``list`` of strings
    (debug-only ``macro_names``), or a contiguous ``numpy.ndarray``.  There
    are no ``Macro`` or ``Net`` objects, no nested ``dict``s, and no
    references back to ``torch`` tensors.

    All ``np.ndarray`` attributes are guaranteed to be ``C``-contiguous so
    that they can be safely passed to ``@njit``-compiled kernels and
    pickled across multiprocessing boundaries with zero conversion cost.
    """

    # ── identity ────────────────────────────────────────────────────────
    name: str

    # ── canvas dimensions (microns) ─────────────────────────────────────
    canvas_width: float
    canvas_height: float

    # ── macros (hard first, then soft) ──────────────────────────────────
    num_macros: int
    num_hard_macros: int
    num_soft_macros: int

    macro_coords: np.ndarray   # (N, 2) float64 – bottom-left (x_l, y_l)
    macro_dims:   np.ndarray   # (N, 2) float64 – (width,  height)
    macro_fixed:  np.ndarray   # (N,)   bool    – True if pinned
    macro_is_hard: np.ndarray  # (N,)   bool    – True for indices [0, N_h)
    macro_names:  List[str]    # length N – diagnostic only

    # ── I/O ports (fixed, no dimensions) ────────────────────────────────
    num_ports: int
    port_coords: np.ndarray    # (P, 2) float64 – pin location

    # ── netlist (forward CSR: net → pins) ───────────────────────────────
    num_nets: int
    net_pin_owners:  np.ndarray  # (total_pins,)   int32  – node ids
    net_pin_offsets: np.ndarray  # (num_nets + 1,) int32
    net_weights:     np.ndarray  # (num_nets,)     float64

    # ── netlist (reverse CSR: macro → nets) ─────────────────────────────
    macro_net_ids:     np.ndarray  # (total_macro_net_pins,) int32
    macro_net_offsets: np.ndarray  # (num_macros + 1,)       int32

    # ── cached half-perimeter wirelength bounding box per net ───────────
    # Layout: net_bbox[k] = [min_x, max_x, min_y, max_y].  Maintained
    # incrementally by fast_eval.update_net_bbox_after_move.
    net_bbox: np.ndarray  # (num_nets, 4) float64

    # ── spatial collision grid ──────────────────────────────────────────
    grid_bin_width:  float
    grid_bin_height: float
    grid_num_cols:   int
    grid_num_rows:   int
    spatial_grid:    np.ndarray  # (rows, cols) int32, sentinel EMPTY_CELL

    # ── benchmark-defined grid (used by the reference proxy cost) ───────
    bench_grid_rows: int
    bench_grid_cols: int

    # ── canvas-derived convenience constants (cached for hot kernels) ───
    inv_bin_width:  float = 0.0
    inv_bin_height: float = 0.0


# ─── Coordinate convention conversions ───────────────────────────────────────

def centers_to_bottom_left(centers: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Convert ``(x_c, y_c)`` centres to ``(x_l, y_l)`` bottom-left.

    .. math:: (x_\\ell, y_\\ell) = (x_c, y_c) - \\tfrac{1}{2}(w, h)

    Args:
        centers: ``(N, 2)`` float array of macro centre coordinates.
        dims:    ``(N, 2)`` float array of macro ``(width, height)``.

    Returns:
        A freshly-allocated ``(N, 2)`` ``float64`` array of bottom-left
        coordinates.
    """
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    dims = np.ascontiguousarray(dims, dtype=np.float64)
    return np.subtract(centers, 0.5 * dims, dtype=np.float64)


def bottom_left_to_centers(bottom_left: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Inverse of :func:`centers_to_bottom_left`.

    .. math:: (x_c, y_c) = (x_\\ell, y_\\ell) + \\tfrac{1}{2}(w, h)
    """
    bottom_left = np.ascontiguousarray(bottom_left, dtype=np.float64)
    dims = np.ascontiguousarray(dims, dtype=np.float64)
    return np.add(bottom_left, 0.5 * dims, dtype=np.float64)


# ─── Spatial grid helpers ────────────────────────────────────────────────────

def compute_default_bin_size(
    macro_dims: np.ndarray,
    num_hard_macros: int,
    canvas_width: float,
    canvas_height: float,
    *,
    oversample: float = DEFAULT_BIN_OVERSAMPLE,
    max_dim: int = DEFAULT_GRID_MAX_DIM,
) -> Tuple[float, float]:
    """Pick reasonable ``(bin_width, bin_height)`` defaults.

    The bin must be small enough that the smallest hard macro spans at
    least one bin (so collisions are detectable), and large enough that
    the resulting grid does not exceed ``max_dim`` cells per axis (so it
    fits in cache).

    Args:
        macro_dims: ``(N, 2)`` ``float64`` array of macro ``(W, H)``.
        num_hard_macros: number of hard macros (the leading slice of
            ``macro_dims``).  Soft macros are ignored when computing the
            minimum because they may legitimately be tiny standard-cell
            clusters and would otherwise force a wastefully fine grid.
        canvas_width, canvas_height: canvas extents in microns.
        oversample: multiplicative factor on the smallest hard macro
            dimension.  ``1.0`` is the most conservative (guaranteed
            coverage), values ``< 1`` give finer grids.
        max_dim: hard cap on the number of bins per axis.

    Returns:
        ``(bin_width, bin_height)`` in microns.
    """
    if num_hard_macros <= 0:
        # Degenerate benchmark with no hard macros: use the canvas diagonal
        # as a coarse fallback so the grid is still well-defined.
        diag = math.hypot(canvas_width, canvas_height)
        return diag, diag

    hard_dims = macro_dims[:num_hard_macros]
    min_w_hard = float(hard_dims[:, 0].min())
    min_h_hard = float(hard_dims[:, 1].min())

    # Avoid zero/negative bin sizes from degenerate inputs.
    bin_w = max(min_w_hard * oversample, canvas_width  / max_dim, EPS)
    bin_h = max(min_h_hard * oversample, canvas_height / max_dim, EPS)
    return bin_w, bin_h


def build_spatial_grid(
    canvas_width: float,
    canvas_height: float,
    bin_width: float,
    bin_height: float,
) -> Tuple[np.ndarray, int, int]:
    """Allocate the empty ``spatial_grid`` matrix.

    Every cell is pre-populated with :data:`EMPTY_CELL` to signal empty
    space.  The matrix is shaped ``(num_rows, num_cols)`` with

    .. math::

        \\text{num\\_cols} = \\left\\lceil \\frac{W_\\text{canvas}}{b_w} \\right\\rceil, \\qquad
        \\text{num\\_rows} = \\left\\lceil \\frac{H_\\text{canvas}}{b_h} \\right\\rceil .

    The matrix is stored ``row-major`` (``C``-contiguous) so iteration
    over a horizontal sub-strip is cache-friendly, matching the typical
    access pattern of bounding-box paint/clear loops.

    Returns:
        ``(grid, num_rows, num_cols)``.
    """
    if bin_width <= 0 or bin_height <= 0:
        raise ValueError(
            f"bin dimensions must be positive (got bin_width={bin_width}, "
            f"bin_height={bin_height})"
        )
    num_cols = max(1, int(math.ceil(canvas_width  / bin_width)))
    num_rows = max(1, int(math.ceil(canvas_height / bin_height)))
    grid = np.full((num_rows, num_cols), EMPTY_CELL, dtype=np.int32)
    return grid, num_rows, num_cols


def compute_macro_bin_range(
    x_ll: float,
    y_ll: float,
    width: float,
    height: float,
    bin_width: float,
    bin_height: float,
    grid_num_cols: int,
    grid_num_rows: int,
) -> Tuple[int, int, int, int]:
    """Map a macro's continuous bbox to an inclusive grid index range.

    Uses the half-open interval convention so that two macros touching
    exactly on a shared edge (e.g. one at ``x ∈ [0, 10)`` and another at
    ``x ∈ [10, 20)``) do not share any bin.

    Args:
        x_ll, y_ll: bottom-left coordinates of the macro.
        width, height: macro dimensions.
        bin_width, bin_height: spatial grid bin dimensions.
        grid_num_cols, grid_num_rows: spatial grid extents.

    Returns:
        ``(c_lo, c_hi, r_lo, r_hi)``, all inclusive and clamped to the
        valid grid range ``[0, grid_num_cols - 1]`` / ``[0, grid_num_rows - 1]``.
    """
    x_hi = x_ll + width
    y_hi = y_ll + height

    c_lo = int(math.floor(x_ll / bin_width))
    c_hi = int(math.ceil((x_hi - EPS) / bin_width)) - 1
    r_lo = int(math.floor(y_ll / bin_height))
    r_hi = int(math.ceil((y_hi - EPS) / bin_height)) - 1

    if c_lo < 0:                   c_lo = 0
    if r_lo < 0:                   r_lo = 0
    if c_hi >= grid_num_cols:      c_hi = grid_num_cols - 1
    if r_hi >= grid_num_rows:      r_hi = grid_num_rows - 1
    if c_hi < c_lo:                c_hi = c_lo
    if r_hi < r_lo:                r_hi = r_lo
    return c_lo, c_hi, r_lo, r_hi


def stamp_macro_on_grid(state: PlacementState, macro_idx: int) -> None:
    """Write ``macro_idx`` into every spatial bin covered by macro ``macro_idx``.

    No collision check is performed; the caller is responsible for
    ensuring that the target region is empty (or that overwrites are
    intentional, as in the exploration phase of the MCMC loop).
    """
    x_ll, y_ll = float(state.macro_coords[macro_idx, 0]), float(state.macro_coords[macro_idx, 1])
    w,    h    = float(state.macro_dims[macro_idx, 0]),  float(state.macro_dims[macro_idx, 1])
    c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range(
        x_ll, y_ll, w, h,
        state.grid_bin_width, state.grid_bin_height,
        state.grid_num_cols, state.grid_num_rows,
    )
    state.spatial_grid[r_lo:r_hi + 1, c_lo:c_hi + 1] = np.int32(macro_idx)


def clear_macro_on_grid(state: PlacementState, macro_idx: int) -> None:
    """Erase ``macro_idx`` from the spatial grid (reset cells to ``EMPTY_CELL``).

    Only cells that currently hold ``macro_idx`` are cleared, so this is
    safe even when called speculatively on a stale bounding box.
    """
    x_ll, y_ll = float(state.macro_coords[macro_idx, 0]), float(state.macro_coords[macro_idx, 1])
    w,    h    = float(state.macro_dims[macro_idx, 0]),  float(state.macro_dims[macro_idx, 1])
    c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range(
        x_ll, y_ll, w, h,
        state.grid_bin_width, state.grid_bin_height,
        state.grid_num_cols, state.grid_num_rows,
    )
    sub = state.spatial_grid[r_lo:r_hi + 1, c_lo:c_hi + 1]
    np.place(sub, sub == np.int32(macro_idx), EMPTY_CELL)


def stamp_all_hard_macros(state: PlacementState) -> None:
    """Imprint every (movable + fixed) **hard** macro onto an empty grid.

    Soft macros are excluded because the challenge only requires zero
    overlaps between hard macros; soft cells are allowed to occupy the
    same bins as one another and as hard macros.

    The grid is reset to :data:`EMPTY_CELL` first, so this routine is
    idempotent and safe to call repeatedly during initialization sweeps.
    """
    state.spatial_grid.fill(EMPTY_CELL)
    for i in range(state.num_hard_macros):
        stamp_macro_on_grid(state, i)


# ─── HPWL bounding-box cache ─────────────────────────────────────────────────

def _node_center(
    node_id: int,
    macro_coords: np.ndarray,
    macro_dims: np.ndarray,
    port_coords: np.ndarray,
    num_macros: int,
) -> Tuple[float, float]:
    """Return the ``(x, y)`` *centre* of a generic netlist node.

    Macro nodes (``node_id < num_macros``) report their centre by adding
    half of their dimensions to the stored bottom-left.  Port nodes
    (``node_id >= num_macros``) are zero-dimensional and report their
    fixed pin location directly.
    """
    if node_id < num_macros:
        x = macro_coords[node_id, 0] + 0.5 * macro_dims[node_id, 0]
        y = macro_coords[node_id, 1] + 0.5 * macro_dims[node_id, 1]
        return float(x), float(y)
    port_id = node_id - num_macros
    return float(port_coords[port_id, 0]), float(port_coords[port_id, 1])


def compute_net_bbox(state: PlacementState) -> np.ndarray:
    """Recompute the per-net bounding-box cache from scratch.

    Each row ``k`` of the returned array stores
    ``[min_x, max_x, min_y, max_y]`` over the *centres* of all pins
    (macro centres and port pin locations) belonging to net ``k``.

    The Half-Perimeter Wirelength (HPWL) of net ``k`` is then

    .. math::

        \\text{HPWL}(k) = (\\max_x - \\min_x) + (\\max_y - \\min_y)

    which is what ``fast_eval`` uses to compute exact wirelength deltas
    when a macro moves.  Empty nets (which the loader should already
    have dropped) are encoded with all zeros.
    """
    M = state.num_nets
    bbox = np.zeros((M, 4), dtype=np.float64)
    owners  = state.net_pin_owners
    offsets = state.net_pin_offsets
    mc = state.macro_coords
    md = state.macro_dims
    pc = state.port_coords
    Nm = state.num_macros

    for k in range(M):
        s = int(offsets[k])
        e = int(offsets[k + 1])
        if e <= s:
            continue
        min_x =  math.inf
        max_x = -math.inf
        min_y =  math.inf
        max_y = -math.inf
        for p in range(s, e):
            nid = int(owners[p])
            x, y = _node_center(nid, mc, md, pc, Nm)
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        bbox[k, 0] = min_x
        bbox[k, 1] = max_x
        bbox[k, 2] = min_y
        bbox[k, 3] = max_y
    return bbox


# ─── CSR netlist construction ────────────────────────────────────────────────

def _as_int32_node_array(net_nodes_list: Sequence) -> List[np.ndarray]:
    """Normalize the heterogeneous ``Benchmark.net_nodes`` list to ``int32``.

    The loader produces ``torch.LongTensor``s of arbitrary length; we
    extract them into a list of contiguous ``int32`` numpy arrays so the
    downstream vectorised CSR build can use a single ``np.concatenate``.
    """
    out: List[np.ndarray] = []
    for nodes in net_nodes_list:
        if hasattr(nodes, "cpu"):                     # torch.Tensor
            arr = nodes.detach().cpu().numpy()
        else:
            arr = np.asarray(nodes)
        out.append(np.ascontiguousarray(arr, dtype=np.int32))
    return out


def build_csr_netlist(
    net_nodes_list: Sequence,
    num_macros: int,
    num_total_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forward and reverse CSR representations of the netlist hypergraph.

    Args:
        net_nodes_list: sequence of length ``num_nets``; each entry is an
            iterable of integer node ids belonging to that net.  Node ids
            in ``[0, num_macros)`` are macros and ids in
            ``[num_macros, num_total_nodes)`` are I/O ports.
        num_macros: number of macros (``N_h + N_s``).
        num_total_nodes: ``num_macros + num_ports``.  Any node id outside
            ``[0, num_total_nodes)`` is treated as a programming error.

    Returns:
        ``(net_pin_owners, net_pin_offsets, macro_net_ids, macro_net_offsets)``
        as documented at module top.
    """
    nets = _as_int32_node_array(net_nodes_list)
    num_nets = len(nets)

    # ── Forward CSR: net → pins ─────────────────────────────────────────
    pin_counts = np.fromiter(
        (a.size for a in nets), dtype=np.int64, count=num_nets,
    ) if num_nets > 0 else np.zeros(0, dtype=np.int64)

    net_pin_offsets = np.zeros(num_nets + 1, dtype=np.int32)
    if num_nets > 0:
        cum = pin_counts.cumsum()
        if cum[-1] > np.iinfo(np.int32).max:
            raise OverflowError(
                f"total pin count {int(cum[-1])} exceeds int32 range; "
                "increase index dtype in state.py"
            )
        net_pin_offsets[1:] = cum.astype(np.int32)

    total_pins = int(net_pin_offsets[-1]) if num_nets > 0 else 0
    if total_pins > 0:
        net_pin_owners = np.concatenate(nets).astype(np.int32, copy=False)
    else:
        net_pin_owners = np.zeros(0, dtype=np.int32)

    # Validate range of node ids.
    if total_pins > 0:
        min_id = int(net_pin_owners.min())
        max_id = int(net_pin_owners.max())
        if min_id < 0 or max_id >= num_total_nodes:
            raise ValueError(
                f"net pin owner out of range: min={min_id}, max={max_id}, "
                f"valid=[0, {num_total_nodes})"
            )

    # ── Reverse CSR: macro → nets ───────────────────────────────────────
    # Build (macro_id, net_id) pairs for every macro pin, deduplicate so
    # each (macro, net) appears at most once, then bucket-sort by macro.
    if total_pins > 0 and num_macros > 0:
        # Flat per-pin net id matching net_pin_owners.
        pin_net_ids = np.repeat(
            np.arange(num_nets, dtype=np.int32), pin_counts,
        )
        macro_mask = net_pin_owners < np.int32(num_macros)
        m_owners = net_pin_owners[macro_mask].astype(np.int64, copy=False)
        m_nids   = pin_net_ids[macro_mask].astype(np.int64,   copy=False)

        # Encode (owner, net) pair as a single int64 key so np.unique
        # gives lexicographic uniqueness in one vectorised pass.
        stride = np.int64(num_nets) + np.int64(1)
        keys = m_owners * stride + m_nids
        unique_keys = np.unique(keys)
        sorted_owners = (unique_keys // stride).astype(np.int32, copy=False)
        sorted_nids   = (unique_keys %  stride).astype(np.int32, copy=False)

        macro_net_counts = np.bincount(
            sorted_owners.astype(np.int64), minlength=num_macros,
        )
        macro_net_offsets = np.zeros(num_macros + 1, dtype=np.int32)
        cum_m = macro_net_counts.cumsum()
        if cum_m[-1] > np.iinfo(np.int32).max:
            raise OverflowError(
                f"total macro-net pin count {int(cum_m[-1])} exceeds int32; "
                "increase index dtype in state.py"
            )
        macro_net_offsets[1:] = cum_m.astype(np.int32)
        # sorted_nids is already in macro-major, net-minor order thanks to
        # the lexicographic sort, so it IS the CSR value array.
        macro_net_ids = sorted_nids
    else:
        macro_net_ids     = np.zeros(0, dtype=np.int32)
        macro_net_offsets = np.zeros(num_macros + 1, dtype=np.int32)

    return net_pin_owners, net_pin_offsets, macro_net_ids, macro_net_offsets


# ─── Top-level builder ───────────────────────────────────────────────────────

def build_state(
    benchmark,
    *,
    bin_width:  Optional[float] = None,
    bin_height: Optional[float] = None,
    bin_oversample: float = DEFAULT_BIN_OVERSAMPLE,
    max_grid_dim: int = DEFAULT_GRID_MAX_DIM,
    stamp_hard_macros: bool = True,
) -> PlacementState:
    """Convert a ``macro_place.benchmark.Benchmark`` into a ``PlacementState``.

    Performs all parsing, dtype normalization, CSR construction, spatial
    grid allocation, and HPWL bbox cache initialization in one pass.

    Args:
        benchmark: a fully populated ``macro_place.benchmark.Benchmark``
            instance (as produced by ``load_benchmark_from_dir``).  We do
            not import the class symbolically to keep this module
            standalone and pickleable across processes.
        bin_width, bin_height: override the auto-selected spatial grid
            bin size.  When ``None`` (the default) a sensible value is
            derived via :func:`compute_default_bin_size`.
        bin_oversample: see :func:`compute_default_bin_size`.
        max_grid_dim: see :func:`compute_default_bin_size`.
        stamp_hard_macros: when ``True`` (default), the initial macro
            positions are immediately painted onto the spatial grid via
            :func:`stamp_all_hard_macros` so the returned state is ready
            for collision queries.  Set to ``False`` if you intend to
            replace the placement (e.g. inside a GRASP worker) and want
            to start from a pristine empty grid.

    Returns:
        A fully populated, self-consistent :class:`PlacementState`.
    """
    # ── basic scalars ───────────────────────────────────────────────────
    name = str(benchmark.name)
    canvas_width  = float(benchmark.canvas_width)
    canvas_height = float(benchmark.canvas_height)
    num_macros      = int(benchmark.num_macros)
    num_hard_macros = int(benchmark.num_hard_macros)
    num_soft_macros = int(benchmark.num_soft_macros)

    if num_hard_macros + num_soft_macros != num_macros:
        raise ValueError(
            f"benchmark macro counts inconsistent: "
            f"hard={num_hard_macros} + soft={num_soft_macros} != "
            f"total={num_macros}"
        )

    # ── macro arrays (torch → numpy → bottom-left float64) ──────────────
    centers_np = benchmark.macro_positions.detach().cpu().numpy()
    dims_np    = benchmark.macro_sizes.detach().cpu().numpy()
    fixed_np   = benchmark.macro_fixed.detach().cpu().numpy().astype(bool, copy=False)

    if centers_np.shape != (num_macros, 2):
        raise ValueError(
            f"macro_positions shape {centers_np.shape} != ({num_macros}, 2)"
        )
    if dims_np.shape != (num_macros, 2):
        raise ValueError(
            f"macro_sizes shape {dims_np.shape} != ({num_macros}, 2)"
        )

    macro_dims   = np.ascontiguousarray(dims_np, dtype=np.float64)
    macro_coords = centers_to_bottom_left(centers_np, macro_dims)
    macro_coords = np.ascontiguousarray(macro_coords, dtype=np.float64)
    macro_fixed  = np.ascontiguousarray(fixed_np, dtype=np.bool_)

    macro_is_hard = np.zeros(num_macros, dtype=np.bool_)
    macro_is_hard[:num_hard_macros] = True

    macro_names = list(benchmark.macro_names)

    # ── ports ───────────────────────────────────────────────────────────
    if benchmark.port_positions.numel() > 0:
        port_np = benchmark.port_positions.detach().cpu().numpy()
    else:
        port_np = np.zeros((0, 2), dtype=np.float64)
    port_coords = np.ascontiguousarray(port_np, dtype=np.float64).reshape(-1, 2)
    num_ports = int(port_coords.shape[0])
    num_total_nodes = num_macros + num_ports

    # ── netlist (forward + reverse CSR) ─────────────────────────────────
    # The authoritative net count is the length of the populated
    # ``net_nodes`` list: some pre-cached ``.pt`` snapshots carry only
    # ``num_nets`` metadata without the actual pin connectivity, in which
    # case we operate over zero nets rather than indexing out of bounds.
    declared_num_nets = int(benchmark.num_nets)
    (net_pin_owners,
     net_pin_offsets,
     macro_net_ids,
     macro_net_offsets) = build_csr_netlist(
        benchmark.net_nodes, num_macros, num_total_nodes,
    )
    num_nets = max(0, int(net_pin_offsets.shape[0]) - 1)

    if benchmark.net_weights.numel() > 0:
        weights_np = benchmark.net_weights.detach().cpu().numpy()
    else:
        weights_np = np.ones(declared_num_nets, dtype=np.float64)
    weights_np = np.ascontiguousarray(weights_np, dtype=np.float64).ravel()
    if weights_np.shape[0] >= num_nets:
        net_weights = weights_np[:num_nets].copy()
    else:
        net_weights = np.ones(num_nets, dtype=np.float64)
        net_weights[: weights_np.shape[0]] = weights_np

    # ── spatial grid ────────────────────────────────────────────────────
    if bin_width is None or bin_height is None:
        auto_w, auto_h = compute_default_bin_size(
            macro_dims, num_hard_macros, canvas_width, canvas_height,
            oversample=bin_oversample, max_dim=max_grid_dim,
        )
        if bin_width  is None: bin_width  = auto_w
        if bin_height is None: bin_height = auto_h
    bin_width  = float(bin_width)
    bin_height = float(bin_height)

    spatial_grid, grid_num_rows, grid_num_cols = build_spatial_grid(
        canvas_width, canvas_height, bin_width, bin_height,
    )

    # ── benchmark-defined grid (kept for reference proxy cost) ──────────
    bench_grid_rows = int(getattr(benchmark, "grid_rows", 0) or 0)
    bench_grid_cols = int(getattr(benchmark, "grid_cols", 0) or 0)

    # ── assemble ────────────────────────────────────────────────────────
    state = PlacementState(
        name=name,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        num_macros=num_macros,
        num_hard_macros=num_hard_macros,
        num_soft_macros=num_soft_macros,
        macro_coords=macro_coords,
        macro_dims=macro_dims,
        macro_fixed=macro_fixed,
        macro_is_hard=macro_is_hard,
        macro_names=macro_names,
        num_ports=num_ports,
        port_coords=port_coords,
        num_nets=num_nets,
        net_pin_owners=net_pin_owners,
        net_pin_offsets=net_pin_offsets,
        net_weights=net_weights,
        macro_net_ids=macro_net_ids,
        macro_net_offsets=macro_net_offsets,
        net_bbox=np.zeros((num_nets, 4), dtype=np.float64),
        grid_bin_width=bin_width,
        grid_bin_height=bin_height,
        grid_num_cols=grid_num_cols,
        grid_num_rows=grid_num_rows,
        spatial_grid=spatial_grid,
        bench_grid_rows=bench_grid_rows,
        bench_grid_cols=bench_grid_cols,
        inv_bin_width =1.0 / bin_width,
        inv_bin_height=1.0 / bin_height,
    )

    # ── initial caches ──────────────────────────────────────────────────
    state.net_bbox = compute_net_bbox(state)
    if stamp_hard_macros:
        stamp_all_hard_macros(state)
    return state


# ─── Worker isolation helpers ────────────────────────────────────────────────

# Arrays that workers mutate during MCMC exploration and therefore must
# be deep-copied when forking a fresh state for each worker.
_MUTABLE_ARRAYS: Tuple[str, ...] = (
    "macro_coords",
    "net_bbox",
    "spatial_grid",
)

# Arrays that are constant across the entire run and may safely be shared
# (copy-on-write under multiprocessing.fork).
_STATIC_ARRAYS: Tuple[str, ...] = (
    "macro_dims",
    "macro_fixed",
    "macro_is_hard",
    "port_coords",
    "net_pin_owners",
    "net_pin_offsets",
    "net_weights",
    "macro_net_ids",
    "macro_net_offsets",
)


def clone_state(state: PlacementState, *, deep: bool = True) -> PlacementState:
    """Return an independent copy of ``state`` for use by a worker process.

    With ``deep=True`` (default) every mutable array is copied so the
    worker can perturb ``macro_coords``, ``spatial_grid`` and ``net_bbox``
    in place without disturbing the parent's state.  Static arrays
    (``macro_dims``, the CSR structures, port coords, etc.) are *shared*
    by reference because they are never written to during MCMC, which
    saves both memory and pickle bandwidth across the multiprocessing
    pool.

    With ``deep=False`` every array is shared, producing a near-zero-cost
    shallow clone suitable when only the scalar metadata is of interest.
    """
    new = copy.copy(state)              # shallow clone of all fields
    new.macro_names = list(state.macro_names)
    if deep:
        for attr in _MUTABLE_ARRAYS:
            arr = getattr(new, attr)
            setattr(new, attr, np.array(arr, copy=True, order="C"))
    return new


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def verify_state(state: PlacementState) -> List[str]:
    """Run a battery of cheap structural sanity checks.

    Intended for use in unit tests and as a defensive ``assert`` at
    worker start-up.  Returns a list of human-readable violation
    descriptions; an empty list means the state is well-formed.  Does
    NOT raise so the caller can choose its own error handling.
    """
    violations: List[str] = []
    N = state.num_macros
    M = state.num_nets

    if state.macro_coords.shape != (N, 2):
        violations.append(f"macro_coords shape {state.macro_coords.shape} != ({N}, 2)")
    if state.macro_dims.shape != (N, 2):
        violations.append(f"macro_dims shape {state.macro_dims.shape} != ({N}, 2)")
    if state.macro_fixed.shape != (N,):
        violations.append(f"macro_fixed shape {state.macro_fixed.shape} != ({N},)")
    if state.macro_is_hard.shape != (N,):
        violations.append(f"macro_is_hard shape {state.macro_is_hard.shape} != ({N},)")
    if state.macro_is_hard.sum() != state.num_hard_macros:
        violations.append("macro_is_hard count mismatch with num_hard_macros")

    if state.net_pin_offsets.shape != (M + 1,):
        violations.append(f"net_pin_offsets shape {state.net_pin_offsets.shape} != ({M+1},)")
    if state.macro_net_offsets.shape != (N + 1,):
        violations.append(f"macro_net_offsets shape {state.macro_net_offsets.shape} != ({N+1},)")

    if M > 0:
        if state.net_pin_offsets[0] != 0:
            violations.append("net_pin_offsets[0] != 0")
        if state.net_pin_offsets[-1] != state.net_pin_owners.size:
            violations.append(
                f"net_pin_offsets[-1]={int(state.net_pin_offsets[-1])} != "
                f"net_pin_owners.size={state.net_pin_owners.size}"
            )
        if not np.all(np.diff(state.net_pin_offsets) >= 0):
            violations.append("net_pin_offsets not monotone non-decreasing")

    if N > 0:
        if state.macro_net_offsets[0] != 0:
            violations.append("macro_net_offsets[0] != 0")
        if state.macro_net_offsets[-1] != state.macro_net_ids.size:
            violations.append(
                f"macro_net_offsets[-1]={int(state.macro_net_offsets[-1])} != "
                f"macro_net_ids.size={state.macro_net_ids.size}"
            )

    if state.net_pin_owners.size > 0:
        max_id = int(state.net_pin_owners.max())
        if max_id >= N + state.num_ports:
            violations.append(
                f"net pin owner id {max_id} >= num_macros + num_ports"
                f" ({N + state.num_ports})"
            )

    if state.spatial_grid.shape != (state.grid_num_rows, state.grid_num_cols):
        violations.append(
            f"spatial_grid shape {state.spatial_grid.shape} != "
            f"({state.grid_num_rows}, {state.grid_num_cols})"
        )

    if not (state.canvas_width > 0 and state.canvas_height > 0):
        violations.append("non-positive canvas dimensions")
    if not (state.grid_bin_width > 0 and state.grid_bin_height > 0):
        violations.append("non-positive bin dimensions")

    # Per-macro bbox inside canvas?  (For fixed macros only; movable ones
    # may temporarily violate during exploration.)
    if N > 0:
        x_lo = state.macro_coords[:, 0]
        y_lo = state.macro_coords[:, 1]
        x_hi = x_lo + state.macro_dims[:, 0]
        y_hi = y_lo + state.macro_dims[:, 1]
        fixed = state.macro_fixed
        if np.any(fixed & ((x_lo < -EPS) | (x_hi > state.canvas_width  + EPS))):
            violations.append("fixed macro outside horizontal canvas bounds")
        if np.any(fixed & ((y_lo < -EPS) | (y_hi > state.canvas_height + EPS))):
            violations.append("fixed macro outside vertical canvas bounds")

    return violations


# ─── Quick smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Self-contained smoke test that does not require torch or any
    # external benchmark file.  Run with:  python -m modified_annealing.state
    import types

    class _StubTensor:
        """Minimal stand-in for torch.Tensor exposing only what build_state uses."""
        def __init__(self, arr):
            self._arr = np.ascontiguousarray(arr)
        def detach(self):                  return self
        def cpu(self):                     return self
        def numpy(self):                   return self._arr
        def numel(self):                   return self._arr.size
        @property
        def shape(self):                   return self._arr.shape

    bench = types.SimpleNamespace(
        name="smoke",
        canvas_width=100.0,
        canvas_height=80.0,
        num_macros=3,
        num_hard_macros=2,
        num_soft_macros=1,
        macro_positions=_StubTensor([[10.0, 10.0], [50.0, 30.0], [80.0, 60.0]]),
        macro_sizes    =_StubTensor([[20.0, 20.0], [30.0, 40.0], [10.0, 10.0]]),
        macro_fixed    =_StubTensor([False, False, False]),
        macro_names    =["m0", "m1", "soft0"],
        num_nets=2,
        net_nodes=[np.array([0, 1, 3], dtype=np.int64),       # macros 0,1 + port 0
                   np.array([1, 2],    dtype=np.int64)],     # macro 1 + soft 0
        net_weights=_StubTensor([1.0, 1.0]),
        port_positions=_StubTensor([[0.0, 0.0]]),
        grid_rows=10,
        grid_cols=10,
    )

    st = build_state(bench)
    bad = verify_state(st)
    assert not bad, bad
    print(f"PlacementState OK | macros={st.num_macros} hard={st.num_hard_macros} "
          f"nets={st.num_nets} ports={st.num_ports} "
          f"grid={st.grid_num_rows}x{st.grid_num_cols} "
          f"bin=({st.grid_bin_width:.2f}, {st.grid_bin_height:.2f})")
    print(f"macro_coords[bottom-left]:\n{st.macro_coords}")
    print(f"net_pin_owners: {st.net_pin_owners}")
    print(f"net_pin_offsets: {st.net_pin_offsets}")
    print(f"macro_net_ids:  {st.macro_net_ids}")
    print(f"macro_net_offsets: {st.macro_net_offsets}")
    print(f"net_bbox:\n{st.net_bbox}")
    occupied = int((st.spatial_grid != EMPTY_CELL).sum())
    print(f"occupied grid cells: {occupied} / {st.spatial_grid.size}")
