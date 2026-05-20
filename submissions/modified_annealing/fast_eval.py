"""
modified_annealing.fast_eval
===================

Procedural, Numba-JIT-compiled mathematics pipeline for the MCMC engine.

This module is intentionally **stateless**.  Every function is a free,
``@njit(cache=True)``-decorated kernel that operates exclusively on the
flat ``numpy`` arrays defined in :mod:`modified_annealing.state`.  No instance of
``PlacementState`` is ever passed in: that container is only meaningful
at the orchestration layer (``worker.py`` / ``main.py``).  This design
keeps the hot path free of Python attribute lookups and lets the kernels
compile down to tight loops over contiguous memory.

The kernels fall into three groups:

1. **Spatial-grid primitives** (:func:`compute_macro_bin_range_njit`,
   :func:`paint_macro_njit`, :func:`clear_macro_njit`,
   :func:`check_collision_for_shift_njit`).  These maintain and query
   ``spatial_grid``, the integer matrix that gives :math:`O(1)` macro-
   to-cell membership lookups.

2. **Half-Perimeter Wirelength deltas**
   (:func:`compute_total_hpwl_njit`, :func:`net_bbox_with_override_njit`,
   :func:`hpwl_delta_for_shift_njit`, :func:`hpwl_delta_for_swap_njit`,
   :func:`commit_hpwl_delta_njit`).  Given a candidate macro move the
   kernels walk only the nets attached to that macro via the
   ``macro_net_*`` reverse-CSR adjacency, recompute each affected net's
   bounding box exactly from its pin list, and return the exact change
   in weighted HPWL.

3. **Density / congestion deltas**
   (:func:`compute_density_grid_njit`,
   :func:`density_grid_shift_delta_njit`,
   :func:`density_grid_swap_delta_njit`,
   :func:`commit_density_grid_delta_njit`,
   :func:`density_cost_from_grid_njit`,
   :func:`congestion_cost_from_grid_njit`).  An auxiliary
   ``density_grid`` ``float64`` matrix (allocated by ``worker.py``,
   *not* part of ``PlacementState`` so as not to disturb the locked
   ``state.py`` schema) tracks the *continuous* fraction of bin area
   covered by macros.  A macro shift only touches the bins it exits or
   enters; contributions from every other macro cancel exactly in the
   per-cell delta, so the cost change reduces to a closed-form sum over
   the union of old and new bin ranges.

Mathematical conventions
------------------------
The cost proxies aggregated by this module are

.. math::

    \\text{WL}     &= \\sum_{k=1}^{M} w_k\\bigl[(x^\\max_k - x^\\min_k) + (y^\\max_k - y^\\min_k)\\bigr] \\\\
    \\text{Dens}   &= \\sum_{r,c} d_{r,c}^{\\,2} \\\\
    \\text{Cong}   &= \\sum_{r,c} d_{r,c}^{\\,4}

where :math:`d_{r,c}` is the cell density

.. math:: d_{r,c} = \\frac{1}{b_w b_h} \\sum_{m \\in \\mathcal{M}} \\bigl|\\,[x^\\ell_m, x^\\ell_m + w_m] \\cap [c b_w, (c+1) b_w]\\bigr| \\,\\bigl|\\,[y^\\ell_m, y^\\ell_m + h_m] \\cap [r b_h, (r+1) b_h]\\bigr|

i.e. the *continuous* overlap area of all macros with the bin
:math:`(r, c)`, divided by the bin area.  The :math:`L^2` aggregate
penalises hotspots; the :math:`L^4` aggregate is a sharper congestion
proxy that grows steeply as a few cells become overloaded.  Both are
closed under additive per-cell deltas, which is what makes the kernels
:math:`O(\\text{cells touched})` per move rather than
:math:`O(\\text{grid size})`.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ─── Numba decorator shim ────────────────────────────────────────────────────
#
# ``numba`` is listed as a core dependency in ``pyproject.toml``, but if the
# environment lacks it (e.g. an isolated linting or unit-test run) we
# transparently fall back to a no-op decorator so the module remains
# importable and all kernels stay executable as pure Python.

try:
    from numba import njit as _njit  # type: ignore

    def njit(*args, **kwargs):
        """``@njit(cache=True, ...)``-compatible decorator alias.

        Always sets ``cache=True`` unless the caller explicitly overrides
        it, matching the directive in ``cursor.md`` §E.
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _njit(cache=True)(args[0])
        kwargs.setdefault("cache", True)
        return _njit(*args, **kwargs)

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover – numba unavailable in some sandboxes
    def njit(*args, **kwargs):
        """Identity decorator used when Numba is unavailable."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _wrap(f):
            return f
        return _wrap

    NUMBA_AVAILABLE = False


# ─── Shared constants (must match state.py) ──────────────────────────────────

EMPTY_CELL: int = -1
"""Sentinel value for ``spatial_grid`` cells with no occupying macro.
Must agree with :data:`modified_annealing.state.EMPTY_CELL`."""

EPS: float = 1e-9
"""Numerical tolerance for bin-range edge rounding (mirrors ``state.EPS``)."""


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 1.  Spatial-grid primitives                                            ║
# ╚════════════════════════════════════════════════════════════════════════╝

@njit(cache=True)
def compute_macro_bin_range_njit(
    x_ll: float, y_ll: float,
    width: float, height: float,
    bin_width: float, bin_height: float,
    grid_num_cols: int, grid_num_rows: int,
) -> Tuple[int, int, int, int]:
    """Inclusive bin range covered by a macro's continuous bounding box.

    Mirrors :func:`modified_annealing.state.compute_macro_bin_range` but as an
    ``@njit`` kernel so it can be called from the inner Metropolis loop
    without a Python round-trip.  Uses the half-open interval convention
    on the upper edge so macros touching exactly at a bin boundary do
    not share any bin.
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


@njit(cache=True)
def paint_macro_njit(
    spatial_grid: np.ndarray,
    macro_idx: int,
    x_ll: float, y_ll: float, width: float, height: float,
    bin_width: float, bin_height: float,
) -> None:
    """Stamp ``macro_idx`` into every cell of ``spatial_grid`` it covers.

    Overwrites any existing occupant in the affected sub-matrix.  The
    caller is expected to have either cleared the previous occupant or
    to be operating in the exploration phase where overlapping stamps
    are intentional.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range_njit(
        x_ll, y_ll, width, height,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            spatial_grid[r, c] = macro_idx


@njit(cache=True)
def clear_macro_njit(
    spatial_grid: np.ndarray,
    macro_idx: int,
    x_ll: float, y_ll: float, width: float, height: float,
    bin_width: float, bin_height: float,
) -> None:
    """Reset every cell that holds ``macro_idx`` in the macro's bbox to ``EMPTY_CELL``.

    Cells holding a *different* id are left untouched – this safely
    handles the case where another macro has overwritten ``macro_idx``
    in some bins during the exploration phase.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range_njit(
        x_ll, y_ll, width, height,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if spatial_grid[r, c] == macro_idx:
                spatial_grid[r, c] = EMPTY_CELL


@njit(cache=True)
def check_collision_for_shift_njit(
    spatial_grid: np.ndarray,
    macro_idx: int,
    new_x_ll: float, new_y_ll: float, width: float, height: float,
    bin_width: float, bin_height: float,
) -> int:
    """Strict :math:`O(\\text{bins covered})` collision test for a shift.

    Returns the *first foreign macro id* found in any bin the candidate
    bbox would occupy, or :data:`EMPTY_CELL` (``-1``) if the move is
    collision-free.  A cell holding the moving macro itself does **not**
    count as a collision (the stale stamp will be cleared and re-painted
    on commit).

    Used by ``worker.py`` Phase-3 (legalisation mode) to reject any move
    that introduces a new overlap on the spatial grid.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range_njit(
        new_x_ll, new_y_ll, width, height,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            occupant = spatial_grid[r, c]
            if occupant != EMPTY_CELL and occupant != macro_idx:
                return occupant
    return EMPTY_CELL


@njit(cache=True)
def count_grid_collisions_njit(
    spatial_grid: np.ndarray,
    macro_coords: np.ndarray,
    macro_dims: np.ndarray,
    num_hard_macros: int,
    bin_width: float, bin_height: float,
) -> int:
    """Total number of cells that a hard macro shares with a *different*
    hard macro.

    Used by the worker's final validity sweep to confirm the spatial
    grid is overlap-free before reporting a solution.  Operates by
    re-painting every hard macro onto a scratch grid and counting cell
    write-conflicts.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    scratch = np.full((grid_num_rows, grid_num_cols), EMPTY_CELL, dtype=np.int32)
    conflicts = 0
    for m in range(num_hard_macros):
        x_ll = macro_coords[m, 0]
        y_ll = macro_coords[m, 1]
        w    = macro_dims[m, 0]
        h    = macro_dims[m, 1]
        c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range_njit(
            x_ll, y_ll, w, h,
            bin_width, bin_height, grid_num_cols, grid_num_rows,
        )
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if scratch[r, c] != EMPTY_CELL and scratch[r, c] != m:
                    conflicts += 1
                scratch[r, c] = m
    return conflicts


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 2.  Half-Perimeter Wirelength delta kernels                            ║
# ╚════════════════════════════════════════════════════════════════════════╝

@njit(cache=True)
def _node_center_x_njit(
    node_id: int,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
) -> float:
    """X-coordinate of the centre of a generic netlist node."""
    if node_id < num_macros:
        return macro_coords[node_id, 0] + 0.5 * macro_dims[node_id, 0]
    return port_coords[node_id - num_macros, 0]


@njit(cache=True)
def _node_center_y_njit(
    node_id: int,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
) -> float:
    """Y-coordinate of the centre of a generic netlist node."""
    if node_id < num_macros:
        return macro_coords[node_id, 1] + 0.5 * macro_dims[node_id, 1]
    return port_coords[node_id - num_macros, 1]


@njit(cache=True)
def net_bbox_with_override_njit(
    net_id: int,
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    override_a: int, x_ll_a: float, y_ll_a: float,
    override_b: int, x_ll_b: float, y_ll_b: float,
) -> Tuple[float, float, float, float]:
    """Compute ``(min_x, max_x, min_y, max_y)`` for one net.

    ``override_a`` and ``override_b`` give the kernel an opportunity to
    substitute hypothetical bottom-left coordinates for one or two
    macros without actually mutating ``macro_coords``.  Pass ``-1`` for
    either slot to disable the corresponding override.

    The net's centre coordinates are computed identically to
    :func:`modified_annealing.state.compute_net_bbox` so the cached
    ``net_bbox`` and a fresh recomputation are bit-identical when no
    override is supplied.
    """
    s = net_pin_offsets[net_id]
    e = net_pin_offsets[net_id + 1]
    min_x =  np.inf
    max_x = -np.inf
    min_y =  np.inf
    max_y = -np.inf
    for p in range(s, e):
        nid = net_pin_owners[p]
        if nid == override_a:
            x = x_ll_a + 0.5 * macro_dims[nid, 0]
            y = y_ll_a + 0.5 * macro_dims[nid, 1]
        elif nid == override_b:
            x = x_ll_b + 0.5 * macro_dims[nid, 0]
            y = y_ll_b + 0.5 * macro_dims[nid, 1]
        elif nid < num_macros:
            x = macro_coords[nid, 0] + 0.5 * macro_dims[nid, 0]
            y = macro_coords[nid, 1] + 0.5 * macro_dims[nid, 1]
        else:
            port_id = nid - num_macros
            x = port_coords[port_id, 0]
            y = port_coords[port_id, 1]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    return min_x, max_x, min_y, max_y


@njit(cache=True)
def compute_total_hpwl_njit(
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    net_weights: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    num_nets: int,
) -> float:
    """Total weighted Half-Perimeter Wirelength over every net.

    .. math::

        \\text{HPWL} = \\sum_k w_k \\Bigl[(x^\\max_k - x^\\min_k) + (y^\\max_k - y^\\min_k)\\Bigr]

    This is the *ground-truth* recomputation; the MCMC engine never
    calls it inside the inner loop, only at construction / verification
    time.
    """
    total = 0.0
    for k in range(num_nets):
        s = net_pin_offsets[k]
        e = net_pin_offsets[k + 1]
        if e <= s:
            continue
        mnx, mxx, mny, mxy = net_bbox_with_override_njit(
            k, net_pin_owners, net_pin_offsets,
            macro_coords, macro_dims, port_coords, num_macros,
            -1, 0.0, 0.0, -1, 0.0, 0.0,
        )
        total += net_weights[k] * ((mxx - mnx) + (mxy - mny))
    return total


@njit(cache=True)
def populate_net_bbox_njit(
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    num_nets: int,
    net_bbox: np.ndarray,
) -> None:
    """Fill ``net_bbox`` from scratch using current macro coordinates.

    Equivalent to :func:`modified_annealing.state.compute_net_bbox` but as an
    ``@njit`` kernel.  Useful when a worker process wants to
    re-synchronise the cache after a non-incremental update (e.g. after
    GRASP initialisation).
    """
    for k in range(num_nets):
        s = net_pin_offsets[k]
        e = net_pin_offsets[k + 1]
        if e <= s:
            net_bbox[k, 0] = 0.0
            net_bbox[k, 1] = 0.0
            net_bbox[k, 2] = 0.0
            net_bbox[k, 3] = 0.0
            continue
        mnx, mxx, mny, mxy = net_bbox_with_override_njit(
            k, net_pin_owners, net_pin_offsets,
            macro_coords, macro_dims, port_coords, num_macros,
            -1, 0.0, 0.0, -1, 0.0, 0.0,
        )
        net_bbox[k, 0] = mnx
        net_bbox[k, 1] = mxx
        net_bbox[k, 2] = mny
        net_bbox[k, 3] = mxy


@njit(cache=True)
def hpwl_delta_for_shift_njit(
    macro_idx: int,
    new_x_ll: float, new_y_ll: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    macro_net_ids: np.ndarray, macro_net_offsets: np.ndarray,
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    net_weights: np.ndarray,
    net_bbox: np.ndarray,
    out_affected_nets: np.ndarray,
    out_new_bboxes: np.ndarray,
) -> Tuple[float, int]:
    """Exact weighted-HPWL change for shifting one macro.

    Walks only the nets attached to ``macro_idx`` via the reverse-CSR
    ``macro_net_*`` adjacency.  For each affected net the new bounding
    box is recomputed from scratch with the moving macro's coordinate
    substituted on the fly – this is exact regardless of whether the
    macro currently sits on the bbox boundary or in the interior.

    The kernel does **not** mutate ``net_bbox`` or ``macro_coords``.
    Instead, the caller's pre-allocated scratch buffers are filled:

    * ``out_affected_nets[:n]`` – ids of the touched nets (one per net)
    * ``out_new_bboxes[:n, :]`` – proposed ``[min_x, max_x, min_y, max_y]``

    The caller decides accept/reject; on acceptance it invokes
    :func:`commit_hpwl_delta_njit` to apply the cached new bboxes.

    Returns
    -------
    delta : float
        ``Σ w_k · (HPWL_new(k) − HPWL_old(k))`` over all affected nets.
    n     : int
        Number of nets written into the output buffers.
    """
    s = macro_net_offsets[macro_idx]
    e = macro_net_offsets[macro_idx + 1]
    delta = 0.0
    n = 0
    for i in range(s, e):
        net_id = macro_net_ids[i]
        old_min_x = net_bbox[net_id, 0]
        old_max_x = net_bbox[net_id, 1]
        old_min_y = net_bbox[net_id, 2]
        old_max_y = net_bbox[net_id, 3]
        old_hpwl = (old_max_x - old_min_x) + (old_max_y - old_min_y)

        new_min_x, new_max_x, new_min_y, new_max_y = net_bbox_with_override_njit(
            net_id, net_pin_owners, net_pin_offsets,
            macro_coords, macro_dims, port_coords, num_macros,
            macro_idx, new_x_ll, new_y_ll,
            -1, 0.0, 0.0,
        )
        new_hpwl = (new_max_x - new_min_x) + (new_max_y - new_min_y)

        delta += net_weights[net_id] * (new_hpwl - old_hpwl)
        out_affected_nets[n] = net_id
        out_new_bboxes[n, 0] = new_min_x
        out_new_bboxes[n, 1] = new_max_x
        out_new_bboxes[n, 2] = new_min_y
        out_new_bboxes[n, 3] = new_max_y
        n += 1
    return delta, n


@njit(cache=True)
def hpwl_delta_for_swap_njit(
    macro_a: int, macro_b: int,
    new_xa_ll: float, new_ya_ll: float,
    new_xb_ll: float, new_yb_ll: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    macro_net_ids: np.ndarray, macro_net_offsets: np.ndarray,
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    net_weights: np.ndarray,
    net_bbox: np.ndarray,
    out_affected_nets: np.ndarray,
    out_new_bboxes: np.ndarray,
) -> Tuple[float, int]:
    """Exact weighted-HPWL change for a simultaneous two-macro update.

    Used by the *swap* mutation, where macros ``a`` and ``b`` exchange
    coordinates (``new_xa_ll, new_ya_ll`` may equal the *current*
    ``macro_coords[b]`` and vice versa, but the kernel does not assume
    this – it merely substitutes whatever bottom-left coordinates the
    caller supplies).

    Affected nets are :math:`\\mathcal{N}(a) \\cup \\mathcal{N}(b)`.
    Because ``macro_net_ids`` is stored in ascending net-id order for
    each macro (a property of the ``build_csr_netlist`` sort), the union
    is computed in :math:`O(\\deg(a) + \\deg(b))` via a linear merge with
    deduplication.

    Output buffer semantics are identical to
    :func:`hpwl_delta_for_shift_njit`.
    """
    s_a = macro_net_offsets[macro_a]; e_a = macro_net_offsets[macro_a + 1]
    s_b = macro_net_offsets[macro_b]; e_b = macro_net_offsets[macro_b + 1]
    i_a = s_a
    i_b = s_b
    delta = 0.0
    n = 0

    while i_a < e_a or i_b < e_b:
        if i_a < e_a and i_b < e_b:
            na = macro_net_ids[i_a]
            nb = macro_net_ids[i_b]
            if na < nb:
                net_id = na
                i_a += 1
            elif na > nb:
                net_id = nb
                i_b += 1
            else:
                net_id = na
                i_a += 1
                i_b += 1
        elif i_a < e_a:
            net_id = macro_net_ids[i_a]
            i_a += 1
        else:
            net_id = macro_net_ids[i_b]
            i_b += 1

        old_min_x = net_bbox[net_id, 0]
        old_max_x = net_bbox[net_id, 1]
        old_min_y = net_bbox[net_id, 2]
        old_max_y = net_bbox[net_id, 3]
        old_hpwl = (old_max_x - old_min_x) + (old_max_y - old_min_y)

        new_min_x, new_max_x, new_min_y, new_max_y = net_bbox_with_override_njit(
            net_id, net_pin_owners, net_pin_offsets,
            macro_coords, macro_dims, port_coords, num_macros,
            macro_a, new_xa_ll, new_ya_ll,
            macro_b, new_xb_ll, new_yb_ll,
        )
        new_hpwl = (new_max_x - new_min_x) + (new_max_y - new_min_y)

        delta += net_weights[net_id] * (new_hpwl - old_hpwl)
        out_affected_nets[n] = net_id
        out_new_bboxes[n, 0] = new_min_x
        out_new_bboxes[n, 1] = new_max_x
        out_new_bboxes[n, 2] = new_min_y
        out_new_bboxes[n, 3] = new_max_y
        n += 1
    return delta, n


@njit(cache=True)
def commit_hpwl_delta_njit(
    affected_nets: np.ndarray,
    new_bboxes: np.ndarray,
    num_affected: int,
    net_bbox: np.ndarray,
) -> None:
    """Apply a delta computed by ``hpwl_delta_for_*`` into ``net_bbox``."""
    for i in range(num_affected):
        nid = affected_nets[i]
        net_bbox[nid, 0] = new_bboxes[i, 0]
        net_bbox[nid, 1] = new_bboxes[i, 1]
        net_bbox[nid, 2] = new_bboxes[i, 2]
        net_bbox[nid, 3] = new_bboxes[i, 3]


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 3.  Density & congestion delta kernels                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝
#
# ``density_grid`` is a contiguous ``(grid_num_rows, grid_num_cols)``
# ``float64`` matrix that lives outside ``PlacementState`` (the schema in
# ``state.py`` is locked).  ``worker.py`` allocates one per worker
# process, initialises it via :func:`compute_density_grid_njit`, and
# then maintains it incrementally with
# :func:`density_grid_shift_delta_njit` / :func:`commit_density_grid_delta_njit`.
#
# Cells store
#
# .. math::
#
#     d_{r,c} = \frac{\sum_m \text{overlap}_{r,c}(m)}{b_w b_h}
#
# i.e. the **continuous** macro area fraction inside the bin.  Allowing
# values to exceed ``1`` lets the engine *see* overlaps in the cost
# landscape during the exploration phase even though the spatial grid
# has lost that information (the integer grid keeps only the last
# writer).

@njit(cache=True)
def macro_cell_overlap_area_njit(
    x_ll: float, y_ll: float, width: float, height: float,
    cell_x0: float, cell_x1: float,
    cell_y0: float, cell_y1: float,
) -> float:
    """Continuous overlap area between a macro bbox and one grid cell.

    .. math::
        \\text{overlap} = \\max\\bigl(0, \\min(x^\\ell+w, c_{x1}) - \\max(x^\\ell, c_{x0})\\bigr) \\\\
        \\times\\,\\max\\bigl(0, \\min(y^\\ell+h, c_{y1}) - \\max(y^\\ell, c_{y0})\\bigr)
    """
    x_lo = x_ll if x_ll > cell_x0 else cell_x0
    x_hi = (x_ll + width) if (x_ll + width) < cell_x1 else cell_x1
    y_lo = y_ll if y_ll > cell_y0 else cell_y0
    y_hi = (y_ll + height) if (y_ll + height) < cell_y1 else cell_y1
    dx = x_hi - x_lo
    dy = y_hi - y_lo
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


@njit(cache=True)
def compute_density_grid_njit(
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    num_macros: int,
    bin_width: float, bin_height: float,
    grid_num_rows: int, grid_num_cols: int,
    density_grid: np.ndarray,
) -> None:
    """Build ``density_grid`` from scratch given current macro positions.

    Iterates every macro, maps its bbox to the relevant bin range, and
    *adds* its per-bin continuous overlap-fraction contribution to the
    cell value.  This is the only :math:`O(\\sum_m \\text{cells}(m))`
    kernel; every subsequent mutation uses
    :func:`density_grid_shift_delta_njit` for :math:`O(\\text{cells touched})`
    incremental updates.

    The grid is **not** zeroed first – callers should pass a freshly
    allocated zero matrix (or call ``density_grid[:] = 0.0`` themselves)
    when rebuilding from scratch.
    """
    inv_bin_area = 1.0 / (bin_width * bin_height)
    for m in range(num_macros):
        x_ll = macro_coords[m, 0]
        y_ll = macro_coords[m, 1]
        w    = macro_dims[m, 0]
        h    = macro_dims[m, 1]
        c_lo, c_hi, r_lo, r_hi = compute_macro_bin_range_njit(
            x_ll, y_ll, w, h,
            bin_width, bin_height, grid_num_cols, grid_num_rows,
        )
        for r in range(r_lo, r_hi + 1):
            cell_y0 = r * bin_height
            cell_y1 = cell_y0 + bin_height
            for c in range(c_lo, c_hi + 1):
                cell_x0 = c * bin_width
                cell_x1 = cell_x0 + bin_width
                a = macro_cell_overlap_area_njit(
                    x_ll, y_ll, w, h,
                    cell_x0, cell_x1, cell_y0, cell_y1,
                )
                density_grid[r, c] += a * inv_bin_area


@njit(cache=True)
def density_grid_shift_delta_njit(
    macro_idx: int,
    new_x_ll: float, new_y_ll: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    bin_width: float, bin_height: float,
    grid_num_rows: int, grid_num_cols: int,
    density_grid: np.ndarray,
    out_cell_rs: np.ndarray,
    out_cell_cs: np.ndarray,
    out_new_densities: np.ndarray,
) -> Tuple[float, float, int]:
    """Local density/congestion deltas for a single-macro shift.

    Iterates the **union** of the macro's *current* bin range and its
    *candidate* bin range (every cell outside this union is mathematically
    unaffected because the macro's contribution to other cells is zero
    both before and after the move).  For each touched cell ``(r, c)``
    the kernel computes

    * ``old_overlap`` – continuous area shared with the *current* bbox
    * ``new_overlap`` – continuous area shared with the *candidate* bbox
    * ``Δd          = (new - old) / (b_w b_h)``
    * ``d_new       = d_old + Δd``

    and accumulates the closed-form cost deltas

    .. math::
        \\Delta\\text{Dens} = \\sum_{(r,c)} d_\\text{new}^2 - d_\\text{old}^2
        ,\\qquad
        \\Delta\\text{Cong} = \\sum_{(r,c)} d_\\text{new}^4 - d_\\text{old}^4

    Output buffers:

    * ``out_cell_rs[:n]``, ``out_cell_cs[:n]`` – touched cell indices
    * ``out_new_densities[:n]``                – proposed new ``d_{r,c}``

    Cells whose delta is mathematically zero (the corners of the union
    range that neither old nor new bbox actually overlaps) are skipped,
    so ``n`` may be smaller than the worst-case
    ``(union_rows × union_cols)`` upper bound.  Callers should size
    their scratch buffers to the worst case (see
    :func:`max_cells_per_shift`).
    """
    old_x_ll = macro_coords[macro_idx, 0]
    old_y_ll = macro_coords[macro_idx, 1]
    w        = macro_dims[macro_idx, 0]
    h        = macro_dims[macro_idx, 1]

    old_c_lo, old_c_hi, old_r_lo, old_r_hi = compute_macro_bin_range_njit(
        old_x_ll, old_y_ll, w, h,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    new_c_lo, new_c_hi, new_r_lo, new_r_hi = compute_macro_bin_range_njit(
        new_x_ll, new_y_ll, w, h,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    c_lo = old_c_lo if old_c_lo < new_c_lo else new_c_lo
    c_hi = old_c_hi if old_c_hi > new_c_hi else new_c_hi
    r_lo = old_r_lo if old_r_lo < new_r_lo else new_r_lo
    r_hi = old_r_hi if old_r_hi > new_r_hi else new_r_hi

    inv_bin_area = 1.0 / (bin_width * bin_height)

    delta_density_cost    = 0.0
    delta_congestion_cost = 0.0
    n = 0

    for r in range(r_lo, r_hi + 1):
        cell_y0 = r * bin_height
        cell_y1 = cell_y0 + bin_height
        for c in range(c_lo, c_hi + 1):
            cell_x0 = c * bin_width
            cell_x1 = cell_x0 + bin_width

            old_a = macro_cell_overlap_area_njit(
                old_x_ll, old_y_ll, w, h,
                cell_x0, cell_x1, cell_y0, cell_y1,
            )
            new_a = macro_cell_overlap_area_njit(
                new_x_ll, new_y_ll, w, h,
                cell_x0, cell_x1, cell_y0, cell_y1,
            )
            cell_delta = (new_a - old_a) * inv_bin_area
            if cell_delta == 0.0:
                continue

            d_old = density_grid[r, c]
            d_new = d_old + cell_delta

            delta_density_cost    += d_new * d_new            - d_old * d_old
            delta_congestion_cost += (d_new * d_new) * (d_new * d_new) \
                                  -  (d_old * d_old) * (d_old * d_old)

            out_cell_rs[n]       = r
            out_cell_cs[n]       = c
            out_new_densities[n] = d_new
            n += 1

    return delta_density_cost, delta_congestion_cost, n


@njit(cache=True)
def density_grid_swap_delta_njit(
    macro_a: int, macro_b: int,
    new_xa_ll: float, new_ya_ll: float,
    new_xb_ll: float, new_yb_ll: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    bin_width: float, bin_height: float,
    grid_num_rows: int, grid_num_cols: int,
    density_grid: np.ndarray,
    out_cell_rs: np.ndarray,
    out_cell_cs: np.ndarray,
    out_new_densities: np.ndarray,
) -> Tuple[float, float, int]:
    """Local density/congestion deltas for a two-macro swap mutation.

    Touches the union of ``old_bbox(a) ∪ new_bbox(a) ∪ old_bbox(b) ∪
    new_bbox(b)`` and accumulates the *combined* per-cell area change
    from both macros simultaneously, so a cell that gains area from
    macro a and loses an equal area from macro b registers a zero
    delta (as it should).  Output buffer semantics match
    :func:`density_grid_shift_delta_njit`.
    """
    ax_ll = macro_coords[macro_a, 0]
    ay_ll = macro_coords[macro_a, 1]
    aw    = macro_dims[macro_a, 0]
    ah    = macro_dims[macro_a, 1]

    bx_ll = macro_coords[macro_b, 0]
    by_ll = macro_coords[macro_b, 1]
    bw_   = macro_dims[macro_b, 0]
    bh_   = macro_dims[macro_b, 1]

    a_old_c_lo, a_old_c_hi, a_old_r_lo, a_old_r_hi = compute_macro_bin_range_njit(
        ax_ll, ay_ll, aw, ah, bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    a_new_c_lo, a_new_c_hi, a_new_r_lo, a_new_r_hi = compute_macro_bin_range_njit(
        new_xa_ll, new_ya_ll, aw, ah, bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    b_old_c_lo, b_old_c_hi, b_old_r_lo, b_old_r_hi = compute_macro_bin_range_njit(
        bx_ll, by_ll, bw_, bh_, bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    b_new_c_lo, b_new_c_hi, b_new_r_lo, b_new_r_hi = compute_macro_bin_range_njit(
        new_xb_ll, new_yb_ll, bw_, bh_, bin_width, bin_height, grid_num_cols, grid_num_rows,
    )

    c_lo = a_old_c_lo
    if a_new_c_lo < c_lo: c_lo = a_new_c_lo
    if b_old_c_lo < c_lo: c_lo = b_old_c_lo
    if b_new_c_lo < c_lo: c_lo = b_new_c_lo

    c_hi = a_old_c_hi
    if a_new_c_hi > c_hi: c_hi = a_new_c_hi
    if b_old_c_hi > c_hi: c_hi = b_old_c_hi
    if b_new_c_hi > c_hi: c_hi = b_new_c_hi

    r_lo = a_old_r_lo
    if a_new_r_lo < r_lo: r_lo = a_new_r_lo
    if b_old_r_lo < r_lo: r_lo = b_old_r_lo
    if b_new_r_lo < r_lo: r_lo = b_new_r_lo

    r_hi = a_old_r_hi
    if a_new_r_hi > r_hi: r_hi = a_new_r_hi
    if b_old_r_hi > r_hi: r_hi = b_old_r_hi
    if b_new_r_hi > r_hi: r_hi = b_new_r_hi

    inv_bin_area = 1.0 / (bin_width * bin_height)

    delta_density_cost    = 0.0
    delta_congestion_cost = 0.0
    n = 0

    for r in range(r_lo, r_hi + 1):
        cell_y0 = r * bin_height
        cell_y1 = cell_y0 + bin_height
        for c in range(c_lo, c_hi + 1):
            cell_x0 = c * bin_width
            cell_x1 = cell_x0 + bin_width

            a_old = macro_cell_overlap_area_njit(
                ax_ll, ay_ll, aw, ah, cell_x0, cell_x1, cell_y0, cell_y1,
            )
            a_new = macro_cell_overlap_area_njit(
                new_xa_ll, new_ya_ll, aw, ah, cell_x0, cell_x1, cell_y0, cell_y1,
            )
            b_old = macro_cell_overlap_area_njit(
                bx_ll, by_ll, bw_, bh_, cell_x0, cell_x1, cell_y0, cell_y1,
            )
            b_new = macro_cell_overlap_area_njit(
                new_xb_ll, new_yb_ll, bw_, bh_, cell_x0, cell_x1, cell_y0, cell_y1,
            )
            cell_delta = ((a_new - a_old) + (b_new - b_old)) * inv_bin_area
            if cell_delta == 0.0:
                continue

            d_old = density_grid[r, c]
            d_new = d_old + cell_delta

            delta_density_cost    += d_new * d_new      - d_old * d_old
            delta_congestion_cost += (d_new * d_new) * (d_new * d_new) \
                                  -  (d_old * d_old) * (d_old * d_old)

            out_cell_rs[n]       = r
            out_cell_cs[n]       = c
            out_new_densities[n] = d_new
            n += 1

    return delta_density_cost, delta_congestion_cost, n


@njit(cache=True)
def density_grid_reshape_delta_njit(
    macro_idx: int,
    new_x_ll: float, new_y_ll: float,
    new_width: float, new_height: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    bin_width: float, bin_height: float,
    grid_num_rows: int, grid_num_cols: int,
    density_grid: np.ndarray,
    out_cell_rs: np.ndarray,
    out_cell_cs: np.ndarray,
    out_new_densities: np.ndarray,
) -> Tuple[float, float, int]:
    """Local density/congestion deltas for a soft-macro *reshape*.

    The reshape mutation (cursor.md §D.3) changes both the position and
    the ``(width, height)`` of a soft macro while preserving its area.
    The kernel is structurally identical to
    :func:`density_grid_shift_delta_njit` but uses ``(new_width,
    new_height)`` for the candidate bbox instead of the current
    ``macro_dims`` row.  The caller is responsible for actually writing
    the new dimensions back into ``macro_dims`` on acceptance.
    """
    old_x_ll = macro_coords[macro_idx, 0]
    old_y_ll = macro_coords[macro_idx, 1]
    old_w    = macro_dims[macro_idx, 0]
    old_h    = macro_dims[macro_idx, 1]

    old_c_lo, old_c_hi, old_r_lo, old_r_hi = compute_macro_bin_range_njit(
        old_x_ll, old_y_ll, old_w, old_h,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    new_c_lo, new_c_hi, new_r_lo, new_r_hi = compute_macro_bin_range_njit(
        new_x_ll, new_y_ll, new_width, new_height,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    c_lo = old_c_lo if old_c_lo < new_c_lo else new_c_lo
    c_hi = old_c_hi if old_c_hi > new_c_hi else new_c_hi
    r_lo = old_r_lo if old_r_lo < new_r_lo else new_r_lo
    r_hi = old_r_hi if old_r_hi > new_r_hi else new_r_hi

    inv_bin_area = 1.0 / (bin_width * bin_height)

    delta_density_cost    = 0.0
    delta_congestion_cost = 0.0
    n = 0

    for r in range(r_lo, r_hi + 1):
        cell_y0 = r * bin_height
        cell_y1 = cell_y0 + bin_height
        for c in range(c_lo, c_hi + 1):
            cell_x0 = c * bin_width
            cell_x1 = cell_x0 + bin_width

            old_a = macro_cell_overlap_area_njit(
                old_x_ll, old_y_ll, old_w, old_h,
                cell_x0, cell_x1, cell_y0, cell_y1,
            )
            new_a = macro_cell_overlap_area_njit(
                new_x_ll, new_y_ll, new_width, new_height,
                cell_x0, cell_x1, cell_y0, cell_y1,
            )
            cell_delta = (new_a - old_a) * inv_bin_area
            if cell_delta == 0.0:
                continue

            d_old = density_grid[r, c]
            d_new = d_old + cell_delta

            delta_density_cost    += d_new * d_new      - d_old * d_old
            delta_congestion_cost += (d_new * d_new) * (d_new * d_new) \
                                  -  (d_old * d_old) * (d_old * d_old)

            out_cell_rs[n]       = r
            out_cell_cs[n]       = c
            out_new_densities[n] = d_new
            n += 1

    return delta_density_cost, delta_congestion_cost, n


@njit(cache=True)
def commit_density_grid_delta_njit(
    cell_rs: np.ndarray,
    cell_cs: np.ndarray,
    new_densities: np.ndarray,
    num_cells: int,
    density_grid: np.ndarray,
) -> None:
    """Write the proposed new per-cell densities back into ``density_grid``."""
    for i in range(num_cells):
        density_grid[cell_rs[i], cell_cs[i]] = new_densities[i]


@njit(cache=True)
def density_cost_from_grid_njit(density_grid: np.ndarray) -> float:
    """Global density cost :math:`\\sum_{r,c} d_{r,c}^{\\,2}`.

    Used only at construction and verification time; the MCMC inner loop
    tracks an incrementally updated scalar instead of recomputing this
    sweep.
    """
    grid_num_rows = density_grid.shape[0]
    grid_num_cols = density_grid.shape[1]
    s = 0.0
    for r in range(grid_num_rows):
        for c in range(grid_num_cols):
            d = density_grid[r, c]
            s += d * d
    return s


@njit(cache=True)
def congestion_cost_from_grid_njit(density_grid: np.ndarray) -> float:
    """Global congestion cost :math:`\\sum_{r,c} d_{r,c}^{\\,4}`.

    The :math:`L^4` aggregate grows much faster than :math:`L^2` as
    individual cells approach saturation, biasing the MCMC away from
    routing chokepoints.
    """
    grid_num_rows = density_grid.shape[0]
    grid_num_cols = density_grid.shape[1]
    s = 0.0
    for r in range(grid_num_rows):
        for c in range(grid_num_cols):
            d = density_grid[r, c]
            d2 = d * d
            s += d2 * d2
    return s


# ─── Scratch-buffer sizing helpers (pure Python, allocation-time only) ─────

def max_nets_per_macro(macro_net_offsets: np.ndarray) -> int:
    """Largest per-macro net degree; safe size for HPWL scratch buffers.

    The shift kernel touches at most ``deg(macro)`` nets; the swap
    kernel touches at most ``deg(a) + deg(b)`` so callers handling
    swaps should allocate ``2 * max_nets_per_macro`` to be safe.
    """
    if macro_net_offsets.size <= 1:
        return 0
    return int(np.diff(macro_net_offsets.astype(np.int64)).max(initial=0))


def max_cells_per_shift(
    macro_dims: np.ndarray, bin_width: float, bin_height: float,
) -> int:
    """Upper bound on cells touched by a single shift mutation.

    A worst-case shift moves a macro by an arbitrarily large distance,
    so the union of old + new bin ranges is bounded by *twice* the bin
    footprint of the largest macro.  We deliberately over-estimate by
    adding a one-cell guard band on each axis to absorb the ``EPS``
    rounding in :func:`compute_macro_bin_range_njit`.
    """
    if macro_dims.size == 0:
        return 0
    max_w = float(macro_dims[:, 0].max())
    max_h = float(macro_dims[:, 1].max())
    cells_w = int(math.ceil(max_w / bin_width))  + 2
    cells_h = int(math.ceil(max_h / bin_height)) + 2
    # Worst-case union of old and new bbox = 2× along each axis.
    return 2 * cells_w * 2 * cells_h


def max_cells_per_swap(
    macro_dims: np.ndarray, bin_width: float, bin_height: float,
) -> int:
    """Upper bound on cells touched by a two-macro swap mutation."""
    return 2 * max_cells_per_shift(macro_dims, bin_width, bin_height)


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Numerical self-test: every kernel is checked against an
    # independent pure-Python ground-truth implementation on a tiny but
    # non-trivial scenario.  Runs in <1 s with or without Numba.
    print(f"Numba available: {NUMBA_AVAILABLE}")

    rng = np.random.default_rng(0)

    # ── Construct a tiny but representative scene ──────────────────────
    num_macros      = 5
    num_hard_macros = 4
    num_ports       = 2

    macro_coords = np.array([
        [ 1.0,  1.0],
        [ 7.0,  2.0],
        [ 2.5,  6.0],
        [ 8.0,  7.5],
        [ 4.0,  4.0],   # soft
    ], dtype=np.float64)
    macro_dims = np.array([
        [3.0, 3.0],
        [2.0, 3.0],
        [3.0, 2.5],
        [2.5, 2.0],
        [2.0, 2.0],
    ], dtype=np.float64)
    port_coords = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ], dtype=np.float64)

    canvas_w = 12.0
    canvas_h = 12.0
    bin_w    = 1.5
    bin_h    = 1.5
    grid_num_cols = int(math.ceil(canvas_w / bin_w))
    grid_num_rows = int(math.ceil(canvas_h / bin_h))

    # Netlist: 3 nets
    #   net 0: macros 0, 1, port 5
    #   net 1: macros 1, 2, 4
    #   net 2: macros 3, port 6
    net_pin_owners  = np.array(
        [0, 1, 5,  1, 2, 4,  3, 6], dtype=np.int32,
    )
    net_pin_offsets = np.array([0, 3, 6, 8], dtype=np.int32)
    net_weights     = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    num_nets = 3

    # Reverse CSR (sorted per macro – matches build_csr_netlist output)
    macro_net_offsets = np.array([0, 1, 3, 4, 5, 6], dtype=np.int32)
    macro_net_ids     = np.array([0,  0, 1,  1,  2,  1], dtype=np.int32)

    spatial_grid = np.full((grid_num_rows, grid_num_cols), EMPTY_CELL, dtype=np.int32)

    # ── Spatial grid primitives ────────────────────────────────────────
    for m in range(num_hard_macros):
        paint_macro_njit(
            spatial_grid, m,
            macro_coords[m, 0], macro_coords[m, 1],
            macro_dims[m, 0],   macro_dims[m, 1],
            bin_w, bin_h,
        )
    assert int((spatial_grid != EMPTY_CELL).sum()) > 0
    coll = check_collision_for_shift_njit(
        spatial_grid, 0,
        macro_coords[1, 0], macro_coords[1, 1],     # try to drop macro 0 on top of macro 1
        macro_dims[0, 0],   macro_dims[0, 1],
        bin_w, bin_h,
    )
    assert coll != EMPTY_CELL, "should collide with another macro id"
    # ground truth: occupant is whichever macro sits at the centre of macro 1
    coll2 = check_collision_for_shift_njit(
        spatial_grid, 0,
        macro_coords[0, 0], macro_coords[0, 1],     # current position of itself
        macro_dims[0, 0],   macro_dims[0, 1],
        bin_w, bin_h,
    )
    assert coll2 == EMPTY_CELL, "self-overlap must not be flagged"
    assert count_grid_collisions_njit(
        spatial_grid, macro_coords, macro_dims, num_hard_macros, bin_w, bin_h,
    ) == 0, "initial layout has no hard-macro overlaps"
    print("spatial-grid kernels OK")

    # ── HPWL ground truth (pure python) ────────────────────────────────
    def py_net_bbox(net_id, mc, md, overrides=()):
        """Plain numpy bbox for one net with optional overrides
        as a list of (macro_idx, x_ll, y_ll)."""
        s, e = int(net_pin_offsets[net_id]), int(net_pin_offsets[net_id + 1])
        ox = {idx: (x, y) for idx, x, y in overrides}
        xs, ys = [], []
        for p in range(s, e):
            nid = int(net_pin_owners[p])
            if nid in ox:
                x = ox[nid][0] + 0.5 * md[nid, 0]
                y = ox[nid][1] + 0.5 * md[nid, 1]
            elif nid < num_macros:
                x = mc[nid, 0] + 0.5 * md[nid, 0]
                y = mc[nid, 1] + 0.5 * md[nid, 1]
            else:
                pid = nid - num_macros
                x = port_coords[pid, 0]; y = port_coords[pid, 1]
            xs.append(x); ys.append(y)
        return min(xs), max(xs), min(ys), max(ys)

    # populate cache
    net_bbox = np.zeros((num_nets, 4), dtype=np.float64)
    populate_net_bbox_njit(
        net_pin_owners, net_pin_offsets,
        macro_coords, macro_dims, port_coords, num_macros, num_nets,
        net_bbox,
    )
    for k in range(num_nets):
        py = py_net_bbox(k, macro_coords, macro_dims)
        assert np.allclose(net_bbox[k], py, atol=1e-12), \
            f"net {k} bbox mismatch: njit={net_bbox[k]} py={py}"

    total_hpwl = compute_total_hpwl_njit(
        net_pin_owners, net_pin_offsets, net_weights,
        macro_coords, macro_dims, port_coords, num_macros, num_nets,
    )
    expected_total = sum(
        net_weights[k] * ((py_net_bbox(k, macro_coords, macro_dims)[1]
                          - py_net_bbox(k, macro_coords, macro_dims)[0])
                         + (py_net_bbox(k, macro_coords, macro_dims)[3]
                          - py_net_bbox(k, macro_coords, macro_dims)[2]))
        for k in range(num_nets)
    )
    assert math.isclose(total_hpwl, expected_total, abs_tol=1e-10)
    print(f"total HPWL kernel OK  ({total_hpwl:.4f})")

    # ── HPWL shift delta ───────────────────────────────────────────────
    mover = 1
    new_x, new_y = 5.0, 8.0
    max_deg = max_nets_per_macro(macro_net_offsets)
    aff = np.zeros(max_deg * 2, dtype=np.int32)
    new_bb = np.zeros((max_deg * 2, 4), dtype=np.float64)

    delta_njit, n = hpwl_delta_for_shift_njit(
        mover, new_x, new_y,
        macro_coords, macro_dims, port_coords, num_macros,
        macro_net_ids, macro_net_offsets,
        net_pin_owners, net_pin_offsets,
        net_weights, net_bbox,
        aff, new_bb,
    )
    # pure-python ground truth
    moved = macro_coords.copy(); moved[mover] = (new_x, new_y)
    py_delta = 0.0
    for k in range(num_nets):
        bb_old = py_net_bbox(k, macro_coords, macro_dims)
        bb_new = py_net_bbox(k, moved,        macro_dims)
        hp_old = (bb_old[1] - bb_old[0]) + (bb_old[3] - bb_old[2])
        hp_new = (bb_new[1] - bb_new[0]) + (bb_new[3] - bb_new[2])
        py_delta += net_weights[k] * (hp_new - hp_old)
    assert math.isclose(delta_njit, py_delta, abs_tol=1e-10), (delta_njit, py_delta)
    print(f"HPWL shift-delta kernel OK  (Δ={delta_njit:+.4f})")

    # commit and verify net_bbox post-shift matches a fresh recompute
    commit_hpwl_delta_njit(aff, new_bb, n, net_bbox)
    macro_coords[mover] = (new_x, new_y)
    fresh = np.zeros((num_nets, 4), dtype=np.float64)
    populate_net_bbox_njit(
        net_pin_owners, net_pin_offsets,
        macro_coords, macro_dims, port_coords, num_macros, num_nets,
        fresh,
    )
    assert np.allclose(net_bbox, fresh, atol=1e-12), \
        f"commit_hpwl mismatch:\n{net_bbox}\nvs\n{fresh}"
    print("HPWL commit kernel OK")

    # ── HPWL swap delta ────────────────────────────────────────────────
    a, b = 0, 3
    new_xa, new_ya = macro_coords[b, 0], macro_coords[b, 1]
    new_xb, new_yb = macro_coords[a, 0], macro_coords[a, 1]
    aff2 = np.zeros(max_deg * 4, dtype=np.int32)
    new_bb2 = np.zeros((max_deg * 4, 4), dtype=np.float64)
    delta_swap, n_swap = hpwl_delta_for_swap_njit(
        a, b, new_xa, new_ya, new_xb, new_yb,
        macro_coords, macro_dims, port_coords, num_macros,
        macro_net_ids, macro_net_offsets,
        net_pin_owners, net_pin_offsets,
        net_weights, net_bbox,
        aff2, new_bb2,
    )
    swapped = macro_coords.copy()
    swapped[a] = (new_xa, new_ya)
    swapped[b] = (new_xb, new_yb)
    py_delta_swap = 0.0
    for k in range(num_nets):
        bb_old = py_net_bbox(k, macro_coords, macro_dims)
        bb_new = py_net_bbox(k, swapped,      macro_dims)
        hp_old = (bb_old[1] - bb_old[0]) + (bb_old[3] - bb_old[2])
        hp_new = (bb_new[1] - bb_new[0]) + (bb_new[3] - bb_new[2])
        py_delta_swap += net_weights[k] * (hp_new - hp_old)
    assert math.isclose(delta_swap, py_delta_swap, abs_tol=1e-10), \
        (delta_swap, py_delta_swap)
    print(f"HPWL swap-delta kernel OK   (Δ={delta_swap:+.4f}, {n_swap} nets)")

    # ── density grid construction ──────────────────────────────────────
    density_grid = np.zeros((grid_num_rows, grid_num_cols), dtype=np.float64)
    compute_density_grid_njit(
        macro_coords, macro_dims, num_macros,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        density_grid,
    )
    # Pure-python ground truth using same continuous-overlap formula:
    py_dens = np.zeros_like(density_grid)
    inv_ba = 1.0 / (bin_w * bin_h)
    for m in range(num_macros):
        x_ll, y_ll = macro_coords[m]
        w, h = macro_dims[m]
        for r in range(grid_num_rows):
            for c in range(grid_num_cols):
                ox = max(0.0, min(x_ll + w, (c + 1) * bin_w) - max(x_ll, c * bin_w))
                oy = max(0.0, min(y_ll + h, (r + 1) * bin_h) - max(y_ll, r * bin_h))
                py_dens[r, c] += ox * oy * inv_ba
    assert np.allclose(density_grid, py_dens, atol=1e-12), \
        f"density grid mismatch:\n{density_grid}\nvs\n{py_dens}"
    base_density_cost    = density_cost_from_grid_njit(density_grid)
    base_congestion_cost = congestion_cost_from_grid_njit(density_grid)
    assert math.isclose(base_density_cost,    float((py_dens ** 2).sum()), abs_tol=1e-10)
    assert math.isclose(base_congestion_cost, float((py_dens ** 4).sum()), abs_tol=1e-10)
    print(f"density grid kernel OK     (dens={base_density_cost:.4f}, "
          f"cong={base_congestion_cost:.4f})")

    # ── density shift delta ────────────────────────────────────────────
    mover2 = 2
    nx, ny = 6.0, 3.0
    max_cells = max_cells_per_shift(macro_dims, bin_w, bin_h)
    cell_rs = np.zeros(max_cells, dtype=np.int32)
    cell_cs = np.zeros(max_cells, dtype=np.int32)
    new_ds  = np.zeros(max_cells, dtype=np.float64)
    d_den, d_cong, n_cells = density_grid_shift_delta_njit(
        mover2, nx, ny,
        macro_coords, macro_dims,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        density_grid,
        cell_rs, cell_cs, new_ds,
    )
    # ground truth: rebuild density grid with mover2 at (nx, ny) and diff costs
    test_coords = macro_coords.copy(); test_coords[mover2] = (nx, ny)
    py_dens2 = np.zeros_like(density_grid)
    compute_density_grid_njit(
        test_coords, macro_dims, num_macros,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        py_dens2,
    )
    expect_d   = float((py_dens2 ** 2).sum()) - base_density_cost
    expect_c   = float((py_dens2 ** 4).sum()) - base_congestion_cost
    assert math.isclose(d_den,  expect_d, abs_tol=1e-10), (d_den, expect_d)
    assert math.isclose(d_cong, expect_c, abs_tol=1e-10), (d_cong, expect_c)
    commit_density_grid_delta_njit(cell_rs, cell_cs, new_ds, n_cells, density_grid)
    assert np.allclose(density_grid, py_dens2, atol=1e-12), \
        "post-commit density grid disagrees with rebuild"
    print(f"density shift-delta kernel OK  (Δd={d_den:+.4f}, Δc={d_cong:+.4f}, "
          f"{n_cells} cells touched)")
    # commit the position change so subsequent tests see a consistent state
    macro_coords[mover2] = (nx, ny)

    # ── density swap delta ─────────────────────────────────────────────
    a2, b2 = 0, 4
    nxa, nya = macro_coords[b2, 0], macro_coords[b2, 1]
    nxb, nyb = macro_coords[a2, 0], macro_coords[a2, 1]
    max_cells_s = max_cells_per_swap(macro_dims, bin_w, bin_h)
    cell_rs2 = np.zeros(max_cells_s, dtype=np.int32)
    cell_cs2 = np.zeros(max_cells_s, dtype=np.int32)
    new_ds2  = np.zeros(max_cells_s, dtype=np.float64)
    pre_dens_cost = density_cost_from_grid_njit(density_grid)
    pre_cong_cost = congestion_cost_from_grid_njit(density_grid)
    d_den_s, d_cong_s, n_cells_s = density_grid_swap_delta_njit(
        a2, b2, nxa, nya, nxb, nyb,
        macro_coords, macro_dims,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        density_grid,
        cell_rs2, cell_cs2, new_ds2,
    )
    test_coords2 = macro_coords.copy()
    test_coords2[a2] = (nxa, nya); test_coords2[b2] = (nxb, nyb)
    py_dens3 = np.zeros_like(density_grid)
    compute_density_grid_njit(
        test_coords2, macro_dims, num_macros,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        py_dens3,
    )
    expect_d_s = float((py_dens3 ** 2).sum()) - pre_dens_cost
    expect_c_s = float((py_dens3 ** 4).sum()) - pre_cong_cost
    assert math.isclose(d_den_s,  expect_d_s, abs_tol=1e-10), (d_den_s, expect_d_s)
    assert math.isclose(d_cong_s, expect_c_s, abs_tol=1e-10), (d_cong_s, expect_c_s)
    commit_density_grid_delta_njit(cell_rs2, cell_cs2, new_ds2, n_cells_s, density_grid)
    assert np.allclose(density_grid, py_dens3, atol=1e-12)
    print(f"density swap-delta kernel OK   (Δd={d_den_s:+.4f}, Δc={d_cong_s:+.4f}, "
          f"{n_cells_s} cells touched)")
    # commit the swap into the coordinates so subsequent tests see a consistent state
    macro_coords[a2] = (nxa, nya); macro_coords[b2] = (nxb, nyb)

    # ── reshape delta ──────────────────────────────────────────────────
    mover3 = 4
    nx3, ny3, nw3, nh3 = 5.0, 5.0, 4.0, 1.0    # same area (4), different aspect
    pre_dens_cost = density_cost_from_grid_njit(density_grid)
    pre_cong_cost = congestion_cost_from_grid_njit(density_grid)
    d_den_r, d_cong_r, n_cells_r = density_grid_reshape_delta_njit(
        mover3, nx3, ny3, nw3, nh3,
        macro_coords, macro_dims,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        density_grid,
        cell_rs, cell_cs, new_ds,
    )
    test_coords3 = macro_coords.copy(); test_coords3[mover3] = (nx3, ny3)
    test_dims3   = macro_dims.copy();   test_dims3[mover3]   = (nw3, nh3)
    py_dens4 = np.zeros_like(density_grid)
    compute_density_grid_njit(
        test_coords3, test_dims3, num_macros,
        bin_w, bin_h, grid_num_rows, grid_num_cols,
        py_dens4,
    )
    expect_d_r = float((py_dens4 ** 2).sum()) - pre_dens_cost
    expect_c_r = float((py_dens4 ** 4).sum()) - pre_cong_cost
    assert math.isclose(d_den_r,  expect_d_r, abs_tol=1e-10), (d_den_r, expect_d_r)
    assert math.isclose(d_cong_r, expect_c_r, abs_tol=1e-10), (d_cong_r, expect_c_r)
    print(f"density reshape-delta kernel OK (Δd={d_den_r:+.4f}, Δc={d_cong_r:+.4f}, "
          f"{n_cells_r} cells touched)")

    # ── grid clear primitives ──────────────────────────────────────────
    # Pass the *original stamp coordinates* explicitly; macro_coords has
    # since been mutated by the swap test above.
    clear_macro_njit(
        spatial_grid, 0,
        1.0, 1.0,
        macro_dims[0, 0],   macro_dims[0, 1],
        bin_w, bin_h,
    )
    assert (spatial_grid == 0).sum() == 0, "macro 0 fully cleared from grid"
    print("clear_macro kernel OK")

    print("\nALL FAST_EVAL KERNELS OK")
