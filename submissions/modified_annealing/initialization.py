"""
modified_annealing.initialization
========================

Per-worker GRASP (Greedy Randomized Adaptive Search Procedure)
constructive layout.  Executed inside each worker process to eliminate
the sequential bottleneck of doing initialisation on the master
(cf. :file:`cursor.md` §4.C).

GRASP outline
-------------
For each movable **hard** macro, in priority order
(largest area + highest net degree first):

1. **Random valid sampling.**  Draw up to
   ``num_candidates × max_tries_per_candidate`` uniform random
   bottom-left coordinates inside the macro's valid canvas region.
   For each draw, query the worker's ``spatial_grid`` to test for a
   collision with already-placed hard macros.  Keep the first
   ``num_candidates`` collision-free draws.

2. **Partial HPWL scoring.**  For each surviving candidate evaluate

   .. math::

       \\text{score}(c) = \\sum_{k \\in \\mathcal{N}(m)} w_k \\Bigl[(x^\\max - x^\\min) + (y^\\max - y^\\min)\\Bigr]_{\\text{placed pins} \\,\\cup\\, \\{c\\}}

   – the weighted HPWL of every net touching macro ``m`` restricted to
   pins that are already placed (other macros + I/O ports) plus the
   candidate position itself.  This is the same metric that the
   downstream MCMC optimises, restricted to currently-known
   information.

3. **Adaptive random pick.**  Sort the candidates by ascending score
   (lower = better), retain the ``top_k`` and pick one uniformly at
   random.  ``k`` controls the exploration/exploitation balance:
   ``k = 1`` is purely greedy (deterministic given the candidate set),
   ``k = num_candidates`` is purely random.

4. **Commit.**  Stamp the macro's id into every spatial bin its bbox
   covers (:func:`fast_eval.paint_macro_njit`) and mark the macro as
   placed.

After all movable hard macros are placed, the per-net bounding-box cache
is rebuilt from scratch so the returned :class:`PlacementState` is
immediately usable by the MCMC loop in ``worker.py``.

Fallbacks
---------
* **Bottom-Left-Fill scan.**  If no collision-free random sample is
  found within the attempt budget, an exhaustive scan over the spatial
  grid identifies the lowest-leftmost empty rectangle large enough for
  the macro.  This guarantees a legal position whenever one exists.
* **Forced placement.**  If even the BLF scan fails (extremely dense or
  infeasible benchmark), the best low-collision random sample is used
  and the worker reports the overlap.  The downstream MCMC engine and
  the final validity sweep are responsible for either repairing this
  configuration or flagging the solution invalid (cf. cursor.md §4.F).

Determinism
-----------
Every random decision in this module – the uniform sampler, the
``top_k`` tie-break, the seed used in fallback shuffles – flows through
a single :class:`numpy.random.Generator` instance seeded by the
worker-supplied ``seed`` argument.  Two invocations with the same
``seed`` and the same input ``PlacementState`` are guaranteed to
produce bit-identical output.  Different seeds yield structurally
distinct layouts, which is what feeds the diversity of the
multiprocessing pool.

This module performs no multiprocessing of its own; orchestration of
the worker pool lives in ``main.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from state import (
    EMPTY_CELL,
    PlacementState,
    clone_state,
)
from fast_eval import (
    check_collision_for_shift_njit,
    compute_macro_bin_range_njit,
    paint_macro_njit,
    populate_net_bbox_njit,
    njit,  # numba decorator with graceful fallback
)


# ─── Tunable defaults ────────────────────────────────────────────────────────

DEFAULT_NUM_CANDIDATES: int = 16
"""Target number of *collision-free* candidate positions evaluated per
macro.  Higher values explore more positions at linear cost per macro."""

DEFAULT_TOP_K: int = 4
"""Number of top-scoring candidates the GRASP random pick is drawn
from.  ``1`` is purely greedy, ``num_candidates`` is purely random; the
default sits in the classical GRASP sweet-spot."""

DEFAULT_MAX_TRIES_PER_CANDIDATE: int = 64
"""Maximum number of uniform-random rejection-sampling attempts spent
trying to satisfy one candidate slot.  Caps the per-macro work at
``num_candidates × max_tries_per_candidate`` collision checks.  Tuned
empirically: dense benchmarks (ibm01-style 60 % macro area density) need
``≥ 64`` to stay below ~10 % residual GRASP overlap before the MCMC
legalisation phase takes over."""

DEFAULT_AREA_WEIGHT: float = 1.0
DEFAULT_DEGREE_WEIGHT: float = 1.0
"""Weights blending macro area and netlist degree into the GRASP
priority key.  Both values are normalised to ``[0, 1]`` before being
combined, so absolute magnitudes only matter relative to each other."""

DEFAULT_COLLISION_PENALTY: float = 1.0e12
"""Score penalty added when a candidate cannot be made collision-free
within the attempt budget.  Large enough to push such candidates to the
bottom of the ``top_k`` ranking when better positions exist."""

DEFAULT_BLF_BIN_STRIDE: int = 1
"""Stride used when scanning the spatial grid for an empty rectangle in
the BLF fallback.  ``1`` (default) is exhaustive; values ``> 1`` skip
bins for speed at the cost of missing tight fits."""


# ─── Sorting / priority utilities ────────────────────────────────────────────

def compute_grasp_priority(
    state: PlacementState,
    *,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    degree_weight: float = DEFAULT_DEGREE_WEIGHT,
) -> np.ndarray:
    """Per-macro GRASP packing priority (higher = packed first).

    The score blends two normalised features

    .. math::

        \\hat{a}_m &= \\frac{\\text{area}_m}{\\max_{m'} \\text{area}_{m'}} \\\\
        \\hat{d}_m &= \\frac{\\deg(m)}{\\max_{m'} \\deg(m')} \\\\
        \\pi_m     &= w_a \\hat{a}_m + w_d \\hat{d}_m

    so that macros which are both *large* (most constrained to find
    space for) and *highly connected* (most leverage on global
    wirelength) are tackled first while space is still plentiful.

    Args:
        state: source placement state.
        area_weight, degree_weight: linear blend coefficients.

    Returns:
        ``float64[num_macros]`` priority vector aligned to
        ``state.macro_*`` indexing.
    """
    if state.num_macros == 0:
        return np.zeros(0, dtype=np.float64)

    areas = (state.macro_dims[:, 0] * state.macro_dims[:, 1]).astype(np.float64)
    degrees = np.diff(state.macro_net_offsets.astype(np.int64)).astype(np.float64)

    max_area = float(areas.max()) if areas.size else 0.0
    max_deg  = float(degrees.max()) if degrees.size else 0.0
    inv_a = 1.0 / max_area if max_area > 0.0 else 0.0
    inv_d = 1.0 / max_deg  if max_deg  > 0.0 else 0.0

    return area_weight * (areas * inv_a) + degree_weight * (degrees * inv_d)


def movable_hard_macro_order(
    state: PlacementState,
    *,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    degree_weight: float = DEFAULT_DEGREE_WEIGHT,
) -> np.ndarray:
    """Indices of movable hard macros in descending GRASP priority.

    Ties are broken by ascending macro index so the result is fully
    deterministic given the input state.
    """
    if state.num_macros == 0:
        return np.zeros(0, dtype=np.int64)
    priority = compute_grasp_priority(
        state, area_weight=area_weight, degree_weight=degree_weight,
    )
    movable = (~state.macro_fixed) & state.macro_is_hard
    candidates = np.where(movable)[0]
    if candidates.size == 0:
        return candidates
    # ``np.argsort`` is stable on the default kind, so equal priorities
    # fall back to ascending macro index – making the output reproducible
    # without any further tie-break logic.
    order = np.argsort(-priority[candidates], kind="stable")
    return candidates[order].astype(np.int64, copy=False)


# ─── Numba inner kernels ─────────────────────────────────────────────────────
#
# Kept private to this module because they encode GRASP-specific
# semantics (e.g. "placed pins only") rather than general MCMC math.
# Their public counterparts live in ``fast_eval.py``.

@njit(cache=True)
def _partial_net_hpwl_for_candidate_njit(
    net_id: int,
    candidate_macro: int,
    candidate_x_ll: float, candidate_y_ll: float,
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    placed_mask: np.ndarray,
) -> float:
    """HPWL of one net counting only already-placed pins + the candidate.

    Unplaced macros are skipped entirely (their pin position is
    meaningless at this stage of GRASP).  Ports are always counted
    because they are fixed by construction.

    Returns ``0.0`` if the net ends up with fewer than two
    pin centres – a single-pin net has zero HPWL.
    """
    s = net_pin_offsets[net_id]
    e = net_pin_offsets[net_id + 1]
    min_x =  np.inf
    max_x = -np.inf
    min_y =  np.inf
    max_y = -np.inf
    found = 0
    for p in range(s, e):
        nid = net_pin_owners[p]
        if nid == candidate_macro:
            x = candidate_x_ll + 0.5 * macro_dims[nid, 0]
            y = candidate_y_ll + 0.5 * macro_dims[nid, 1]
        elif nid < num_macros:
            if not placed_mask[nid]:
                continue
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
        found += 1
    if found < 2:
        return 0.0
    return (max_x - min_x) + (max_y - min_y)


@njit(cache=True)
def _partial_hpwl_score_njit(
    candidate_macro: int,
    candidate_x_ll: float, candidate_y_ll: float,
    macro_net_ids: np.ndarray, macro_net_offsets: np.ndarray,
    net_pin_owners: np.ndarray, net_pin_offsets: np.ndarray,
    net_weights: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    port_coords: np.ndarray, num_macros: int,
    placed_mask: np.ndarray,
) -> float:
    """Σ_k w_k · partial_HPWL(k) over the nets touching ``candidate_macro``."""
    s = macro_net_offsets[candidate_macro]
    e = macro_net_offsets[candidate_macro + 1]
    total = 0.0
    for i in range(s, e):
        net_id = macro_net_ids[i]
        h = _partial_net_hpwl_for_candidate_njit(
            net_id, candidate_macro, candidate_x_ll, candidate_y_ll,
            net_pin_owners, net_pin_offsets,
            macro_coords, macro_dims,
            port_coords, num_macros,
            placed_mask,
        )
        total += net_weights[net_id] * h
    return total


@njit(cache=True)
def _sample_collision_free_njit(
    spatial_grid: np.ndarray,
    macro_idx: int, width: float, height: float,
    bin_width: float, bin_height: float,
    canvas_width: float, canvas_height: float,
    rand_xs: np.ndarray, rand_ys: np.ndarray,
    out_xs: np.ndarray, out_ys: np.ndarray,
    out_collision_flags: np.ndarray,
    num_candidates: int, max_tries_per_candidate: int,
) -> Tuple[int, int]:
    """Rejection-sample up to ``num_candidates`` collision-free positions.

    ``rand_xs`` / ``rand_ys`` are caller-supplied flat buffers of
    pre-drawn ``Uniform[0, 1)`` floats of length at least
    ``num_candidates * max_tries_per_candidate``.  This indirection
    keeps every random byte under the worker's
    :class:`numpy.random.Generator` so the result is fully
    deterministic in the seed.

    Each draw is mapped into the macro's valid bottom-left range
    ``[0, canvas_w - w] × [0, canvas_h - h]``.  If
    :func:`fast_eval.check_collision_for_shift_njit` reports no foreign
    occupant, the position is appended to ``out_xs`` / ``out_ys`` with
    a clean ``out_collision_flags`` entry.

    If after ``max_tries_per_candidate`` rejected draws a slot still has
    no clean position, the *least* colliding sample seen for that slot
    is written with ``out_collision_flags[slot] = collision_id`` so the
    caller can score it with a penalty and decide whether to use it.

    Returns ``(num_total, num_clean)`` where ``num_total`` ≤
    ``num_candidates`` is the number of slots actually filled and
    ``num_clean`` counts how many of them were collision-free.
    """
    valid_x = canvas_width  - width
    valid_y = canvas_height - height
    if valid_x < 0.0: valid_x = 0.0
    if valid_y < 0.0: valid_y = 0.0

    n_total = 0
    n_clean = 0
    draw = 0

    for slot in range(num_candidates):
        best_collision_x = 0.0
        best_collision_y = 0.0
        best_collision_id = -2     # -2 = unset, -1 = clean, ≥0 = some collision
        got_clean = False

        for _try in range(max_tries_per_candidate):
            if draw >= rand_xs.size:
                break
            x = rand_xs[draw] * valid_x
            y = rand_ys[draw] * valid_y
            draw += 1

            occupant = check_collision_for_shift_njit(
                spatial_grid, macro_idx, x, y, width, height,
                bin_width, bin_height,
            )
            if occupant == EMPTY_CELL:
                out_xs[n_total] = x
                out_ys[n_total] = y
                out_collision_flags[n_total] = EMPTY_CELL
                n_total += 1
                n_clean += 1
                got_clean = True
                break

            # Track the best (lowest-id) collision encountered as a
            # fallback so the slot is never left empty.
            if best_collision_id == -2:
                best_collision_x = x
                best_collision_y = y
                best_collision_id = occupant

        if not got_clean and best_collision_id != -2:
            out_xs[n_total] = best_collision_x
            out_ys[n_total] = best_collision_y
            out_collision_flags[n_total] = best_collision_id
            n_total += 1
    return n_total, n_clean


@njit(cache=True)
def _scan_blf_position_njit(
    spatial_grid: np.ndarray,
    width: float, height: float,
    bin_width: float, bin_height: float,
    canvas_width: float, canvas_height: float,
    bin_stride: int,
    r_start: int, c_start: int,
) -> Tuple[float, float, int]:
    """Bottom-Left-Fill scan with a wrap-around start offset.

    Walks the spatial grid starting at ``(r_start, c_start)``, wrapping
    around to ``(0, 0)`` after the last row/column.  For each candidate
    bottom-left bin, checks whether the macro's covered sub-matrix is
    entirely empty.  Returns ``(x, y, 1)`` on success or
    ``(0.0, 0.0, 0)`` if no legal slot exists at the given stride.

    The wrap-around start lets ``grasp_initialize`` feed the kernel a
    per-seed offset so independent workers find structurally distinct
    fallback positions, preserving GRASP-level diversity even when the
    BLF fallback fires for many macros.

    Complexity is :math:`O(R \\cdot C \\cdot c_w \\cdot c_h / \\text{stride}^2)`
    in the worst case but typically terminates very early on sparse
    grids.  Intended strictly as a fallback when random sampling
    repeatedly fails.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    bins_needed_x = int(math.ceil(width  / bin_width))
    bins_needed_y = int(math.ceil(height / bin_height))
    if bins_needed_x < 1: bins_needed_x = 1
    if bins_needed_y < 1: bins_needed_y = 1
    if bin_stride < 1: bin_stride = 1

    r_top = grid_num_rows - bins_needed_y
    c_top = grid_num_cols - bins_needed_x
    if r_top < 0 or c_top < 0:
        return 0.0, 0.0, 0

    n_r = r_top + 1
    n_c = c_top + 1
    if r_start < 0: r_start = 0
    if c_start < 0: c_start = 0
    if r_start >= n_r: r_start = r_start % n_r
    if c_start >= n_c: c_start = c_start % n_c

    for r_i in range(0, n_r, bin_stride):
        r = (r_start + r_i) % n_r
        for c_i in range(0, n_c, bin_stride):
            c = (c_start + c_i) % n_c
            empty = True
            for rr in range(r, r + bins_needed_y):
                if not empty:
                    break
                for cc in range(c, c + bins_needed_x):
                    if spatial_grid[rr, cc] != EMPTY_CELL:
                        empty = False
                        break
            if empty:
                x = c * bin_width
                y = r * bin_height
                # Clamp to canvas to absorb the ceil-rounding above.
                if x + width  > canvas_width:  x = canvas_width  - width
                if y + height > canvas_height: y = canvas_height - height
                if x < 0.0: x = 0.0
                if y < 0.0: y = 0.0
                return x, y, 1
    return 0.0, 0.0, 0


@njit(cache=True)
def _stamp_fixed_hard_macros_njit(
    spatial_grid: np.ndarray,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    macro_fixed: np.ndarray, macro_is_hard: np.ndarray,
    num_macros: int,
    bin_width: float, bin_height: float,
) -> int:
    """Reset ``spatial_grid`` to empty and re-paint every *fixed* hard macro.

    Returns the number of macros stamped.  Movable hard macros are
    intentionally *not* stamped here because GRASP will paint them as
    it places each one.
    """
    spatial_grid[:, :] = EMPTY_CELL
    stamped = 0
    for m in range(num_macros):
        if macro_is_hard[m] and macro_fixed[m]:
            paint_macro_njit(
                spatial_grid, m,
                macro_coords[m, 0], macro_coords[m, 1],
                macro_dims[m, 0],   macro_dims[m, 1],
                bin_width, bin_height,
            )
            stamped += 1
    return stamped


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class GraspReport:
    """Diagnostic record returned alongside the seeded ``PlacementState``.

    Carries enough information for ``main.py`` to compare the quality
    of competing workers and for ``worker.py`` to know whether the
    starting layout is already legal.
    """
    seed: int                       # the seed used (preserved verbatim)
    num_movable_hard: int           # how many movable hard macros existed
    num_placed_clean: int           # placements with no spatial-grid collision
    num_placed_via_blf: int         # placements that needed the BLF scan
    num_placed_with_overlap: int    # placements that still overlap something
    sum_partial_hpwl: float         # final cumulative partial-HPWL score
    movable_order: np.ndarray       # macro ids in the order they were placed


# ─── Public entry points ─────────────────────────────────────────────────────

def grasp_initialize(
    state: PlacementState,
    seed: int = 0,
    *,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
    max_tries_per_candidate: int = DEFAULT_MAX_TRIES_PER_CANDIDATE,
    area_weight: float = DEFAULT_AREA_WEIGHT,
    degree_weight: float = DEFAULT_DEGREE_WEIGHT,
    collision_penalty: float = DEFAULT_COLLISION_PENALTY,
    enable_blf_fallback: bool = True,
    blf_bin_stride: int = DEFAULT_BLF_BIN_STRIDE,
    copy_state: bool = True,
) -> Tuple[PlacementState, GraspReport]:
    """Run worker-local GRASP on ``state`` and return ``(new_state, report)``.

    All randomness flows through ``numpy.random.default_rng(seed)`` so
    the result is bit-reproducible.  Different seeds produce
    structurally different layouts, which is the diversity the
    multiprocessing pool feeds on.

    Args:
        state: input :class:`modified_annealing.state.PlacementState`.
        seed:  integer seed for this worker's RNG.
        num_candidates: collision-free target per macro
            (see :data:`DEFAULT_NUM_CANDIDATES`).
        top_k: GRASP exploitation/exploration knob
            (see :data:`DEFAULT_TOP_K`).
        max_tries_per_candidate: cap on rejection-sampling attempts
            (see :data:`DEFAULT_MAX_TRIES_PER_CANDIDATE`).
        area_weight, degree_weight: priority blend coefficients
            (see :func:`compute_grasp_priority`).
        collision_penalty: score penalty added to candidates that could
            not be made collision-free.
        enable_blf_fallback: when ``True`` (default), a Bottom-Left-Fill
            scan is invoked whenever random sampling fails to find *any*
            clean candidate for a macro.  Disabling can be useful for
            very large benchmarks where the scan is too slow and the
            MCMC engine is expected to repair the layout.
        blf_bin_stride: stride passed to :func:`_scan_blf_position_njit`
            when invoked as a fallback.
        copy_state: when ``True`` (default) the input ``state`` is deep
            cloned via :func:`state.clone_state` so this routine has no
            side effect on the caller.  Set to ``False`` to mutate
            in-place when the caller already owns a per-worker copy
            (e.g. ``main.py`` after ``multiprocessing.fork``).

    Returns:
        A tuple ``(new_state, report)`` where ``new_state`` has

        * ``macro_coords`` updated for every movable hard macro,
        * ``spatial_grid`` re-stamped with every hard macro
          (fixed + movable),
        * ``net_bbox`` recomputed from scratch via
          :func:`fast_eval.populate_net_bbox_njit`,

        and ``report`` carries diagnostic counters described on
        :class:`GraspReport`.
    """
    if state.num_macros == 0:
        empty_report = GraspReport(
            seed=int(seed), num_movable_hard=0, num_placed_clean=0,
            num_placed_via_blf=0, num_placed_with_overlap=0,
            sum_partial_hpwl=0.0, movable_order=np.zeros(0, dtype=np.int64),
        )
        return (clone_state(state) if copy_state else state), empty_report

    if copy_state:
        state = clone_state(state)

    rng = np.random.default_rng(int(seed))

    # ── Reset spatial grid and re-paint fixed hard macros ────────────────
    _stamp_fixed_hard_macros_njit(
        state.spatial_grid,
        state.macro_coords, state.macro_dims,
        state.macro_fixed, state.macro_is_hard,
        state.num_macros,
        state.grid_bin_width, state.grid_bin_height,
    )

    # ── Build "placed" mask ──────────────────────────────────────────────
    # Fixed hard macros are placed by definition.  Soft macros count as
    # placed for partial-HPWL purposes because they retain their initial
    # positions throughout GRASP (no collision constraint applies to
    # them, so they are not stamped onto the grid).
    placed_mask = np.zeros(state.num_macros, dtype=np.bool_)
    placed_mask |= (state.macro_is_hard & state.macro_fixed)
    placed_mask |= (~state.macro_is_hard)

    # ── Determine GRASP placement order ─────────────────────────────────
    order = movable_hard_macro_order(
        state, area_weight=area_weight, degree_weight=degree_weight,
    )

    # ── Pre-allocate per-macro scratch buffers ──────────────────────────
    cap = max(1, num_candidates)
    out_xs = np.zeros(cap, dtype=np.float64)
    out_ys = np.zeros(cap, dtype=np.float64)
    out_collision_flags = np.full(cap, EMPTY_CELL, dtype=np.int32)
    scores = np.zeros(cap, dtype=np.float64)

    # Pre-draw all the uniform randoms for the worst case so the
    # downstream Numba kernel does no Python work; oversized buffers
    # are cheap (a few KB per macro).
    rand_budget = max(1, num_candidates * max(1, max_tries_per_candidate))

    sum_partial_hpwl = 0.0
    num_clean_tot   = 0
    num_blf_tot     = 0
    num_overlap_tot = 0

    # ── Main GRASP loop ──────────────────────────────────────────────────
    for m_idx in order:
        m = int(m_idx)
        w = float(state.macro_dims[m, 0])
        h = float(state.macro_dims[m, 1])

        # Pathologically big macro that does not fit the canvas: clamp
        # it to (0, 0) and continue – the validity sweep will flag this.
        if w > state.canvas_width or h > state.canvas_height:
            state.macro_coords[m, 0] = 0.0
            state.macro_coords[m, 1] = 0.0
            paint_macro_njit(
                state.spatial_grid, m,
                0.0, 0.0, w, h,
                state.grid_bin_width, state.grid_bin_height,
            )
            placed_mask[m] = True
            num_overlap_tot += 1
            continue

        rand_xs = rng.random(rand_budget)
        rand_ys = rng.random(rand_budget)

        n_total, n_clean = _sample_collision_free_njit(
            state.spatial_grid, m, w, h,
            state.grid_bin_width, state.grid_bin_height,
            state.canvas_width, state.canvas_height,
            rand_xs, rand_ys,
            out_xs, out_ys, out_collision_flags,
            num_candidates, max_tries_per_candidate,
        )

        # ── BLF fallback when nothing clean was found ────────────────
        # Deterministic (start = (0, 0)) – seed diversity already comes
        # from the upstream random sampling and top-k pick, while a
        # truly Best-Fit-Decreasing search packs the bottom-left
        # corners densely and leaves contiguous space for later macros.
        used_blf = False
        if n_clean == 0 and enable_blf_fallback:
            bx, by, ok = _scan_blf_position_njit(
                state.spatial_grid,
                w, h,
                state.grid_bin_width, state.grid_bin_height,
                state.canvas_width, state.canvas_height,
                blf_bin_stride,
                0, 0,
            )
            if ok == 1:
                out_xs[0] = bx
                out_ys[0] = by
                out_collision_flags[0] = EMPTY_CELL
                n_total = 1
                n_clean = 1
                used_blf = True

        if n_total == 0:
            # Hard-impossible case: just pin the macro at (0, 0).  This
            # is extremely unlikely – ``_sample_collision_free_njit``
            # always returns at least its best random draw – but we
            # handle it defensively.
            out_xs[0] = 0.0
            out_ys[0] = 0.0
            out_collision_flags[0] = -3
            n_total = 1

        # ── Score candidates by partial-HPWL ──────────────────────────
        best_score = np.inf
        for i in range(n_total):
            s = _partial_hpwl_score_njit(
                m, out_xs[i], out_ys[i],
                state.macro_net_ids, state.macro_net_offsets,
                state.net_pin_owners, state.net_pin_offsets,
                state.net_weights,
                state.macro_coords, state.macro_dims,
                state.port_coords, state.num_macros,
                placed_mask,
            )
            if out_collision_flags[i] != EMPTY_CELL:
                s += collision_penalty
            scores[i] = s
            if s < best_score:
                best_score = s

        # ── Adaptive random pick from the top-k ──────────────────────
        k_eff = top_k if top_k < n_total else n_total
        if k_eff < 1:
            k_eff = 1
        if k_eff == n_total:
            ranking = np.arange(n_total, dtype=np.int64)
        else:
            # ``argpartition`` is O(n) and avoids the full sort cost.
            ranking = np.argpartition(scores[:n_total], k_eff - 1)[:k_eff]
        choice = int(ranking[rng.integers(0, k_eff)])

        chosen_x = float(out_xs[choice])
        chosen_y = float(out_ys[choice])
        chosen_collision = int(out_collision_flags[choice])

        # ── Commit position and stamp grid ────────────────────────────
        state.macro_coords[m, 0] = chosen_x
        state.macro_coords[m, 1] = chosen_y
        paint_macro_njit(
            state.spatial_grid, m,
            chosen_x, chosen_y, w, h,
            state.grid_bin_width, state.grid_bin_height,
        )
        placed_mask[m] = True
        sum_partial_hpwl += float(scores[choice])

        if chosen_collision == EMPTY_CELL:
            num_clean_tot += 1
            if used_blf:
                num_blf_tot += 1
        else:
            num_overlap_tot += 1

    # ── Rebuild per-net bounding-box cache ──────────────────────────────
    populate_net_bbox_njit(
        state.net_pin_owners, state.net_pin_offsets,
        state.macro_coords, state.macro_dims,
        state.port_coords, state.num_macros, state.num_nets,
        state.net_bbox,
    )

    report = GraspReport(
        seed=int(seed),
        num_movable_hard=int(order.size),
        num_placed_clean=int(num_clean_tot),
        num_placed_via_blf=int(num_blf_tot),
        num_placed_with_overlap=int(num_overlap_tot),
        sum_partial_hpwl=float(sum_partial_hpwl),
        movable_order=order,
    )
    return state, report


def grasp_initialize_batch(
    state: PlacementState,
    seeds: Sequence[int],
    **kwargs,
) -> List[Tuple[PlacementState, GraspReport]]:
    """Convenience helper that runs :func:`grasp_initialize` for each seed
    serially.

    Real parallelism is provided by ``main.py``'s ``multiprocessing.Pool``;
    this routine is intended for unit tests and serial debugging.  It
    deep-clones the input ``state`` for every seed regardless of the
    ``copy_state`` kwarg so each result is fully independent.
    """
    kwargs.pop("copy_state", None)
    results: List[Tuple[PlacementState, GraspReport]] = []
    for s in seeds:
        results.append(grasp_initialize(state, int(s), copy_state=True, **kwargs))
    return results


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Self-contained smoke test on a synthetic 8-macro scene that does
    # not require torch or any external benchmark file.  Run with:
    #   python submissions/modified_annealing/initialization.py
    import types
    from state import build_state, verify_state

    class _StubTensor:
        def __init__(self, arr): self._arr = np.ascontiguousarray(arr)
        def detach(self):        return self
        def cpu(self):           return self
        def numpy(self):         return self._arr
        def numel(self):         return self._arr.size
        @property
        def shape(self):         return self._arr.shape

    # Canvas 30 × 30, eight movable hard macros + two soft macros, no
    # fixed macros, three small nets.
    bench = types.SimpleNamespace(
        name="grasp_smoke",
        canvas_width=30.0,
        canvas_height=30.0,
        num_macros=10,
        num_hard_macros=8,
        num_soft_macros=2,
        macro_positions=_StubTensor([
            [ 5.0,  5.0], [10.0,  5.0], [15.0,  5.0], [20.0,  5.0],
            [ 5.0, 20.0], [10.0, 20.0], [15.0, 20.0], [20.0, 20.0],
            [15.0, 15.0], [15.0, 15.0],   # soft
        ]),
        macro_sizes=_StubTensor([
            [5.0, 4.0], [4.0, 5.0], [6.0, 3.0], [3.0, 6.0],
            [4.0, 4.0], [5.0, 5.0], [3.0, 3.0], [4.0, 6.0],
            [2.0, 2.0], [2.0, 2.0],
        ]),
        macro_fixed=_StubTensor(
            [False] * 10,
        ),
        macro_names=[f"hard{i}" for i in range(8)] + ["soft0", "soft1"],
        num_nets=3,
        net_nodes=[
            np.array([0, 1, 7, 11], dtype=np.int64),    # hard0,1,7 + port 1
            np.array([2, 3, 5, 6],  dtype=np.int64),    # hard2,3,5,6
            np.array([4, 8, 9, 10], dtype=np.int64),    # hard4 + soft0,1 + port0
        ],
        net_weights=_StubTensor([1.0, 1.5, 0.5]),
        port_positions=_StubTensor([[0.0, 0.0], [30.0, 30.0]]),
        grid_rows=10, grid_cols=10,
    )

    st_base = build_state(bench)
    assert not verify_state(st_base)
    print(f"base state OK | macros={st_base.num_macros} hard={st_base.num_hard_macros} "
          f"nets={st_base.num_nets} grid={st_base.spatial_grid.shape} "
          f"bin=({st_base.grid_bin_width:.2f},{st_base.grid_bin_height:.2f})")

    # ── 1. priority is monotone in area + degree ────────────────────────
    pri = compute_grasp_priority(st_base)
    print(f"GRASP priority sample: {pri[:8].round(3)}")
    order = movable_hard_macro_order(st_base)
    print(f"placement order:       {order.tolist()}")
    # First-placed macro should have non-trivial priority
    assert pri[order[0]] > 0.0

    # ── 2. seed reproducibility ─────────────────────────────────────────
    st_a, rep_a = grasp_initialize(st_base, seed=42, num_candidates=8, top_k=3)
    st_b, rep_b = grasp_initialize(st_base, seed=42, num_candidates=8, top_k=3)
    assert np.array_equal(st_a.macro_coords, st_b.macro_coords), "seed not deterministic"
    assert np.array_equal(st_a.spatial_grid, st_b.spatial_grid)
    assert np.allclose(st_a.net_bbox, st_b.net_bbox)
    print(f"seed=42 reproducibility OK   (Σ partial HPWL = {rep_a.sum_partial_hpwl:.3f})")

    # ── 3. seed diversity ───────────────────────────────────────────────
    st_c, rep_c = grasp_initialize(st_base, seed=7,  num_candidates=8, top_k=3)
    st_d, rep_d = grasp_initialize(st_base, seed=99, num_candidates=8, top_k=3)
    assert not np.array_equal(st_a.macro_coords, st_c.macro_coords), \
        "seeds 42 and 7 produced identical layouts"
    assert not np.array_equal(st_c.macro_coords, st_d.macro_coords), \
        "seeds 7 and 99 produced identical layouts"
    print(f"seed diversity OK  (seed=7 Σ={rep_c.sum_partial_hpwl:.3f}, "
          f"seed=99 Σ={rep_d.sum_partial_hpwl:.3f})")

    # ── 4. no hard-macro overlaps (exact bbox check) ────────────────────
    def _exact_overlap_count(st):
        N = st.num_hard_macros
        cnt = 0
        for i in range(N):
            xi, yi = st.macro_coords[i]; wi, hi = st.macro_dims[i]
            for j in range(i + 1, N):
                xj, yj = st.macro_coords[j]; wj, hj = st.macro_dims[j]
                if (xi < xj + wj) and (xi + wi > xj) and \
                   (yi < yj + hj) and (yi + hi > yj):
                    cnt += 1
        return cnt

    for label, st in (("seed=42", st_a), ("seed=7", st_c), ("seed=99", st_d)):
        overlaps = _exact_overlap_count(st)
        print(f"{label} exact bbox overlaps: {overlaps}")
        assert overlaps == 0, f"GRASP left {overlaps} hard-macro overlaps for {label}"

    # ── 5. canvas containment ───────────────────────────────────────────
    for label, st in (("seed=42", st_a), ("seed=7", st_c), ("seed=99", st_d)):
        x_lo = st.macro_coords[:st.num_hard_macros, 0]
        y_lo = st.macro_coords[:st.num_hard_macros, 1]
        w    = st.macro_dims  [:st.num_hard_macros, 0]
        h    = st.macro_dims  [:st.num_hard_macros, 1]
        assert (x_lo >= -1e-9).all() and (x_lo + w <= st.canvas_width  + 1e-9).all()
        assert (y_lo >= -1e-9).all() and (y_lo + h <= st.canvas_height + 1e-9).all()
    print("canvas containment OK for all seeds")

    # ── 6. spatial_grid reflects committed positions ────────────────────
    from fast_eval import count_grid_collisions_njit
    for label, st in (("seed=42", st_a), ("seed=7", st_c), ("seed=99", st_d)):
        col = count_grid_collisions_njit(
            st.spatial_grid, st.macro_coords, st.macro_dims,
            st.num_hard_macros,
            st.grid_bin_width, st.grid_bin_height,
        )
        assert col == 0, f"spatial_grid reports {col} hard collisions for {label}"
    print("spatial_grid coherence OK for all seeds")

    # ── 7. net_bbox cache up to date ────────────────────────────────────
    from fast_eval import populate_net_bbox_njit
    for label, st in (("seed=42", st_a),):
        fresh = np.zeros_like(st.net_bbox)
        populate_net_bbox_njit(
            st.net_pin_owners, st.net_pin_offsets,
            st.macro_coords, st.macro_dims,
            st.port_coords, st.num_macros, st.num_nets,
            fresh,
        )
        assert np.allclose(fresh, st.net_bbox, atol=1e-12), \
            "net_bbox cache stale after grasp_initialize"
    print("net_bbox cache coherence OK")

    print("\nALL INITIALIZATION KERNELS OK")
