from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from fast_eval import (
    grid_has_overlap_for_macro,
    grid_insert_macro,
    init_grid,
    total_hpwl,
)
from state import HardState


def _clamp_center(cx: float, cy: float, hw: float, hh: float, cw: float, ch: float) -> Tuple[float, float]:
    if cx < hw:
        cx = hw
    elif cx > cw - hw:
        cx = cw - hw
    if cy < hh:
        cy = hh
    elif cy > ch - hh:
        cy = ch - hh
    return cx, cy


def grasp_initialize(
    st: HardState,
    seed: int,
    rows: int,
    cols: int,
    max_per_bin: int,
    gap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Semi-greedy randomized legal initializer for hard macros.

    Returns (pos_xy, grid, counts, wl_cost).
    """
    rng = np.random.default_rng(seed)

    pos = st.pos_xy.copy()
    half_w = st.half_w
    half_h = st.half_h
    sizes = st.size_wh

    # Bin sizes: uniform grid over canvas
    bin_w = st.canvas_w / cols
    bin_h = st.canvas_h / rows
    grid, counts = init_grid(rows, cols, max_per_bin)

    # Place fixed macros first at given positions (still hard macros)
    for i in range(st.n_hard):
        if not st.movable[i]:
            cx, cy = pos[i, 0], pos[i, 1]
            cx, cy = _clamp_center(cx, cy, half_w[i], half_h[i], st.canvas_w, st.canvas_h)
            pos[i, 0] = cx
            pos[i, 1] = cy
            grid_insert_macro(grid, counts, i, cx, cy, half_w[i], half_h[i], bin_w, bin_h, rows, cols)

    # Order movable by area (desc) for better packing
    movable_idx = np.where(st.movable)[0]
    areas = sizes[movable_idx, 0] * sizes[movable_idx, 1]
    order = movable_idx[np.argsort(-areas)]

    # For each macro: sample candidates (random + around connected macros), pick best WL among legal
    for i in order:
        hw = half_w[i]
        hh = half_h[i]

        # connectivity-guided center: average of neighbors in same nets
        cx0, cy0 = pos[i, 0], pos[i, 1]
        a = int(st.macro_net_ptr[i])
        b = int(st.macro_net_ptr[i + 1])
        if b > a:
            sx = 0.0
            sy = 0.0
            cnt = 0
            for kk in range(a, b):
                net_id = int(st.macro_nets[kk])
                na = int(st.net_ptr[net_id])
                nb = int(st.net_ptr[net_id + 1])
                for t in range(na, nb):
                    j = int(st.net_macros[t])
                    if j == i:
                        continue
                    sx += pos[j, 0]
                    sy += pos[j, 1]
                    cnt += 1
            if cnt > 0:
                cx0 = sx / cnt
                cy0 = sy / cnt

        best = None
        best_wl = float("inf")

        # Candidate pool
        num_cands = 48
        for k in range(num_cands):
            if k < 16:
                # global random
                cx = rng.random() * st.canvas_w
                cy = rng.random() * st.canvas_h
            else:
                # local gaussian around connectivity center
                scale = max(st.canvas_w, st.canvas_h) * (0.12 if k < 32 else 0.25)
                cx = cx0 + rng.normal(0.0, scale)
                cy = cy0 + rng.normal(0.0, scale)

            cx, cy = _clamp_center(cx, cy, hw, hh, st.canvas_w, st.canvas_h)
            if grid_has_overlap_for_macro(pos, half_w, half_h, grid, counts, i, cx, cy, bin_w, bin_h, rows, cols, gap):
                continue

            # Temporarily set pos and evaluate WL (still O(sum nets sizes) overall; ok for init)
            oldx, oldy = pos[i, 0], pos[i, 1]
            pos[i, 0], pos[i, 1] = cx, cy
            wl = total_hpwl(pos, st.net_ptr, st.net_macros)
            pos[i, 0], pos[i, 1] = oldx, oldy

            if wl < best_wl:
                best_wl = wl
                best = (cx, cy)

        if best is None:
            # fallback: spiral search around current
            cx, cy = _clamp_center(pos[i, 0], pos[i, 1], hw, hh, st.canvas_w, st.canvas_h)
            found = False
            step = max(hw * 2.0, hh * 2.0) * 0.5 + gap
            for r in range(1, 200):
                for dxm in range(-r, r + 1):
                    for dym in range(-r, r + 1):
                        if abs(dxm) != r and abs(dym) != r:
                            continue
                        tx, ty = _clamp_center(cx + dxm * step, cy + dym * step, hw, hh, st.canvas_w, st.canvas_h)
                        if not grid_has_overlap_for_macro(
                            pos, half_w, half_h, grid, counts, i, tx, ty, bin_w, bin_h, rows, cols, gap
                        ):
                            best = (tx, ty)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        cx, cy = best if best is not None else (hw, hh)
        pos[i, 0] = cx
        pos[i, 1] = cy
        grid_insert_macro(grid, counts, i, cx, cy, hw, hh, bin_w, bin_h, rows, cols)

    wl_cost = float(total_hpwl(pos, st.net_ptr, st.net_macros))
    return pos, grid, counts, wl_cost

