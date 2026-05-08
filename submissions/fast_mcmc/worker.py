from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np

from fast_eval import (
    delta_hpwl_for_macro_move,
    grid_has_overlap_for_macro,
    grid_insert_macro,
    grid_remove_macro,
    hpwl_for_net,
    total_hpwl,
)
from initialization import grasp_initialize
from state import HardState


class WorkerConfig:
    __slots__ = ("seed", "time_limit_s", "rows", "cols", "max_per_bin", "gap")

    def __init__(self, seed: int, time_limit_s: float, rows: int, cols: int, max_per_bin: int, gap: float):
        self.seed = int(seed)
        self.time_limit_s = float(time_limit_s)
        self.rows = int(rows)
        self.cols = int(cols)
        self.max_per_bin = int(max_per_bin)
        self.gap = float(gap)


def _clamp(cx: float, cy: float, hw: float, hh: float, cw: float, ch: float) -> Tuple[float, float]:
    if cx < hw:
        cx = hw
    elif cx > cw - hw:
        cx = cw - hw
    if cy < hh:
        cy = hh
    elif cy > ch - hh:
        cy = ch - hh
    return cx, cy


def run_worker(st: HardState, cfg: WorkerConfig) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    pos, grid, counts, wl = grasp_initialize(
        st, seed=cfg.seed, rows=cfg.rows, cols=cfg.cols, max_per_bin=cfg.max_per_bin, gap=cfg.gap
    )

    bin_w = st.canvas_w / cfg.cols
    bin_h = st.canvas_h / cfg.rows

    movable_idx = np.where(st.movable)[0]
    if len(movable_idx) == 0:
        return {"pos": pos, "wl": float(wl)}

    # Biased macro selection weights: start uniform; refresh periodically via net spans
    weights = np.ones(len(movable_idx), dtype=np.float64)
    weights /= weights.sum()

    def refresh_weights():
        # approximate "badness" by sum of connected net hpwl
        w = np.zeros(len(movable_idx), dtype=np.float64)
        for t, m in enumerate(movable_idx):
            a = int(st.macro_net_ptr[m])
            b = int(st.macro_net_ptr[m + 1])
            s = 0.0
            for kk in range(a, b):
                net_id = int(st.macro_nets[kk])
                s += float(hpwl_for_net(pos, st.net_ptr, st.net_macros, net_id))
            # fallback if no nets: small weight
            w[t] = max(1e-6, s)
        # If s isn't computed (above), use local degree as proxy
        if not np.isfinite(w).all() or w.max() <= 0:
            for t, m in enumerate(movable_idx):
                deg = int(st.macro_net_ptr[m + 1] - st.macro_net_ptr[m])
                w[t] = float(max(1, deg))
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
            return w
        return weights

    # Temperature schedule
    canvas = max(st.canvas_w, st.canvas_h)
    T0 = canvas * 0.12
    Tend = canvas * 0.0008

    start = time.time()
    best_pos = pos.copy()
    best_wl = float(total_hpwl(pos, st.net_ptr, st.net_macros))
    cur_wl = best_wl

    it = 0
    while True:
        now = time.time()
        if now - start >= cfg.time_limit_s:
            break

        it += 1
        frac = min(1.0, (now - start) / max(1e-9, cfg.time_limit_s))
        T = T0 * ((Tend / T0) ** frac)

        if it % 4000 == 0:
            weights = refresh_weights()

        # pick macro (biased)
        mi = int(rng.choice(len(movable_idx), p=weights))
        m = int(movable_idx[mi])

        oldx = float(pos[m, 0])
        oldy = float(pos[m, 1])
        hw = float(st.half_w[m])
        hh = float(st.half_h[m])

        move_type = rng.random()
        if move_type < 0.70:
            # SHIFT
            sigma = (0.20 + 0.80 * (1.0 - frac)) * T
            nx = oldx + rng.normal(0.0, sigma)
            ny = oldy + rng.normal(0.0, sigma)
        elif move_type < 0.92:
            # JUMP (global)
            nx = rng.random() * st.canvas_w
            ny = rng.random() * st.canvas_h
        else:
            # SWAP with another movable macro
            j = int(movable_idx[int(rng.integers(0, len(movable_idx)))])
            if j == m:
                continue
            j_oldx = float(pos[j, 0])
            j_oldy = float(pos[j, 1])
            j_hw = float(st.half_w[j])
            j_hh = float(st.half_h[j])

            nx, ny = _clamp(j_oldx, j_oldy, hw, hh, st.canvas_w, st.canvas_h)
            j_nx, j_ny = _clamp(oldx, oldy, j_hw, j_hh, st.canvas_w, st.canvas_h)

            # overlap check
            if grid_has_overlap_for_macro(pos, st.half_w, st.half_h, grid, counts, m, nx, ny, bin_w, bin_h, cfg.rows, cfg.cols, cfg.gap):
                continue
            if grid_has_overlap_for_macro(pos, st.half_w, st.half_h, grid, counts, j, j_nx, j_ny, bin_w, bin_h, cfg.rows, cfg.cols, cfg.gap):
                continue

            # wl delta: two moves
            d1 = float(
                delta_hpwl_for_macro_move(pos, st.net_ptr, st.net_macros, st.macro_net_ptr, st.macro_nets, m, nx, ny)
            )
            d2 = float(
                delta_hpwl_for_macro_move(pos, st.net_ptr, st.net_macros, st.macro_net_ptr, st.macro_nets, j, j_nx, j_ny)
            )
            d = d1 + d2
            accept = d <= 0.0 or rng.random() < math.exp(-d / max(1e-12, T))
            if not accept:
                continue

            # apply swap: update grid for both
            grid_remove_macro(grid, counts, m, oldx, oldy, hw, hh, bin_w, bin_h, cfg.rows, cfg.cols)
            grid_remove_macro(grid, counts, j, j_oldx, j_oldy, j_hw, j_hh, bin_w, bin_h, cfg.rows, cfg.cols)
            pos[m, 0], pos[m, 1] = nx, ny
            pos[j, 0], pos[j, 1] = j_nx, j_ny
            grid_insert_macro(grid, counts, m, nx, ny, hw, hh, bin_w, bin_h, cfg.rows, cfg.cols)
            grid_insert_macro(grid, counts, j, j_nx, j_ny, j_hw, j_hh, bin_w, bin_h, cfg.rows, cfg.cols)

            cur_wl += d
            if cur_wl < best_wl:
                best_wl = float(cur_wl)
                best_pos = pos.copy()
            continue

        nx, ny = _clamp(float(nx), float(ny), hw, hh, st.canvas_w, st.canvas_h)
        if grid_has_overlap_for_macro(pos, st.half_w, st.half_h, grid, counts, m, nx, ny, bin_w, bin_h, cfg.rows, cfg.cols, cfg.gap):
            continue

        d = float(
            delta_hpwl_for_macro_move(pos, st.net_ptr, st.net_macros, st.macro_net_ptr, st.macro_nets, m, nx, ny)
        )
        accept = d <= 0.0 or rng.random() < math.exp(-d / max(1e-12, T))
        if not accept:
            continue

        # apply move
        grid_remove_macro(grid, counts, m, oldx, oldy, hw, hh, bin_w, bin_h, cfg.rows, cfg.cols)
        pos[m, 0], pos[m, 1] = nx, ny
        grid_insert_macro(grid, counts, m, nx, ny, hw, hh, bin_w, bin_h, cfg.rows, cfg.cols)

        cur_wl += d
        if cur_wl < best_wl:
            best_wl = float(cur_wl)
            best_pos = pos.copy()

    return {"pos": best_pos, "wl": best_wl}


def worker_entry(args) -> Dict[str, object]:
    """
    Top-level multiprocessing entrypoint.

    Must live in an importable module (not the importlib-loaded submission entry file),
    otherwise pickling can fail under macro_place.evaluate's loader.
    """
    st, cfg = args
    return run_worker(st, cfg)

