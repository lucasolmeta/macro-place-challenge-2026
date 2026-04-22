from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir


def _load_plc_for_benchmark_name(name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45_map.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d
            / "netlist"
            / "output_CT_Grouping"
        )
        netlist_file = base / "netlist.pb.txt"
        plc_file = base / "initial.plc"
        if netlist_file.exists() and plc_file.exists():
            _, plc = load_benchmark(str(netlist_file), str(plc_file), name=name)
            return plc
    return None


def _extract_nets_hard_only(plc, hard_macro_names: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (net_ptr, net_macros) adjacency for hard macros only.
    Each net is a list of hard-macro indices appearing in that net (unique, >=2 only).
    """
    name_to_hidx = {n: i for i, n in enumerate(hard_macro_names)}

    nets: list[np.ndarray] = []
    for driver_pin, sink_pins in plc.nets.items():
        macros = set()
        for pin in [driver_pin] + sink_pins:
            parent = pin.split("/")[0]
            hidx = name_to_hidx.get(parent)
            if hidx is not None:
                macros.add(hidx)
        if len(macros) >= 2:
            nets.append(np.fromiter(sorted(macros), dtype=np.int32))

    if not nets:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)

    sizes = np.fromiter((len(n) for n in nets), dtype=np.int32)
    net_ptr = np.empty(len(nets) + 1, dtype=np.int32)
    net_ptr[0] = 0
    np.cumsum(sizes, out=net_ptr[1:])
    net_macros = np.empty(int(net_ptr[-1]), dtype=np.int32)

    off = 0
    for n in nets:
        net_macros[off : off + len(n)] = n
        off += len(n)

    return net_ptr, net_macros


def _invert_nets(net_ptr: np.ndarray, net_macros: np.ndarray, num_macros: int):
    counts = np.zeros(num_macros, dtype=np.int32)
    for k in range(len(net_macros)):
        counts[net_macros[k]] += 1

    macro_net_ptr = np.empty(num_macros + 1, dtype=np.int32)
    macro_net_ptr[0] = 0
    np.cumsum(counts, out=macro_net_ptr[1:])
    macro_nets = np.empty(int(macro_net_ptr[-1]), dtype=np.int32)

    write = macro_net_ptr[:-1].copy()
    for net_id in range(len(net_ptr) - 1):
        a = int(net_ptr[net_id])
        b = int(net_ptr[net_id + 1])
        for k in range(a, b):
            m = int(net_macros[k])
            macro_nets[write[m]] = net_id
            write[m] += 1

    return macro_net_ptr, macro_nets


@dataclass(frozen=True)
class HardState:
    name: str
    canvas_w: float
    canvas_h: float
    n_hard: int
    pos_xy: np.ndarray  # float64 (n_hard,2) centers
    size_wh: np.ndarray  # float64 (n_hard,2)
    movable: np.ndarray  # bool (n_hard,)
    half_w: np.ndarray  # float64 (n_hard,)
    half_h: np.ndarray  # float64 (n_hard,)
    net_ptr: np.ndarray  # int32 (n_nets+1,)
    net_macros: np.ndarray  # int32 (sum_deg,)
    macro_net_ptr: np.ndarray  # int32 (n_hard+1,)
    macro_nets: np.ndarray  # int32 (sum_deg,)


def build_hard_state(benchmark: Benchmark) -> Tuple[HardState, Optional[object]]:
    plc = _load_plc_for_benchmark_name(benchmark.name)
    if plc is None:
        raise RuntimeError(
            f"fast_mcmc: could not load PlacementCost (plc) for benchmark '{benchmark.name}'. "
            "Ensure the MacroPlacement submodule is initialized."
        )

    n_hard = int(benchmark.num_hard_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    pos_xy = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64, copy=True)
    size_wh = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=True)
    movable = (
        (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n_hard]
        .detach()
        .cpu()
        .numpy()
        .astype(bool, copy=False)
    )
    half_w = size_wh[:, 0] / 2.0
    half_h = size_wh[:, 1] / 2.0

    # plc.hard_macro_indices maps benchmark hard macro index -> plc.modules_w_pins index
    hard_macro_names = [plc.modules_w_pins[idx].get_name() for idx in plc.hard_macro_indices]
    net_ptr, net_macros = _extract_nets_hard_only(plc, hard_macro_names)
    macro_net_ptr, macro_nets = _invert_nets(net_ptr, net_macros, n_hard)

    st = HardState(
        name=str(benchmark.name),
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        n_hard=n_hard,
        pos_xy=pos_xy,
        size_wh=size_wh,
        movable=movable,
        half_w=half_w,
        half_h=half_h,
        net_ptr=net_ptr,
        net_macros=net_macros,
        macro_net_ptr=macro_net_ptr,
        macro_nets=macro_nets,
    )
    return st, plc


def hard_to_full_placement(benchmark: Benchmark, hard_xy: np.ndarray) -> torch.Tensor:
    full = benchmark.macro_positions.clone()
    full[: benchmark.num_hard_macros] = torch.tensor(hard_xy, dtype=torch.float32)
    return full

